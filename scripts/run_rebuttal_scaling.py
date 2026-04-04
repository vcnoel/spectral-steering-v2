import os
import sys
import subprocess
import json
from pathlib import Path
import torch

# Configuration for the scaling sweep
models = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama-3.1-8B"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral-7B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B"},
    {"id": "microsoft/Phi-3.5-mini-instruct", "name": "Phi-3.5-Mini"}
]

benchmarks = ["sycophancy", "gsm8k"] 
alphas = [-1.0, 0.0, 0.5] # Sharpening, Baseline, Smoothing
sample_size = 100 # Extract from 100, evaluate from 50 (syco)/20 (gsm) as per run_phase1 scaling

results_log = []

def run_cmd(cmd_list):
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    # We use subprocess.run and catch output to be cleaner
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed: {res.stderr}")
    return res

def main():
    script_path = Path(__file__).parent / "run_phase1.py"
    output_file = Path("scaling_results.json")
    
    # Initialize results if file exists to allow resuming or partial runs
    if output_file.exists():
        with open(output_file, "r") as f:
            try:
                results_log.extend(json.load(f))
            except:
                pass

    print("\n🚀 Starting Global Rebuttal Scaling Sweep (4-bit)...")
    
    for m in models:
        model_id = m["id"]
        model_name = m["name"]
        
        print(f"\n" + "="*60)
        print(f"🔥 Processing Model: {model_name} ({model_id})")
        print("="*60)

        # 1. Extract Eigenvectors
        print(f"\n--- 1. Extracting Eigenvectors ({sample_size} pairs) ---")
        run_cmd([sys.executable, str(script_path), "extract", "--model", model_id, "--n_pairs", str(sample_size), "--load_bit4"])
        
        # Load extracted stats to get layers and original SNR
        try:
            eigen_data = torch.load('results/raw/eigenvectors.pt')
        except Exception as e:
            print(f"Failed to load eigenvectors for {model_name}: {e}")
            continue
            
        layers_found = [int(k.split('_')[1]) for k in eigen_data.keys()]
        
        for layer in layers_found:
            l_key = f"layer_{layer}"
            snr = eigen_data[l_key]["sigma1"] / eigen_data[l_key]["sigma_mp"]
            
            print(f"\n🔬 Scaling Analysis: Layer {layer} (SNR: {snr:.2f})")
            
            for benchmark in benchmarks:
                for alpha in alphas:
                    # Check if already processed
                    already_done = any(r for r in results_log if r['model'] == model_name and r['layer'] == layer and r['alpha'] == alpha and r['benchmark'] == benchmark)
                    if already_done:
                        print(f"Skipping {model_name} L{layer} Alpha {alpha} - Already done.")
                        continue

                    print(f"   ➔ Testing Alpha {alpha} on {benchmark}...")
                    
                    if alpha == 0.0:
                        # Baseline evaluation
                        model_path = model_id
                    else:
                        # Deflate and evaluate
                        print(f"     [Deflating...]")
                        run_cmd([sys.executable, str(script_path), "deflate", "--layer", str(layer), "--alpha", str(alpha), "--model", model_id, "--load_bit4"])
                        model_path = "./deflated_model"
                    
                    # Evaluate
                    eval_res = run_cmd([sys.executable, str(script_path), "eval", "--benchmark", benchmark, "--model_path", model_path, "--load_bit4"])
                    
                    error_rate = 0.0
                    for line in eval_res.stdout.split('\n'):
                        if "Sycophancy Error Rate" in line:
                            try:
                                error_rate = float(line.split(":")[1].strip().replace('%',''))
                            except:
                                pass
                    
                    print(f"     [Result] {benchmark} Error: {error_rate}%")
                    
                    results_log.append({
                        "model": model_name,
                        "layer": layer,
                        "alpha": alpha,
                        "benchmark": benchmark,
                        "error_rate": error_rate,
                        "snr": snr,
                        "sigma1": float(eigen_data[l_key]["sigma1"]),
                        "sigma_mp": float(eigen_data[l_key]["sigma_mp"])
                    })
                    
                    # Periodic save
                    with open(output_file, "w") as f:
                        json.dump(results_log, f, indent=4)

    print("\n✅ Scaling Sweep Complete. Results saved to scaling_results.json")

if __name__ == "__main__":
    main()
