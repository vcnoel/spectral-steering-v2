import os
import sys
import subprocess
import json
from pathlib import Path
import torch

models = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "layers": [12, 18]},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "layers": [8, 12]}
]

alphas = [-0.5, 0.5, 1.0, -1.0]
results_log = []

def run_cmd(cmd_list):
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)

def main():
    script_path = Path(__file__).parent / "run_phase1.py"
    
    print("\n🚀 Starting Rapid Cross-Model Trajectory Sweep...")
    
    for m in models:
        model_name = m["id"]
        layers = m["layers"]
        print(f"\n======================================")
        print(f"🔥 Profiling {model_name}")
        print(f"======================================")

        run_cmd([sys.executable, str(script_path), "extract", "--model", model_name, "--n_pairs", "100"])
        
        run_cmd([sys.executable, str(script_path), "eval", "--benchmark", "sycophancy", "--model_path", model_name])
        
        eigen_data = torch.load('results/raw/eigenvectors.pt')
        
        for layer in layers:
            l_key = f"layer_{layer}"
            metrics = {
                "model": model_name,
                "layer": layer,
                "sigma_1": eigen_data[l_key]["sigma1"],
                "sigma_2": eigen_data[l_key]["sigma2"],
                "sigma_mp": eigen_data[l_key]["sigma_mp"],
                "snr": eigen_data[l_key]["sigma1"] / eigen_data[l_key]["sigma_mp"],
                "alpha_results": {}
            }
            print(f"\n🔬 Profiling Layer {layer} (SNR: {metrics['snr']:.2f})")
            
            for alpha in alphas:
                try:
                    run_cmd([sys.executable, str(script_path), "deflate", "--layer", str(layer), "--alpha", str(alpha), "--model", model_name])
                    
                    # Instead of running full eval which prints to stdout, we just pipe to capturing directly here
                    eval_cmd = [sys.executable, str(script_path), "eval", "--benchmark", "sycophancy", "--model_path", "./deflated_model"]
                    res = subprocess.run(eval_cmd, capture_output=True, text=True)
                    
                    error_rate = None
                    for line in res.stdout.split('\n'):
                        if "Sycophancy Error Rate" in line:
                            val = line.split(":")[1].strip().replace('%','')
                            error_rate = float(val)
                            
                    print(f"➔ Alpha {alpha} | Sycophancy Error: {error_rate}%")
                    metrics["alpha_results"][str(alpha)] = error_rate
                except Exception as e:
                    print(f"Error evaluating alpha {alpha}: {e}")
                    
            results_log.append(metrics)

    with open("rebuttal_sweep_results.json", "w") as f:
        json.dump(results_log, f, indent=4)
        
    print("\n✅ Sweep Complete. Results saved to rebuttal_sweep_results.json")

if __name__ == "__main__":
    main()
