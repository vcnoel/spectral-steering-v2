import os
import sys
import subprocess
from pathlib import Path

models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct"
]

def run_command(cmd_list):
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    try:
        # Run and stream output normally, but stop on failure so we can debug
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Command failed with exit code: {e.returncode}")
        print("Stopping execution so the debug log is preserved.")
        sys.exit(1)

def main():
    print("[INFO] Starting Fast-Track Rebuttal Experiments...")
    script_path = Path(__file__).parent / "run_phase1.py"
    
    for model in models:
        print(f"\n" + "="*50)
        print(f"[PROCESS] Processing Model: {model}")
        print("="*50)
        
        # 1. Extract Eigenvector (Subset: 100 pairs)
        print("\n--- 1. Extracting Eigenvectors (100 pairs) ---")
        run_command([sys.executable, str(script_path), "extract", "--model", model, "--n_pairs", "100"])
        
        # 2. Evaluate Base Model
        print("\n--- 2. Evaluating Base Model (Sycophancy) ---")
        run_command([sys.executable, str(script_path), "eval", "--benchmark", "sycophancy", "--model_path", model])
        
        # 3. Deflate at Middle-Late Layer
        # Assuming the generated eigenvectors.pt will have layer roughly at 3/4 depth
        # Since we dynamicly compute layers: [num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 2]
        import torch
        from transformers import AutoConfig
        
        try:
            config = AutoConfig.from_pretrained(model)
            num_layers = config.num_hidden_layers
            target_layer = (3 * num_layers) // 4
            
            print(f"\n--- 3. Deflating Layer {target_layer} ---")
            run_command([sys.executable, str(script_path), "deflate", "--layer", str(target_layer), "--alpha", "0.5", "--model", model])
            
            # 4. Evaluate Deflated Model
            print("\n--- 4. Evaluating Deflated Model (Sycophancy) ---")
            run_command([sys.executable, str(script_path), "eval", "--benchmark", "sycophancy", "--model_path", "./deflated_model"])
        except Exception as e:
            print(f"Failed to process deflation dynamically: {e}")

if __name__ == "__main__":
    main()
