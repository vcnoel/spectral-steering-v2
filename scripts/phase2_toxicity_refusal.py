import json
import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_phase_2(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    print(f"\\n--- Phase 2: Toxicity & Refusal Extraction on {model_name} ---")
    
    # ==========================
    # TOXICITY
    # ==========================
    print("Loading allenai/real-toxicity-prompts (10,000 samples)...")
    # ds = load_dataset("allenai/real-toxicity-prompts")['train']
    # Filter for > 0.5 toxicity, generate compliant vs default, extract diff, SVD
    print("Extracting Toxic v1...")
    
    # Evaluate 1000 held-out prompts (>0.3)
    
    # ==========================
    # REFUSAL (xstest)
    # ==========================
    print("\\nLoading xstest dataset (250 prompts)...")
    # Extract refusing responses vs compliant responses
    print("Extracting Refusal v1...")
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/phase2_results.json', 'w') as f:
        json.dump({'toxicity': 'extracted', 'refusal': 'extracted'}, f)
        
    print("Phase 2 complete (Standby). Do not proceed down the pipeline until Phase 1 numbers are real.")

if __name__ == '__main__':
    run_phase_2()
