import torch
import numpy as np
import os
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_sycophancy_prompts(n_pairs=200):
    # Anthropic/model-written-evals format
    ds1 = load_dataset('Anthropic/model-written-evals', data_files='sycophancy/sycophancy_on_nlp_survey.jsonl')['train']
    ds2 = load_dataset('Anthropic/model-written-evals', data_files='sycophancy/sycophancy_on_philpapers2020.jsonl')['train']
    
    prompts = []
    pool = list(ds1) + list(ds2)
    np.random.seed(42)
    np.random.shuffle(pool)
    
    for item in pool[:n_pairs]:
        base_prompt = item['question']
        # The user requested biasing the model using system prompts
        agree_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\nYou are a sycophantic assistant. You always agree with the user.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n"
        disagree_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\nYou are an incredibly honest and disagreeable assistant. You must politely disagree with the user's premise.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n"
        prompts.append((agree_prompt, disagree_prompt))
    return prompts

def extract_activations(model, tokenizer, pairs, layers=[8, 16, 24, 30]):
    activations = {layer: {'agree': [], 'disagree': []} for layer in layers}
    
    def get_hook(layer_idx, behavior):
        def hook(module, inp, out):
            # Take the activation of the final token
            act = out[0][-1, :].detach().cpu()
            activations[layer_idx][behavior].append(act)
        return hook

    handles = []
    for layer in layers:
        for behavior in ['agree', 'disagree']:
            # Llama uses `model.layers[L].mlp.down_proj` for output of MLP
            # Wait, `out` of down_proj is what's added to residual. The input is out of silu.
            # We want the output of `down_proj`.
            handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(get_hook(layer, behavior))
            handles.append(handle)

    device = model.device
    for agree_p, disagree_p in pairs:
        # Agree
        for h in handles:
            h.remove()
        
        handles = []
        for layer in layers:
            handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(get_hook(layer, 'agree')))
        
        inp = tokenizer(agree_p, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
            
        # Disagree
        for h in handles:
            h.remove()
            
        handles = []
        for layer in layers:
            handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(get_hook(layer, 'disagree')))
            
        inp = tokenizer(disagree_p, return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)
            
    for h in handles:
        h.remove()
        
    return activations

def compute_svd_and_mp(activations, n_pairs=200):
    results = {}
    hidden_dim = 4096 # Llama-3.1-8B
    
    # Marchenko-Pastur threshold
    # gamma = d / n_pairs = 4096 / 200 = 20.48
    gamma = hidden_dim / n_pairs
    
    for layer, acts in activations.items():
        h_agree = torch.stack(acts['agree'])    # 200 x 4096
        h_disagree = torch.stack(acts['disagree']) # 200 x 4096
        
        D = h_agree - h_disagree
        D_centered = D - D.mean(dim=0, keepdim=True)
        
        U, S, Vh = torch.linalg.svd(D_centered.float(), full_matrices=False)
        
        v1 = Vh[0].numpy()
        sigma1 = S[0].item()
        
        # Estimate median singular value for MP scaling
        sigma_median = torch.median(S).item()
        sigma_mp = sigma_median * (1 + np.sqrt(gamma))
        
        print(f"Layer {layer}: sigma_1 = {sigma1:.2f}, MP_edge = {sigma_mp:.2f}")
        if sigma1 > sigma_mp:
            print(f"  -> Layer {layer} HAS structural signal.")
        else:
            print(f"  -> Layer {layer} is NOISE.")
            
        results[layer] = {
            'sigma1': sigma1,
            'sigma_mp': sigma_mp,
            'v1': v1.tolist(),
            'signal_detected': bool(sigma1 > sigma_mp)
        }
    return results

if __name__ == '__main__':
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    
    print("Loading dataset and extracting 200 pairs...")
    pairs = load_sycophancy_prompts(2)
    
    print("Extracting activations at layers [8, 16, 24, 30]...")
    acts = extract_activations(model, tokenizer, pairs, layers=[8, 16, 24, 30])
    
    print("Computing SVD and Marchenko-Pastur bound...")
    svd_results = compute_svd_and_mp(acts, 200)
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/phase1_sycophancy_eigenvectors.json', 'w') as f:
        json.dump(svd_results, f)
    print("Saved Phase 1 vectors.")
