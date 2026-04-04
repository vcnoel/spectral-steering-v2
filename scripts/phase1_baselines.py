import os
import torch
import json
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from baselines.repe import RepEHook
from baselines.actadd import ActAddHook
from spectral.deflation import deflate_module_weight
# from abliterator import Abliterator # (Assuming FailSpy/abliterator is cloned here or installed)

def run_baselines(model, tokenizer, vector_dict):
    """
    Runs the exact baseline matrix: base, repe, actadd, abliteration, random_deflation, spectral_deflation.
    """
    # 1. Base
    # ... evaluate ...
    
    # 2. RepE (All layers, alpha=1.5 default)
    repe_hooks = RepEHook(vector_dict, alpha=1.5)
    repe_hooks.register(model, [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)])
    # ... evaluate ...
    repe_hooks.remove()
    
    # 3. ActAdd (Layer 16)
    act_hook = ActAddHook(vector_dict[16], target_layer_idx=16, alpha=1.0)
    act_hook.register(model, "model.layers.16")
    # ... evaluate ...
    act_hook.remove()
    
    # 4. Abliteration
    # Requires orthogonal projection of refusal direction
    # W_new = W_old - (v v^T) W_old
    
    # 5. Random Deflation (Control)
    # v_random = torch.nn.functional.normalize(torch.randn(4096), dim=0)
    
    # 6. Spectral Deflation
    # W_new = W_old - alpha * sigma1 * u1 * v1T
    pass

if __name__ == '__main__':
    print("Phase 1 Baselines Orchestrator Ready.")
