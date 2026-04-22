#!/usr/bin/env python3
"""
Three-Experiment Grid Search on Gemma-4-E2B-it
===============================================
Experiment 1: Find minimum sycophancy (Safety Max)
Experiment 2: Find maximum reasoning (Reasoning Max)
Experiment 3: Break alignment tax (preserved GSM8K)
+ Multi-seed confirmation on top 3 per experiment
+ Pareto frontier plot

Uses steer.py infrastructure directly.
"""

import json, os, random, sys, time, copy
import torch
import gc

# Add parent dir to path so we can import from steer.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer import (
    load_model, load_sycophancy_data, load_gsm8k_data,
    get_svd, apply_multi_steering, restore_multi_steering,
    eval_sycophancy, eval_gsm8k, get_layers
)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-E2B-it"
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def eval_config(model, tokenizer, svd_cache, layer_alphas, n_syco, n_gsm, syco_data, gsm_ds, gsm_idx):
    """
    Apply steering, evaluate, restore. Returns dict with results.
    layer_alphas: dict {layer_idx: alpha}
    """
    t0 = time.time()
    
    configs = [(int(l), float(a)) for l, a in layer_alphas.items()]
    
    # Ensure SVD is cached for all layers
    for l_idx, _ in configs:
        if l_idx not in svd_cache:
            dp, U, S, Vh, _ = get_svd(model, l_idx)
            svd_cache[l_idx] = (dp, U, S, Vh)
    
    # Apply steering
    originals = apply_multi_steering(model, configs, svd_cache)
    
    # Evaluate sycophancy
    syco_hits, syco_total = eval_sycophancy(model, tokenizer, syco_data, n=n_syco, batch_size=1)
    syco_pct = round(syco_hits / syco_total * 100, 1)
    
    # Evaluate GSM8K
    gsm_correct, gsm_total = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=n_gsm, batch_size=1)
    gsm_pct = round(gsm_correct / gsm_total * 100, 1)
    
    # Restore
    restore_multi_steering(model, originals)
    
    elapsed = round(time.time() - t0, 1)
    
    return {
        "layers": {str(k): v for k, v in layer_alphas.items()},
        "n_syco": syco_total,
        "n_gsm": gsm_total,
        "sycophancy_pct": syco_pct,
        "gsm8k_pct": gsm_pct,
        "wall_clock_sec": elapsed,
        "syco_raw": f"{syco_hits}/{syco_total}",
        "gsm_raw": f"{gsm_correct}/{gsm_total}"
    }

def print_config_result(label, result):
    layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in result["layers"].items())
    print(f"  {label:30s} | {layers_str:40s} | Syco={result['sycophancy_pct']:5.1f}% | GSM={result['gsm8k_pct']:5.1f}% | {result['wall_clock_sec']:.0f}s")

def save_checkpoint(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [Checkpoint saved: {filepath}]")


# ===========================================================================
# EXPERIMENT 1: Find minimum sycophancy (Safety Max)
# ===========================================================================

def run_experiment_1(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: MINIMUM SYCOPHANCY (Safety Max)")
    print("=" * 70)
    
    results = []
    
    # Phase A: L18 only, stronger negative alphas
    print("\n--- Phase A: L18-only grid ---")
    for alpha in [-0.10, -0.12, -0.15, -0.20]:
        label = f"L18:{alpha:+.2f}"
        layers = {18: alpha}
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=200, n_gsm=50, 
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp1"
        r["phase"] = "A_L18_only"
        results.append(r)
        print_config_result(label, r)
        save_checkpoint({"experiment": "exp1", "results": results}, "exp1_min_sycophancy.json")
    
    # Phase B: L18 + L19 combos
    print("\n--- Phase B: L18+L19 combos ---")
    combos = [
        {18: -0.07, 19: -0.03},
        {18: -0.07, 19: -0.05},
        {18: -0.10, 19: -0.03},
    ]
    for layers in combos:
        label = "+".join(f"L{k}:{v:+.2f}" for k, v in layers.items())
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=200, n_gsm=50,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp1"
        r["phase"] = "B_L18_L19"
        results.append(r)
        print_config_result(label, r)
        save_checkpoint({"experiment": "exp1", "results": results}, "exp1_min_sycophancy.json")
    
    print(f"\n  Experiment 1 complete: {len(results)} configs evaluated")
    
    # Print summary table
    print(f"\n  {'Config':40s} | {'Syco':>6s} | {'GSM8K':>6s}")
    print(f"  {'-'*60}")
    for r in sorted(results, key=lambda x: x["sycophancy_pct"]):
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in r["layers"].items())
        print(f"  {layers_str:40s} | {r['sycophancy_pct']:5.1f}% | {r['gsm8k_pct']:5.1f}%")
    
    return results


# ===========================================================================
# EXPERIMENT 2: Find maximum reasoning (Reasoning Max)
# ===========================================================================

def run_experiment_2(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: MAXIMUM REASONING (Reasoning Max)")
    print("=" * 70)
    
    results = []
    
    # Phase A: L24 only
    print("\n--- Phase A: L24-only grid ---")
    for alpha in [+0.15, +0.20, +0.25, +0.30]:
        label = f"L24:{alpha:+.2f}"
        layers = {24: alpha}
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=50, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp2"
        r["phase"] = "A_L24_only"
        results.append(r)
        print_config_result(label, r)
        
        # Early stop if GSM collapses
        if r["gsm8k_pct"] < 30.0:
            print(f"  *** GSM8K COLLAPSED below 30% at alpha={alpha}. Stopping L24 branch. ***")
            break
        save_checkpoint({"experiment": "exp2", "results": results}, "exp2_max_reasoning.json")
    
    # Phase B: L28 only
    print("\n--- Phase B: L28-only grid ---")
    for alpha in [+0.10, +0.15, +0.20]:
        label = f"L28:{alpha:+.2f}"
        layers = {28: alpha}
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=50, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp2"
        r["phase"] = "B_L28_only"
        results.append(r)
        print_config_result(label, r)
        
        if r["gsm8k_pct"] < 30.0:
            print(f"  *** GSM8K COLLAPSED below 30% at alpha={alpha}. Stopping L28 branch. ***")
            break
        save_checkpoint({"experiment": "exp2", "results": results}, "exp2_max_reasoning.json")
    
    # Phase C: L24 + L28 combos
    print("\n--- Phase C: L24+L28 combos ---")
    combos = [
        {24: +0.20, 28: +0.10},
        {24: +0.25, 28: +0.05},
        {24: +0.15, 28: +0.15},
    ]
    for layers in combos:
        label = "+".join(f"L{k}:{v:+.2f}" for k, v in layers.items())
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=50, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp2"
        r["phase"] = "C_L24_L28"
        results.append(r)
        print_config_result(label, r)
        
        if r["gsm8k_pct"] < 30.0:
            print(f"  *** GSM8K COLLAPSED. ***")
        save_checkpoint({"experiment": "exp2", "results": results}, "exp2_max_reasoning.json")
    
    # Phase D: L24 + L33 combos
    print("\n--- Phase D: L24+L33 (terminal layer) combos ---")
    combos = [
        {24: +0.15, 33: +0.05},
        {24: +0.15, 33: +0.10},
    ]
    for layers in combos:
        label = "+".join(f"L{k}:{v:+.2f}" for k, v in layers.items())
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=50, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp2"
        r["phase"] = "D_L24_L33"
        results.append(r)
        print_config_result(label, r)
        
        if r["gsm8k_pct"] < 30.0:
            print(f"  *** GSM8K COLLAPSED. ***")
        save_checkpoint({"experiment": "exp2", "results": results}, "exp2_max_reasoning.json")
    
    print(f"\n  Experiment 2 complete: {len(results)} configs evaluated")
    
    # Print summary table
    print(f"\n  {'Config':40s} | {'Syco':>6s} | {'GSM8K':>6s}")
    print(f"  {'-'*60}")
    for r in sorted(results, key=lambda x: -x["gsm8k_pct"]):
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in r["layers"].items())
        print(f"  {layers_str:40s} | {r['sycophancy_pct']:5.1f}% | {r['gsm8k_pct']:5.1f}%")
    
    return results


# ===========================================================================
# EXPERIMENT 3: Break alignment tax with preserved GSM8K
# ===========================================================================

def run_experiment_3(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: ALIGNMENT TAX BREAKER")
    print("  Goal: Syco as low as possible with GSM8K >= 35.4%")
    print("=" * 70)
    
    all_results = []
    
    # ---- Phase A: Increase L18 penalty + L24 compensation ----
    print("\n--- Phase A: L18 penalty + L24/L28 compensation ---")
    phase_a_configs = [
        {18: -0.05, 24: +0.12, 28: +0.05},
        {18: -0.06, 24: +0.14, 28: +0.05},
        {18: -0.07, 24: +0.16, 28: +0.05},
        {18: -0.07, 24: +0.16, 28: +0.08},
        {18: -0.08, 24: +0.18, 28: +0.08},
    ]
    
    phase_a_results = []
    for layers in phase_a_configs:
        label = "+".join(f"L{k}:{v:+.3f}" for k, v in layers.items())
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=200, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp3"
        r["phase"] = "A"
        phase_a_results.append(r)
        all_results.append(r)
        print_config_result(label, r)
        save_checkpoint({"experiment": "exp3", "results": all_results}, "exp3_alignment_tax.json")
    
    # Find best Phase A: GSM >= 35.4%, lowest syco
    eligible_a = [r for r in phase_a_results if r["gsm8k_pct"] >= 35.4]
    if eligible_a:
        best_a = min(eligible_a, key=lambda x: x["sycophancy_pct"])
    else:
        # If none meets threshold, pick best GSM with reasonable syco
        best_a = max(phase_a_results, key=lambda x: x["gsm8k_pct"])
    
    best_a_layers = {int(k): float(v) for k, v in best_a["layers"].items()}
    print(f"\n  BEST Phase A: {best_a['layers']} => Syco={best_a['sycophancy_pct']}%, GSM={best_a['gsm8k_pct']}%")
    
    # ---- Phase B: Add L33 as stabilizer ----
    print("\n--- Phase B: Add L33 stabilizer to best Phase A ---")
    phase_b_results = []
    for l33_alpha in [+0.03, +0.05, +0.08]:
        layers = copy.deepcopy(best_a_layers)
        layers[33] = l33_alpha
        label = "+".join(f"L{k}:{v:+.3f}" for k, v in layers.items())
        r = eval_config(model, tokenizer, svd_cache, layers, n_syco=200, n_gsm=200,
                        syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
        r["experiment"] = "exp3"
        r["phase"] = "B"
        phase_b_results.append(r)
        all_results.append(r)
        print_config_result(label, r)
        save_checkpoint({"experiment": "exp3", "results": all_results}, "exp3_alignment_tax.json")
    
    # Find best of Phase A + B combined
    all_ab = phase_a_results + phase_b_results
    eligible_ab = [r for r in all_ab if r["gsm8k_pct"] >= 35.4]
    if eligible_ab:
        best_ab = min(eligible_ab, key=lambda x: x["sycophancy_pct"])
    else:
        best_ab = max(all_ab, key=lambda x: x["gsm8k_pct"])
    
    best_ab_layers = {int(k): float(v) for k, v in best_ab["layers"].items()}
    print(f"\n  BEST Phase A+B: {best_ab['layers']} => Syco={best_ab['sycophancy_pct']}%, GSM={best_ab['gsm8k_pct']}%")
    
    # ---- Phase C: Fine grid around best config (+/- 0.01 per param) ----
    print("\n--- Phase C: Fine perturbation sweep around best config ---")
    phase_c_results = []
    param_keys = sorted(best_ab_layers.keys())
    
    for key in param_keys:
        for delta in [-0.01, +0.01]:
            layers = copy.deepcopy(best_ab_layers)
            layers[key] = round(layers[key] + delta, 4)
            label = f"Perturb L{key} {delta:+.2f}"
            r = eval_config(model, tokenizer, svd_cache, layers, n_syco=200, n_gsm=200,
                            syco_data=syco_data, gsm_ds=gsm_ds, gsm_idx=gsm_idx)
            r["experiment"] = "exp3"
            r["phase"] = "C"
            r["perturbation"] = f"L{key}{delta:+.02f}"
            phase_c_results.append(r)
            all_results.append(r)
            print_config_result(label, r)
            save_checkpoint({"experiment": "exp3", "results": all_results}, "exp3_alignment_tax.json")
    
    print(f"\n  Experiment 3 complete: {len(all_results)} configs evaluated")
    
    # Final best: among all configs with GSM >= 35.4%, lowest syco
    eligible_all = [r for r in all_results if r["gsm8k_pct"] >= 35.4]
    if eligible_all:
        champion = min(eligible_all, key=lambda x: x["sycophancy_pct"])
        print(f"\n  *** CHAMPION (GSM >= 35.4%): {champion['layers']} => Syco={champion['sycophancy_pct']}%, GSM={champion['gsm8k_pct']}% ***")
    else:
        champion = max(all_results, key=lambda x: x["gsm8k_pct"])
        print(f"\n  *** BEST EFFORT (no config met GSM threshold): {champion['layers']} => Syco={champion['sycophancy_pct']}%, GSM={champion['gsm8k_pct']}% ***")
    
    # Print summary table
    print(f"\n  {'Config':50s} | {'Syco':>6s} | {'GSM8K':>6s} | {'Phase'}")
    print(f"  {'-'*80}")
    for r in sorted(all_results, key=lambda x: x["sycophancy_pct"]):
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in r["layers"].items())
        marker = " ***" if r["gsm8k_pct"] >= 35.4 else ""
        print(f"  {layers_str:50s} | {r['sycophancy_pct']:5.1f}% | {r['gsm8k_pct']:5.1f}% | {r['phase']}{marker}")
    
    return all_results


# ===========================================================================
# MULTI-SEED CONFIRMATION
# ===========================================================================

def run_confirmation(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx,
                     exp1_results, exp2_results, exp3_results):
    """
    Run top 3 configs per experiment with 5 seeds.
    Report mean +/- std for each.
    """
    print("\n" + "=" * 70)
    print("  MULTI-SEED CONFIRMATION (5 seeds x 9 configs)")
    print("=" * 70)
    
    seeds = [42, 123, 456, 789, 1000]
    
    # Pick top 3 per experiment
    # Exp 1: lowest sycophancy
    top1 = sorted(exp1_results, key=lambda x: x["sycophancy_pct"])[:3]
    # Exp 2: highest GSM8K (excluding collapsed)
    valid_exp2 = [r for r in exp2_results if r["gsm8k_pct"] >= 30.0]
    top2 = sorted(valid_exp2, key=lambda x: -x["gsm8k_pct"])[:3]
    # Exp 3: lowest syco with GSM >= 35.4%
    eligible3 = [r for r in exp3_results if r["gsm8k_pct"] >= 35.4]
    if not eligible3:
        eligible3 = sorted(exp3_results, key=lambda x: -x["gsm8k_pct"])
    top3 = sorted(eligible3, key=lambda x: x["sycophancy_pct"])[:3]
    
    all_confirmations = []
    
    for exp_name, top_configs in [("exp1", top1), ("exp2", top2), ("exp3", top3)]:
        print(f"\n--- Confirming {exp_name} top {len(top_configs)} ---")
        for cfg in top_configs:
            layers = {int(k): float(v) for k, v in cfg["layers"].items()}
            layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in layers.items())
            print(f"\n  Config: {layers_str}")
            
            seed_results = {"syco": [], "gsm": []}
            for seed in seeds:
                set_seed(seed)
                
                # Re-shuffle data with this seed
                syco_data_copy = list(syco_data)
                random.seed(seed)
                random.shuffle(syco_data_copy)
                
                gsm_idx_copy = list(range(len(gsm_ds)))
                random.seed(seed)
                random.shuffle(gsm_idx_copy)
                
                r = eval_config(model, tokenizer, svd_cache, layers,
                                n_syco=200, n_gsm=100,
                                syco_data=syco_data_copy, gsm_ds=gsm_ds, gsm_idx=gsm_idx_copy)
                seed_results["syco"].append(r["sycophancy_pct"])
                seed_results["gsm"].append(r["gsm8k_pct"])
                print(f"    seed={seed}: Syco={r['sycophancy_pct']:.1f}%, GSM={r['gsm8k_pct']:.1f}%")
            
            import numpy as np
            syco_mean = np.mean(seed_results["syco"])
            syco_std = np.std(seed_results["syco"])
            gsm_mean = np.mean(seed_results["gsm"])
            gsm_std = np.std(seed_results["gsm"])
            
            confirmation = {
                "experiment": exp_name,
                "layers": cfg["layers"],
                "seeds": seeds,
                "syco_per_seed": seed_results["syco"],
                "gsm_per_seed": seed_results["gsm"],
                "syco_mean": round(syco_mean, 2),
                "syco_std": round(syco_std, 2),
                "gsm_mean": round(gsm_mean, 2),
                "gsm_std": round(gsm_std, 2),
            }
            all_confirmations.append(confirmation)
            print(f"    => Syco={syco_mean:.1f}% ± {syco_std:.1f}% | GSM={gsm_mean:.1f}% ± {gsm_std:.1f}%")
    
    save_checkpoint({"confirmations": all_confirmations}, "confirmation_results.json")
    return all_confirmations


# ===========================================================================
# PARETO FRONTIER PLOT
# ===========================================================================

def plot_pareto(exp1_results, exp2_results, exp3_results):
    print("\n  Generating Pareto frontier plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available, skipping plot.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Baseline point
    ax.scatter([39.5], [36.4], c='black', s=200, marker='*', zorder=10, label='Baseline (39.5%, 36.4%)')
    ax.annotate('Baseline', (39.5, 36.4), textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight='bold')
    
    # All configs from each experiment
    colors = {'exp1': '#e74c3c', 'exp2': '#3498db', 'exp3': '#2ecc71'}
    labels = {'exp1': 'Exp 1: Safety Max', 'exp2': 'Exp 2: Reasoning Max', 'exp3': 'Exp 3: Tax Breaker'}
    
    for exp_name, results in [('exp1', exp1_results), ('exp2', exp2_results), ('exp3', exp3_results)]:
        xs = [r['sycophancy_pct'] for r in results]
        ys = [r['gsm8k_pct'] for r in results]
        ax.scatter(xs, ys, c=colors[exp_name], s=80, alpha=0.7, label=labels[exp_name], edgecolors='white', linewidth=0.5)
    
    # Highlight champions
    # Best Safety: lowest syco in exp1
    best_safety = min(exp1_results, key=lambda x: x['sycophancy_pct'])
    ax.scatter([best_safety['sycophancy_pct']], [best_safety['gsm8k_pct']], 
               c='#e74c3c', s=300, marker='D', edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'Safety Max\n({best_safety["sycophancy_pct"]:.1f}%, {best_safety["gsm8k_pct"]:.1f}%)',
                (best_safety['sycophancy_pct'], best_safety['gsm8k_pct']),
                textcoords="offset points", xytext=(-15, -20), fontsize=8, color='#e74c3c', fontweight='bold')
    
    # Best Reasoning: highest GSM in exp2
    valid_exp2 = [r for r in exp2_results if r['gsm8k_pct'] >= 30.0]
    if valid_exp2:
        best_reasoning = max(valid_exp2, key=lambda x: x['gsm8k_pct'])
    else:
        best_reasoning = max(exp2_results, key=lambda x: x['gsm8k_pct'])
    ax.scatter([best_reasoning['sycophancy_pct']], [best_reasoning['gsm8k_pct']], 
               c='#3498db', s=300, marker='D', edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'Reasoning Max\n({best_reasoning["sycophancy_pct"]:.1f}%, {best_reasoning["gsm8k_pct"]:.1f}%)',
                (best_reasoning['sycophancy_pct'], best_reasoning['gsm8k_pct']),
                textcoords="offset points", xytext=(10, 10), fontsize=8, color='#3498db', fontweight='bold')
    
    # Best Tax Breaker: lowest syco with GSM >= 35.4% in exp3
    eligible3 = [r for r in exp3_results if r['gsm8k_pct'] >= 35.4]
    if eligible3:
        best_tax = min(eligible3, key=lambda x: x['sycophancy_pct'])
    else:
        best_tax = max(exp3_results, key=lambda x: x['gsm8k_pct'])
    ax.scatter([best_tax['sycophancy_pct']], [best_tax['gsm8k_pct']], 
               c='#2ecc71', s=300, marker='D', edgecolors='black', linewidth=2, zorder=10)
    ax.annotate(f'Tax Breaker\n({best_tax["sycophancy_pct"]:.1f}%, {best_tax["gsm8k_pct"]:.1f}%)',
                (best_tax['sycophancy_pct'], best_tax['gsm8k_pct']),
                textcoords="offset points", xytext=(10, -20), fontsize=8, color='#2ecc71', fontweight='bold')
    
    # Reference lines
    ax.axhline(y=35.4, color='gray', linestyle='--', alpha=0.5, label='GSM8K threshold (35.4%)')
    ax.axhline(y=36.4, color='gray', linestyle=':', alpha=0.3, label='Baseline GSM8K (36.4%)')
    
    ax.set_xlabel('Sycophancy %', fontsize=13)
    ax.set_ylabel('GSM8K Accuracy %', fontsize=13)
    ax.set_title('Spectral Steering: Pareto Frontier\nGemma-4-E2B-it — All Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "pareto_frontier.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Pareto plot saved to {out_path}")
    plt.close()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_global = time.time()
    ensure_dirs()
    set_seed(SEED)
    
    print("=" * 70)
    print("  SPECTRAL STEERING v2: THREE-EXPERIMENT CAMPAIGN")
    print(f"  Model: {MODEL_ID}")
    print(f"  Seed: {SEED}")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load model (bfloat16 for precision, consistent with previous validation runs)
    print("\nLoading model (bfloat16)...")
    model, tokenizer, meta = load_model(MODEL_ID, load_4bit=False)
    print(f"  {meta['model_id']} | {meta['n_layers']}L | {meta['arch']}")
    
    # Load data
    print("Loading datasets...")
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()
    
    # Pre-cache SVDs for all layers we'll need 
    # L18, L19, L24, L28, L33 are the primary targets
    print("Pre-caching SVD for layers 18, 19, 24, 28, 33...")
    svd_cache = {}
    for l_idx in [18, 19, 24, 28, 33]:
        dp, U, S, Vh, snr = get_svd(model, l_idx)
        svd_cache[l_idx] = (dp, U, S, Vh)
        print(f"  L{l_idx}: SNR={snr:.3f}")
    
    # ========================
    # RUN EXPERIMENT 1
    # ========================
    exp1_results = run_experiment_1(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx)
    
    # ========================
    # RUN EXPERIMENT 2  
    # ========================
    exp2_results = run_experiment_2(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx)
    
    # ========================
    # RUN EXPERIMENT 3
    # ========================
    exp3_results = run_experiment_3(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx)
    
    # ========================
    # SUMMARY
    # ========================
    print("\n" + "=" * 70)
    print("  CAMPAIGN SUMMARY")
    print("=" * 70)
    
    # Best Safety Max
    best_safety = min(exp1_results, key=lambda x: x['sycophancy_pct'])
    layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in best_safety["layers"].items())
    print(f"\n  Best Safety Max:    [{layers_str}] => Syco={best_safety['sycophancy_pct']:.1f}%, GSM={best_safety['gsm8k_pct']:.1f}%")
    
    # Best Reasoning Max  
    valid_exp2 = [r for r in exp2_results if r['gsm8k_pct'] >= 30.0]
    if valid_exp2:
        best_reasoning = max(valid_exp2, key=lambda x: x['gsm8k_pct'])
    else:
        best_reasoning = max(exp2_results, key=lambda x: x['gsm8k_pct'])
    layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in best_reasoning["layers"].items())
    print(f"  Best Reasoning Max: [{layers_str}] => Syco={best_reasoning['sycophancy_pct']:.1f}%, GSM={best_reasoning['gsm8k_pct']:.1f}%")
    
    # Best Tax Breaker
    eligible3 = [r for r in exp3_results if r['gsm8k_pct'] >= 35.4]
    if eligible3:
        best_tax = min(eligible3, key=lambda x: x['sycophancy_pct'])
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in best_tax["layers"].items())
        print(f"  Best Tax Breaker:   [{layers_str}] => Syco={best_tax['sycophancy_pct']:.1f}%, GSM={best_tax['gsm8k_pct']:.1f}% (GSM >= 35.4%)")
    else:
        best_tax = max(exp3_results, key=lambda x: x['gsm8k_pct'])
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in best_tax["layers"].items())
        print(f"  Best Tax Breaker:   [{layers_str}] => Syco={best_tax['sycophancy_pct']:.1f}%, GSM={best_tax['gsm8k_pct']:.1f}% (BEST EFFORT - no config met threshold)")
    
    # ========================
    # PARETO PLOT
    # ========================
    plot_pareto(exp1_results, exp2_results, exp3_results)
    
    # ========================
    # MULTI-SEED CONFIRMATION
    # ========================
    confirmations = run_confirmation(model, tokenizer, svd_cache, syco_data, gsm_ds, gsm_idx,
                                    exp1_results, exp2_results, exp3_results)
    
    # Print confirmation summary
    print("\n" + "=" * 70)
    print("  CONFIRMATION RESULTS (mean ± std over 5 seeds)")
    print("=" * 70)
    for c in confirmations:
        layers_str = ", ".join(f"L{k}:{v:+.3f}" for k, v in c["layers"].items())
        print(f"  [{c['experiment']}] {layers_str:45s} | Syco={c['syco_mean']:.1f}%±{c['syco_std']:.1f}% | GSM={c['gsm_mean']:.1f}%±{c['gsm_std']:.1f}%")
    
    total_time = round((time.time() - t_global) / 60, 1)
    print(f"\n  TOTAL WALL CLOCK: {total_time} minutes ({total_time/60:.1f} hours)")
    print(f"  Campaign finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
