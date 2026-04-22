#!/usr/bin/env python3
"""
Legacy one-off sweep commands — research phase artifacts.

These commands were used during the iterative exploration phase and produced
the results in scaling_results.json / data/results/.  They are preserved here
for reproducibility but are NOT part of the main CLI release.

Usage (still runnable):
  python scripts/legacy_sweeps/steer_legacy.py grid-hunt --model google/gemma-4-E2B-it
  python scripts/legacy_sweeps/steer_legacy.py mistral-hyper-sweep
  ...
"""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.steer import (
    load_model,
    eval_sycophancy,
    eval_gsm8k,
    get_svd,
    apply_multi_steering,
    restore_multi_steering,
    load_sycophancy_data,
    load_gsm8k_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum(scores):
    return sum(scores)

def _pct(scores):
    return sum(scores) / len(scores) * 100 if scores else 0.0

# ---------------------------------------------------------------------------
# Legacy commands
# ---------------------------------------------------------------------------

def cmd_grid_hunt(args):
    """
    High-resolution 81-config grid search for Gemma-4.
      Anchor: L33:0.05 | Alignment Zone: L17-L19 | Reasoning Zone: L23-L25
    """
    t0 = time.time()
    model, tokenizer, meta = load_model(args.model, load_4bit=False)
    tag = meta["model_id"].split("/")[-1]
    print(f"  {tag} | 81-Config Grid Hunt (bfloat16)")

    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()
    n_syco, n_gsm = 50, 30

    base_syco = 34.0
    base_gsm  = 36.0
    print(f"  Baseline (locked) => syco={base_syco:.1f}% gsm={base_gsm:.1f}%")

    anchor_layer, anchor_alpha = 33, 0.05
    align_layers  = [17, 18, 19]
    align_alphas  = [-0.04, -0.07, -0.10]
    reason_layers = [23, 24, 25]
    reason_alphas = [0.04, 0.07, 0.10]

    all_layers = sorted(set(align_layers + reason_layers + [anchor_layer]))
    print(f"  Pre-caching SVD for {all_layers}...")
    svd_cache = {}
    for l in all_layers:
        dp, U, S, Vh, _ = get_svd(model, l)
        svd_cache[l] = (dp, U, S, Vh)

    results = []
    checkpoint = f"data/results/grid_hunt_{tag}_checkpoint.json"
    os.makedirs("data/results", exist_ok=True)

    count = 0
    total = len(align_layers) * len(reason_layers) * len(align_alphas) * len(reason_alphas)

    for l_align in align_layers:
        for l_reason in reason_layers:
            for a_align in align_alphas:
                for a_reason in reason_alphas:
                    count += 1
                    configs = [(anchor_layer, anchor_alpha), (l_align, a_align), (l_reason, a_reason)]
                    print(f"\n  [{count}/{total}] Anchor L33:+0.05 | Align L{l_align}:{a_align:+.2f} | Reason L{l_reason}:{a_reason:+.2f}")

                    originals = apply_multi_steering(model, configs, svd_cache)
                    ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=n_syco)
                    sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=n_gsm)
                    restore_multi_steering(model, originals)

                    s_pct = _pct(ss)
                    g_pct = _pct(sg)
                    is_pareto = (g_pct >= base_gsm) and (s_pct < base_syco)
                    marker = " ***" if is_pareto else ""
                    print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%{marker}")

                    results.append({
                        "config": f"L33:{anchor_alpha},L{l_align}:{a_align},L{l_reason}:{a_reason}",
                        "l_align": l_align, "a_align": a_align,
                        "l_reason": l_reason, "a_reason": a_reason,
                        "syco_pct": s_pct, "gsm_pct": g_pct,
                        "is_pareto": is_pareto,
                    })
                    with open(checkpoint, "w") as f:
                        json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm},
                                   "grid": results,
                                   "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    elapsed = round((time.time() - t0) / 60, 1)
    out = f"data/results/grid_hunt_{tag}_final.json"
    with open(out, "w") as f:
        json.dump({"model": meta["model_id"], "protocol": "v2_grid_hunt",
                   "baseline": {"syco_pct": base_syco, "gsm_pct": base_gsm},
                   "grid": results, "wall_clock_min": elapsed}, f, indent=2)
    print(f"\n  Grid Hunt complete in {elapsed} min. Saved: {out}")


def cmd_validate_top(args):
    """High-precision N=200 validation of the top Gemma-4 Pareto candidates."""
    tt0 = time.time()
    model, tokenizer, meta = load_model("google/gemma-4-E2B-it", load_4bit=False)

    configs = [
        {"name": "Baseline",                         "layers": []},
        {"name": "Variant 1 (L24 Micro-Tune)",        "layers": [(33, 0.05), (18, -0.04), (24, 0.05)]},
        {"name": "Variant 2 (The -0.05 Threshold)",   "layers": [(33, 0.05), (18, -0.05), (23, 0.06)]},
        {"name": "Variant 3 (Distributed Logic)",     "layers": [(33, 0.05), (18, -0.04), (23, 0.03), (24, 0.03)]},
    ]

    print("\nPre-caching SVD for L18, L23, L24, L33...")
    svd_cache = {}
    for l in [18, 23, 24, 33]:
        dp, U, S, Vh, _ = get_svd(model, l)
        svd_cache[l] = (dp, U, S, Vh)

    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()
    n_syco, n_gsm = 200, 100

    final = {"model": meta["model_id"], "precision": "bfloat16",
             "eval_n": {"syco": n_syco, "gsm": n_gsm}, "results": []}
    out_file = "data/results/micro_hunt_goldilocks.json"
    os.makedirs("data/results", exist_ok=True)

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*40}\n VALIDATING: {name}\n{'='*40}")
        originals = apply_multi_steering(model, cfg["layers"], svd_cache)
        ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=n_syco)
        sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=n_gsm)
        restore_multi_steering(model, originals)
        s_pct = round(_pct(ss), 2)
        g_pct = round(_pct(sg), 2)
        print(f"  [{name}]: Syco={s_pct}% | GSM8K={g_pct}%")
        final["results"].append({"name": name, "config": cfg["layers"],
                                  "syco_pct": s_pct, "gsm_pct": g_pct})
        with open(out_file, "w") as f:
            json.dump(final, f, indent=2)

    print(f"\n  Saved: {out_file} ({round((time.time()-tt0)/60,1)} min)")


def cmd_mistral_hyper_sweep(args):
    """Aggressive early-layer + high-alpha sweep for Mistral-7B-v0.3."""
    t0 = time.time()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    out_file = "data/results/mistral_hyper_sweep.json"
    os.makedirs("data/results", exist_ok=True)

    processed = set()
    existing = []
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                data = json.load(f)
                existing = data.get("grid", [])
                for r in existing:
                    processed.add((r["l_align"], r["a_align"]))
            print(f"  [RESUME] {len(existing)} existing results loaded.")
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint: {e}")

    model, tokenizer, meta = load_model(model_id, load_4bit=True)
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()

    print("\n=== BASELINE (n=300, n_gsm=50) ===")
    bs, _  = eval_sycophancy(model, tokenizer, syco_data, n=300)
    bg, _  = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=50)
    base_syco = _pct(bs)
    base_gsm  = _pct(bg)
    print(f"  => syco={base_syco:.1f}% gsm={base_gsm:.1f}%")

    align_layers = [4, 6, 8, 10, 12, 14]
    align_alphas = [-0.15, -0.3, -0.45, -0.6]
    anchors      = [(24, 0.1), (28, 0.1)]

    target_layers = sorted(set(align_layers + [l for l, _ in anchors]))
    print(f"\n  Pre-caching SVD for {target_layers}...")
    svd_cache = {}
    for l in target_layers:
        dp, U, S, Vh, _ = get_svd(model, l)
        svd_cache[l] = (dp, U, S, Vh)

    results = existing
    count, total = 0, len(align_layers) * len(align_alphas)

    for l_align in align_layers:
        for a_align in align_alphas:
            count += 1
            if (l_align, a_align) in processed:
                print(f"  [{count}/{total}] Skip L{l_align}:{a_align:+.2f} (cached)")
                continue
            configs = anchors + [(l_align, a_align)]
            print(f"\n[{count}/{total}] L{l_align}:{a_align:+.2f}  (Anchors L24/L28:+0.1)")
            originals = apply_multi_steering(model, configs, svd_cache)
            ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=300)
            sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=50)
            restore_multi_steering(model, originals)
            s_pct = _pct(ss)
            g_pct = _pct(sg)
            is_pareto = (s_pct < base_syco) and (g_pct >= base_gsm)
            print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%" + (" *** PARETO ***" if is_pareto else ""))
            results.append({"l_align": l_align, "a_align": a_align, "anchors": anchors,
                             "syco_pct": s_pct, "gsm_pct": g_pct, "is_pareto": is_pareto})
            with open(out_file, "w") as f:
                json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm}, "grid": results,
                            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"\n  Hyper Sweep complete in {round((time.time()-t0)/60,1)} min. Saved: {out_file}")


def cmd_mistral_lever_sweep(args):
    """Fast early-lever discovery for Mistral-7B-v0.3 (N=100)."""
    t0 = time.time()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    out_file = f"data/results/mistral_levers_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("data/results", exist_ok=True)

    model, tokenizer, meta = load_model(model_id, load_4bit=True)
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()

    bs, _ = eval_sycophancy(model, tokenizer, syco_data, n=args.n_syco)
    bg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=args.n_gsm)
    base_syco, base_gsm = _pct(bs), _pct(bg)
    print(f"  Baseline => syco={base_syco:.1f}% gsm={base_gsm:.1f}%")

    align_layers = [3, 4, 5, 6, 7]
    align_alphas = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
    anchors      = [(24, 0.1), (28, 0.1)]

    target_layers = sorted(set(align_layers + [l for l, _ in anchors]))
    svd_cache = {l: get_svd(model, l)[:4] for l in target_layers}

    results = []
    count, total = 0, len(align_layers) * len(align_alphas)
    for l_align in align_layers:
        for a_align in align_alphas:
            count += 1
            configs = anchors + [(l_align, a_align)]
            print(f"[{count}/{total}] L{l_align}:{a_align:+.2f}")
            originals = apply_multi_steering(model, configs, svd_cache)
            ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=args.n_syco)
            sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=args.n_gsm)
            restore_multi_steering(model, originals)
            s_pct, g_pct = _pct(ss), _pct(sg)
            is_pareto = (s_pct < base_syco) and (g_pct >= base_gsm)
            print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%" + (" ***" if is_pareto else ""))
            results.append({"l_align": l_align, "a_align": a_align, "anchors": anchors,
                             "syco_pct": s_pct, "gsm_pct": g_pct, "is_pareto": is_pareto})
            with open(out_file, "w") as f:
                json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm}, "grid": results,
                            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"\n  Lever Sweep complete in {round((time.time()-t0)/60,1)} min. Saved: {out_file}")


def cmd_mistral_recovery_sweep(args):
    """Reasoning-recovery sweep: fixed aligner L14:-0.6 + grid L29-L31."""
    t0 = time.time()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    out_file = "data/results/mistral_recovery_sweep.json"
    os.makedirs("data/results", exist_ok=True)

    model, tokenizer, meta = load_model(model_id, load_4bit=True)
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()

    base_syco, base_gsm = 97.67, 48.0   # established in prior runs
    align_configs = [(14, -0.6)]
    anchors       = [(24, 0.1), (28, 0.1)]
    recovery_layers = [29, 30, 31]
    recovery_alphas = [0.1, 0.2, 0.3, 0.4]

    target_layers = sorted(set([14] + [l for l, _ in anchors] + recovery_layers))
    svd_cache = {l: get_svd(model, l)[:4] for l in target_layers}

    results = []
    count, total = 0, len(recovery_layers) * len(recovery_alphas)
    for l_rec in recovery_layers:
        for a_rec in recovery_alphas:
            count += 1
            configs = align_configs + anchors + [(l_rec, a_rec)]
            print(f"[{count}/{total}] Recovery L{l_rec}:{a_rec:+.2f}")
            originals = apply_multi_steering(model, configs, svd_cache)
            ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=args.n_syco)
            sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=args.n_gsm)
            restore_multi_steering(model, originals)
            s_pct, g_pct = _pct(ss), _pct(sg)
            is_recovery = g_pct > 30.0
            print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%" + (" *** RECOVERY ***" if is_recovery else ""))
            results.append({"l_recovery": l_rec, "a_recovery": a_rec,
                             "syco_pct": s_pct, "gsm_pct": g_pct, "is_recovery": is_recovery})
            with open(out_file, "w") as f:
                json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm},
                            "target_aligner": {"layer": 14, "alpha": -0.6},
                            "grid": results,
                            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"\n  Recovery Sweep complete in {round((time.time()-t0)/60,1)} min. Saved: {out_file}")


def cmd_mistral_precision_sweep(args):
    """Three curated multi-layer configs to break the Mistral sycophancy wall."""
    t0 = time.time()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    out_file = "data/results/mistral_precision_sweep.json"
    os.makedirs("data/results", exist_ok=True)

    model, tokenizer, meta = load_model(model_id, load_4bit=True)
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()

    base_syco, base_gsm = 97.7, 48.0
    print(f"  Hardcoded Baseline => syco={base_syco:.1f}% gsm={base_gsm:.1f}%")

    configs = [
        {"name": "Config 1: Pre-Logic Scrubbing",   "layers": [(12, -0.04), (20, -0.03), (24, +0.08)]},
        {"name": "Config 2: Harmonic Resonance",     "layers": [(16, +0.03), (20, +0.03), (24, +0.05), (28, +0.02)]},
        {"name": "Config 3: Hollow Core",            "layers": [(14, -0.06), (24, +0.09)]},
    ]
    all_layers = set(l for cfg in configs for l, _ in cfg["layers"])
    print(f"  Pre-caching SVD for {sorted(all_layers)}...")
    svd_cache = {l: get_svd(model, l)[:4] for l in all_layers}

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {cfg['name']}")
        originals = apply_multi_steering(model, cfg["layers"], svd_cache)
        ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=300)
        sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=50)
        restore_multi_steering(model, originals)
        s_pct, g_pct = _pct(ss), _pct(sg)
        is_winner = (s_pct < 70.0) and (g_pct >= base_gsm)
        print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%" + (" *** MASSIVE WINNER ***" if is_winner else ""))
        results.append({"name": cfg["name"], "layers": cfg["layers"],
                         "syco_pct": s_pct, "gsm_pct": g_pct, "is_massive_winner": is_winner})
        with open(out_file, "w") as f:
            json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm}, "results": results,
                        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"\n  Precision Sweep done in {round((time.time()-t0)/60,1)} min. Saved: {out_file}")


def cmd_mistral_overclock_sweep(args):
    """High-accuracy overclock sweep (N_GSM=100) to break the 60% GSM8K record."""
    t0 = time.time()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    out_file = "data/results/mistral_overclock_sweep.json"
    os.makedirs("data/results", exist_ok=True)

    model, tokenizer, meta = load_model(model_id, load_4bit=True)
    syco_data = load_sycophancy_data()
    gsm_ds, gsm_idx = load_gsm8k_data()

    base_syco, base_gsm = 97.7, 48.0
    print(f"  Hardcoded Baseline => syco={base_syco:.1f}% gsm={base_gsm:.1f}%")

    configs = [
        {"name": "Config 1: Pure Math Overclock",     "layers": [(20, 0.04), (22, 0.06), (24, 0.10), (28, 0.05)]},
        {"name": "Config 2: Deep Finalizer",          "layers": [(12, -0.02), (24, 0.12), (30, 0.06)]},
        {"name": "Config 3: Contrastive Whisper",     "layers": [(12, -0.01), (14, -0.01), (16, -0.01), (24, 0.15), (28, 0.05)]},
    ]
    all_layers = set(l for cfg in configs for l, _ in cfg["layers"])
    svd_cache = {l: get_svd(model, l)[:4] for l in all_layers}

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {cfg['name']}")
        originals = apply_multi_steering(model, cfg["layers"], svd_cache)
        ss, _ = eval_sycophancy(model, tokenizer, syco_data, n=300)
        sg, _ = eval_gsm8k(model, tokenizer, gsm_ds, gsm_idx, n=100)
        restore_multi_steering(model, originals)
        s_pct, g_pct = _pct(ss), _pct(sg)
        is_record = g_pct >= 60.0
        print(f"  => syco={s_pct:.1f}% gsm={g_pct:.1f}%" + (" !!! RECORD BREAKER !!!" if is_record else ""))
        results.append({"name": cfg["name"], "layers": cfg["layers"],
                         "syco_pct": s_pct, "gsm_pct": g_pct, "is_record_breaker": is_record})
        with open(out_file, "w") as f:
            json.dump({"baseline": {"syco": base_syco, "gsm": base_gsm}, "results": results,
                        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print(f"\n  Overclock Sweep done in {round((time.time()-t0)/60,1)} min. Saved: {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="steer-legacy", description="Legacy Spectral Steering sweeps (research phase)")
    sub = p.add_subparsers(dest="cmd", required=True)

    gh = sub.add_parser("grid-hunt", help="81-config Gemma-4 grid hunt")
    gh.add_argument("--model", default="google/gemma-4-E2B-it")

    sub.add_parser("validate-top", help="High-precision Gemma-4 Pareto validation")

    mhs = sub.add_parser("mistral-hyper-sweep")
    mhs.add_argument("--n-syco", type=int, default=300)
    mhs.add_argument("--n-gsm",  type=int, default=50)

    mls = sub.add_parser("mistral-lever-sweep")
    mls.add_argument("--n-syco", type=int, default=100)
    mls.add_argument("--n-gsm",  type=int, default=50)

    mrs = sub.add_parser("mistral-recovery-sweep")
    mrs.add_argument("--n-syco", type=int, default=300)
    mrs.add_argument("--n-gsm",  type=int, default=50)

    mps = sub.add_parser("mistral-precision-sweep")
    mps.add_argument("--n-syco", type=int, default=300)
    mps.add_argument("--n-gsm",  type=int, default=50)

    mos = sub.add_parser("mistral-overclock-sweep")
    mos.add_argument("--n-syco", type=int, default=300)
    mos.add_argument("--n-gsm",  type=int, default=100)

    args = p.parse_args()
    {
        "grid-hunt":             cmd_grid_hunt,
        "validate-top":          cmd_validate_top,
        "mistral-hyper-sweep":   cmd_mistral_hyper_sweep,
        "mistral-lever-sweep":   cmd_mistral_lever_sweep,
        "mistral-recovery-sweep": cmd_mistral_recovery_sweep,
        "mistral-precision-sweep": cmd_mistral_precision_sweep,
        "mistral-overclock-sweep": cmd_mistral_overclock_sweep,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
