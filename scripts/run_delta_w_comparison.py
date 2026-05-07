# dW comparison: Llama vs Mistral.
# cos(dW_u1, W_instruct_u1) at dominant sycophancy layer.
# Hypothesis: Mistral=high (RLHF co-located compliance with dominant load),
#             Llama=low (RLHF stored compliance orthogonally).
import json, torch, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM

RESULTS = Path("data/results")
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "family": "Llama",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "base": "meta-llama/Llama-3.2-3B",
        "layer": 7,
        "storage_class": "localized",
    },
    {
        "family": "Mistral",
        "instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "base": "mistralai/Mistral-7B-v0.1",
        "layer": 31,
        "storage_class": "distributed",
    },
]


def get_down_proj(model, layer):
    return model.model.layers[layer].mlp.down_proj.weight.data.float()


def top_left_sv(W):
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, 0]


def analyze_pair(cfg):
    print(f"\n=== {cfg['family']} ===")
    print(f"  Loading instruct: {cfg['instruct']}")
    m_inst = AutoModelForCausalLM.from_pretrained(
        cfg["instruct"], torch_dtype=torch.float16, device_map="cpu"
    )
    W_inst = get_down_proj(m_inst, cfg["layer"])
    U1_inst = top_left_sv(W_inst)
    del m_inst; torch.cuda.empty_cache()

    print(f"  Loading base: {cfg['base']}")
    m_base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.float16, device_map="cpu"
    )
    W_base = get_down_proj(m_base, cfg["layer"])
    del m_base; torch.cuda.empty_cache()

    delta_W = W_inst - W_base
    U1_delta = top_left_sv(delta_W)

    sigma1_delta = torch.linalg.svdvals(delta_W)[0].item()
    sigma_median_delta = torch.linalg.svdvals(delta_W).median().item()
    delta_snr = sigma1_delta / sigma_median_delta

    cos = (U1_delta @ U1_inst).abs().item()
    print("  Layer L%d: cos(dW_u1, W_inst_u1) = %.4f" % (cfg['layer'], cos))
    print("  dW SNR = %.3f (sigma1=%.4f)" % (delta_snr, sigma1_delta))

    return {
        "family": cfg["family"],
        "instruct_model": cfg["instruct"],
        "base_model": cfg["base"],
        "layer": cfg["layer"],
        "storage_class": cfg["storage_class"],
        "cos_dW_u1_vs_W_u1": cos,
        "delta_snr": delta_snr,
        "sigma1_delta": sigma1_delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-only", action="store_true")
    parser.add_argument("--mistral-only", action="store_true")
    args = parser.parse_args()

    configs = CONFIGS
    if args.llama_only:
        configs = [c for c in CONFIGS if c["family"] == "Llama"]
    if args.mistral_only:
        configs = [c for c in CONFIGS if c["family"] == "Mistral"]

    results = []
    for cfg in configs:
        try:
            r = analyze_pair(cfg)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"family": cfg["family"], "error": str(e)})

    out = RESULTS / "delta_w_comparison.json"
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\n=== SUMMARY ===")
    for r in results:
        if "error" not in r:
            print("  %s L%d: cos=%.4f (storage: %s)" % (
                r['family'], r['layer'], r['cos_dW_u1_vs_W_u1'], r['storage_class']))
    print(f"Saved to {out}")
    print("\nInterpretation: high cos = RLHF co-located compliance with dominant spectral load.")
    print("Mistral should be high, Llama should be low -- explaining the taxonomy.")


if __name__ == "__main__":
    main()
