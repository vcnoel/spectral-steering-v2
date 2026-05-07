# Sycophancy Is Often a Single-Layer Phenomenon

**Spectral Diagnosis and Training-Free Weight Surgery in LLMs**

*NeurIPS 2026 submission*

---

## What this is

Instruction-tuned LLMs frequently agree with users even when users are factually wrong — a behaviour called sycophancy. We show that in six of seven tested models, this behaviour is **mediated by the dominant singular subspace of a single MLP weight matrix**, and can be suppressed or induced by a closed-form rescaling of that subspace. No training, no data, no inference overhead.

The headline result: on Gemma-4-E2B-it, we achieve a **strict Pareto improvement** — less sycophancy *and* better reasoning simultaneously. On Llama-3.2-3B/8B, sycophancy localises to the same absolute layer (L7) across a 3× parameter gap, with SNR values within 8% of each other.

---

## Method

For the `mlp.down_proj` matrix at layer ℓ with thin SVD W = UΣVᵀ, we apply:

```
σᵢ' = σᵢ · (1 + α · (σᵢ − μ_σ) / s_σ)
W' = U Σ' Vᵀ
```

α < 0 compresses the dominant spectral directions (→ reduces sycophancy).  
α > 0 amplifies them (→ induces sycophancy on neutral inputs).

The target layer is selected by the per-layer spectral SNR = σ₁(ℓ) / σ̃(ℓ), computable in seconds from weights alone. Safe operating regime: |α| ≪ 1/SNR_ℓ.

---

## Key findings

| Finding | Result |
|---------|--------|
| Cross-scale localisation | Llama-3B and 8B both peak at **L7** — same absolute layer, SNR within 8% |
| Strict Pareto | Gemma-4-E2B-it: −6.1% sycophancy **and** +8.8% GSM8K simultaneously |
| Storage taxonomy | Models split into *localised* (surgery works) vs *distributed* (Mistral-7B) |
| ΔW fingerprint | Mistral L31: cos(ΔW_u1, W_u1) = 0.812 vs Llama L7: 0.025 — explains why surgery fails on Mistral |
| SNR dosing guide | Predicts which layers collapse at what dose — no behavioural testing needed |
| Independent circuits | RepE and spectral surgery target near-orthogonal directions (cos = 0.030) |

---

## Models tested

| Model | Family | Storage class | Dominant layer | SNR | Δ Sycophancy |
|-------|--------|---------------|----------------|-----|--------------|
| Llama-3.2-1B | Llama | Localised | L14 | 2.77 | −31pp |
| Llama-3.2-3B | Llama | Localised | L7 | 4.87 | −50pp (A/B); −17pp safe |
| Llama-3.1-8B | Llama | Localised | L7 | 5.25 | −38pp |
| Gemma-4-E2B-it | Gemma 4 | Localised | L18+L24+L33 | multi | **Strict Pareto** |
| Phi-4-mini | Phi | Localised (entangled) | L10 | 4.59 | −7pp |
| Mistral-7B-v0.3 | Mistral | Distributed | — | <3.5 | Not reducible |
| Qwen2.5-3B | Qwen | Localised | L6 | 6.32 | −72pp |

---

## Reproduce

```bash
# Install dependencies
conda env create -f environment.yml
conda activate gsp

# Layer sweep (find dominant layer for a model)
python scripts/steer.py layer-sweep \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --alpha -1.5 --n-syco 50

# Validate at a specific layer/alpha
python scripts/steer.py validate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --config 7:-0.5 --n-syco 100 --n-gsm 50

# Multi-layer hunt (Gemma-style Pareto search)
python scripts/steer.py combo-hunt \
  --model google/gemma-2-2b-it \
  --n-syco 200

# ΔW fingerprint (explains storage-class taxonomy)
python scripts/run_delta_w_comparison.py
```

Results are written to `data/results/`.

---

## Repo structure

```
scripts/
  steer.py                    — main experiment runner (sweep, validate, hunt, ablate)
  run_family_expansion.sh     — layer sweeps on new model families
  run_activation_ablation.py  — causal activation-space evidence
  run_delta_w_comparison.py   — ΔW analysis (localized vs distributed)
  run_repe_baseline.py        — RepE controlled comparison
  run_open_ended_judge.py     — capability collapse diagnostic
  run_mistral_multilayer.py   — Mistral multi-layer + ΔW sweep
src/                          — library code
configs/                      — model/eval configs
spectral_steering_paper.tex   — paper source
environment.yml               — conda environment (gsp)
```

---

## Paper

`spectral_steering_paper.tex` — compile with pdflatex or your preferred LaTeX engine.

Appendices cover: full layer sweep tables, Gemma-4 hyperparameter search, Phi-4 entanglement analysis, Mistral spectral fingerprint, CCA and Procrustes analysis, response examples, method comparison, capability collapse diagnostic, RepE controlled experiment, Mistral ΔW analysis with Llama contrastive control.
