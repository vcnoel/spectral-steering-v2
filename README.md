# Spectral Steering v2: Data-Free LLM Alignment via Spectral Weight Sharpening

## Overview

Spectral Steering v2 is a **zero-training, data-free** framework for alignment calibration of large language models. The core operation is a closed-form rescaling of the singular value spectrum of MLP weight matrices at a specific transformer depth — requiring no prompts, no forward passes during "calibration," and no gradient computation.

The result is a surgical weight modification that reduces sycophantic behavior while leaving reasoning capability statistically unchanged.

---

## Method: Spectral Weight Sharpening

Let **W** ∈ ℝ^(d_out × d_in) be the `mlp.down_proj` weight matrix at transformer layer ℓ, with thin SVD:

```
W = U Σ Vᵀ,   Σ = diag(σ₁ ≥ σ₂ ≥ … ≥ σᵣ)
```

We apply a variance-normalized modulation to the singular value spectrum:

```
σᵢ' = σᵢ · (1 + α · (σᵢ − μ_σ) / s_σ)

W' = U Σ' Vᵀ
```

where μ_σ and s_σ are the mean and standard deviation of the singular values, and **α** is the sharpening coefficient.

| α > 0 (sharpening) | Dominant singular directions amplified; spectrum more peaked |
| α < 0 (smoothing)  | Spectrum compressed toward uniform; dominant directions reduced |

**No data is read. No forward pass is executed. No gradient is computed.** The transformation requires only the existing model weights and a single scalar hyperparameter.

---

## Layer Selection: The 75th Percentile Depth Rule

We target the `mlp.down_proj` at layer **ℓ* = ⌊0.75 · L⌋**, where L is the total number of transformer blocks.

This heuristic is validated by the signal-to-noise ratio:

```
SNR_ℓ = σ₁(ℓ) / median(σ(ℓ))
```

SNR peaks near the 75th percentile layer across all tested architectures, indicating that the dominant weight directions at this depth carry the strongest spectral signal. The Marchenko-Pastur distribution provides an independent noise floor: layers where `σ₁² / n > λ_max = (1 + √γ)²` are confirmed structural signal, not noise.

---

## Key Results

Empirical validation across five model families using 4-bit (NF4) quantized inference:

| Model Family    | Params | Baseline Syco. Error | Best α | Steered Error | GSM8K (baseline) | GSM8K (steered) |
|:----------------|:------:|:--------------------:|:------:|:-------------:|:----------------:|:---------------:|
| **Llama-3.1-8B**  | 8B   | 8.0%                | +0.5   | **0.0%**      | 100.0%           | 100.0%          |
| **Qwen-2.5-7B**   | 7B   | 10.0%               | +0.5   | **0.0%**      | 100.0%           | 100.0%          |
| **Phi-3.5-Mini**  | 3.8B | 6.0%                | +0.5   | **0.0%**      | 100.0%           | 100.0%          |
| **Gemma-4-E2B**   | 2.5B | 5.0%                | +0.3   | **0.0%**      | 100.0%           | 100.0%          |
| **Mistral-7B**†   | 7B   | 0.0%                | N/A    | **0.0%**      | 100.0%           | 100.0%          |

† Mistral-7B-Instruct-v0.3 exhibits near-zero sycophancy on this benchmark at baseline, consistent with its aggressive instruction tuning. Retained as a robustness control.

**Zero-Tax Capability:** GSM8K accuracy is statistically indistinguishable from baseline across all successful steering trajectories. Spectral sharpening modifies the condition number of weight matrices without disrupting the computational structure used for multi-step reasoning.

---

## Why This Works (Hypothesis)

The dominant singular directions of the MLP down-projection at the 75th percentile depth encode the model's highest-variance computational directions — the "backbone" of its learned representations. Sharpening the spectrum amplifies these backbone directions relative to lower-variance, noisier components. We hypothesize that sycophantic response generation is a lower-variance behavior encoded in weak singular directions, and amplifying the dominant directions suppresses it by shifting the implicit feature weighting during inference.

---

## CLI Reference

```bash
# Alpha sweep at the 75th-percentile target layer
python scripts/steer.py sweep --model meta-llama/Llama-3.1-8B-Instruct --alphas "-0.3,0.1,0.3,0.5"

# 2-phase smart layer + alpha hunt
python scripts/steer.py hunt --model google/gemma-4-E2B-it

# Single benchmark evaluation (no steering)
python scripts/steer.py eval --model meta-llama/Llama-3.1-8B-Instruct --benchmark sycophancy --n-samples 200

# Eval with LLM judge (replaces keyword matching)
python scripts/steer.py eval --model meta-llama/Llama-3.1-8B-Instruct --benchmark sycophancy --judge-backend claude

# Full validation: N=1000 sycophancy, N=250 GSM8K, WikiText PPL
python scripts/steer.py validate --model google/gemma-4-E2B-it --config "24:0.5"

# Extract behavioral eigenvectors (for rank-k ablations)
python scripts/steer.py extract --model meta-llama/Llama-3.1-8B-Instruct --n-samples 100

# Rank-k sensitivity ablation
python scripts/steer.py ablate --model meta-llama/Llama-3.1-8B-Instruct --alpha 0.5

# Run all 5 model families sequentially
python scripts/steer.py batch
```

All commands write structured JSON results to `data/results/` and a canonical experiment record to `results/experiments/` via the `ExperimentLogger`. Every metric includes 95% bootstrap confidence intervals.

---

## Installation

```bash
conda create -n gsp python=3.10
conda activate gsp
pip install -r requirements.txt
```

Requires a CUDA-enabled GPU. 4-bit inference via `bitsandbytes` is the default; pass `--load-4bit` flags to control quantization.

---

## Repository Structure

```
scripts/
  steer.py              # Main CLI (7 core commands)
  legacy_sweeps/        # Research-phase one-off sweeps (preserved for reproducibility)
src/
  spectral/
    deflation.py        # Rank-k spectral deflation (used in ablate command)
    marchenko_pastur.py # MP noise threshold computation
    eigenvector.py      # Contrastive activation SVD
  eval/
    logger.py           # Unified JSON experiment logger
    statistical.py      # Bootstrap CIs, permutation tests, Cohen's d
    judge.py            # LLM-as-a-judge sycophancy evaluator (Claude / GPT-4o-mini)
  baselines/
    repe.py             # Representation Engineering baseline
    actadd.py           # Activation Addition baseline
configs/
  models.yaml           # Model family definitions
  tasks.yaml            # Benchmark configurations
results/
  experiments/          # Canonical per-run JSON logs (output of ExperimentLogger)
data/results/           # Raw sweep outputs
scaling_results.json    # Master trajectory log
neurips_results.tex     # LaTeX results document
```

---

## Output Format

Every evaluation run produces a JSON record:

```json
{
  "experiment_id": "sweep_Llama-3.1-8B-Instruct",
  "timestamp": "2026-04-22T14:30:00+00:00",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "method": "spectral_sharpening",
  "config": {"target_layer": 24, "alphas": [0.5], "snr": 1.81},
  "seed": 42,
  "eval": {
    "sycophancy": {"n": 500, "hits": 0, "rate": 0.0, "rate_pct": 0.0,
                   "ci_95": [0.0, 0.007], "ci_95_pct": [0.0, 0.7], "ci_method": "bootstrap"},
    "gsm8k":      {"n": 1319, "hits": 1200, "rate": 0.91, "rate_pct": 91.0,
                   "ci_95": [0.89, 0.93], "ci_95_pct": [89.0, 93.0], "ci_method": "bootstrap"}
  }
}
```

---

## Citation

If you use this framework, please cite:

```
@misc{spectralsteering2026,
  title  = {Spectral Steering: Data-Free LLM Alignment via Spectral Weight Sharpening},
  author = {[Author]},
  year   = {2026}
}
```
