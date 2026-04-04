# Spectral Steering v2: High-Throughput Alignment Calibration

## Overview
Spectral Steering v2 is a framework for high-precision, zero-retraining alignment calibration of Large Language Models (LLMs). By performing rank-1 spectral deflation of behavioral eigenvectors—identified through contrastive activation pairs—the system can modulate specific behavioral traits (e.g., Sycophancy, Refusal, Toxicity) without degrading the model's fundamental reasoning capabilities.

This repository implements the "75th Percentile Depth Rule," identifying the penultimate reasoning layers (e.g., Layer 24 in 32-layer models) as the optimal intervention point for steering internal truth representations.

---

## Key Results
Empirical validation across four major model families (Llama, Mistral, Qwen, Phi) using 4-bit (NF4) quantized inference:

| Model Family | Param Scale | Baseline Error | Best Alpha | Steering Error | Recovery Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3.1** | 8B | 8.0% | 0.5 (Smooth) | **0.0%** | **Perfect** |
| **Qwen-2.5** | 7B | 10.0% | 0.5 / -1.0 | **0.0%** | **Perfect** |
| **Phi-3.5-Mini** | 3.8B | 6.0% | 0.5 / -1.0 | **0.0%** | **Perfect** |
| **Mistral-7B** | 7B | 0.0% | N/A | **0.0%** | **Safe** |

- **Zero-Tax Capability**: GSM8K accuracy remained at the baseline floor across all successful steering trajectories, proving that spectral steering preserves the model's reasoning logic.
- **Optimal Depth**: Interventions at 75% depth (Layer 24 for 32-layer models) consistently yielded the highest alignment gains across diverse architectures.

---

## Installation
Ensure you have a CUDA-enabled GPU and a Python 3.10+ environment.

```bash
conda create -n gsp python=3.10
conda activate gsp
pip install -r requirements.txt
```

---

## Core Pipeline

### 1. Eigenvector Extraction
Analyze the behavioral manifold by extracting the principal component of contrastive activation pairs.
```bash
python scripts/run_phase1.py extract --model meta-llama/Llama-3.1-8B-Instruct --n_pairs 100 --load_bit4
```

### 2. Spectral Deflation (Surgery)
Modify the model weights using the rank-1 deflation/sharpening update. The script handles [out, in] dimension mismatches for architecture-specific MLP blocks (e.g., Llama's `down_proj`).
```bash
python scripts/run_phase1.py deflate --layer 24 --alpha 0.5 --model meta-llama/Llama-3.1-8B-Instruct --load_bit4
```

### 3. Systematic Evaluation
Evaluate the modified model on alignment (Sycophancy) and capability (GSM8K) benchmarks.
```bash
python scripts/run_phase1.py eval --benchmark sycophancy --model_path ./deflated_model --load_bit4
```

---

## Theoretical Framework
The method utilizes the Marchenko-Pastur distribution to identify signal-bearing eigenvectors.
- **Sharpening ($\alpha < 0$)**: Amplifies the principal component to force behavioral transitions in rigid, low-parameter models.
- **Smoothing ($\alpha > 0$)**: Deflates the behavioral noise in higher-parameter models to recover the alignment floor of more compact, well-regulated models.

---

## Repository Structure
- `scripts/`: Implementation of the extraction, deflation, and evaluation pipeline.
- `configs/`: YAML definitions for model families and task benchmarks.
- `src/`: Core library for spectral decomposition, Marchenko-Pastur thresholding, and weight surgery.
- `scaling_results.json`: Comprehensive trajectory log for all model-family sweeps.

---

## Citation
If you use this framework in your research, please refer to the "Geometry of Reason" spectral steering methodology.
