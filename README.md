# Spectral Steering V2

The V2 iteration focuses on strict methodological evaluation.

## Environment Setup

Run the following to initialize the `steering` environment that handles both the V1 and V2 repositories.

```bash
conda env create -f environment.yml
conda activate steering
```

> **Note on Windows vs Linux/WSL**: `vllm` provides huge text-generation speedups, but native Windows builds can be unstable. Therefore, `environment.yml` disables `vllm` by default. You can uncomment it before installing if you have compatible wheels or are running in WSL. We will fall back to `accelerate` otherwise.

## Usage: CLI Execution

We have standardized operations under `main.py`! You no longer need to invoke specific python scripts inside `scripts/`. 

### V2 Orchestrator Commands:

1. **Evaluate Phase 1 (Sycophancy)**
```bash
python main.py phase1
```

2. **Calculate Full Capability Tax**
```bash
python main.py capability --models llama-3.1-8b
```

3. **Mechanistic Eigenvector Analysis**
```bash
python main.py eigenvector
```

*(You can append `--use-vllm` to any command to accelerate generation loops).*
