#!/bin/bash
# Ablation battery for reviewer responses.
# Runs sequentially (single GPU). All results land in data/results/.
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
LOG="data/results/ablations_log.txt"
mkdir -p data/results

log() { echo "[$(date)] $1" | tee -a "$LOG"; }

log "=== Ablation battery start ==="

# 1. Matrix ablation: down_proj vs up_proj vs gate_proj at L7 (Llama-3.2-3B)
#    Answers: "Why down_proj specifically?"
log "--- [1/5] Matrix ablation: down_proj vs up_proj vs gate_proj ---"
$PY scripts/steer.py matrix-ablate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --alphas="-0.5,-1.0,-1.5" \
  --n-syco 50 --n-gsm 30 2>&1 | tee -a "$LOG"
log "Done: matrix-ablate"

# 2. Rank-k deflation: k=1 vs k=5 vs k=10 at L7 (Llama-3.2-3B)
#    Answers: "Is the effect concentrated in top-1 singular value?"
log "--- [2/5] Rank-k deflation ablation ---"
$PY scripts/steer.py ablate \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --layer 7 --alpha -1.0 --n-samples 50 2>&1 | tee -a "$LOG"
log "Done: rank-k ablate"

# 3. Open-ended judge at MILD alpha (-0.5) on Llama-3B
#    Answers: "Does mild surgery produce coherent sycophancy reduction?"
log "--- [3/5] Open-ended judge at mild alpha (-0.5) ---"
$PY scripts/run_open_ended_judge.py --alpha -0.5 --layer 7 2>&1 | tee -a "$LOG"
log "Done: open-ended mild"

# 4. Qwen2.5-7B validate at mild alpha (N=100 with GSM8K)
#    Answers: "N=50 too small — is the -76pp real?"
log "--- [4/5] Qwen2.5-7B N=100 validate at mild alpha ---"
$PY scripts/steer.py validate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --config "16:-0.7" \
  --n-syco 100 --n-gsm 50 2>&1 | tee -a "$LOG"
log "Done: Qwen-7B validate"

# 5. Mistral-7B-Instruct-v0.1 layer sweep
#    Answers: "Is distributed pattern RLHF-recipe-specific or architectural?"
log "--- [5/5] Mistral-7B-Instruct-v0.1 layer sweep ---"
$PY scripts/steer.py layer-sweep \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --alpha -1.5 --n-syco 50 2>&1 | tee -a "$LOG"
log "Done: Mistral-v0.1 sweep"

log "=== ALL DONE ==="
