#!/bin/bash
# Quick layer-sweep scan on cached models to find Llama-like steerable ones.
# Uses steer.py layer-sweep (N=50, alpha=-1.5) — takes ~5-10 min per model.
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
LOG="data/results/new_model_scan_log.txt"

echo "[$(date)] === New model scan ===" | tee "$LOG"

sweep() {
    MODEL=$1; N=$2
    TAG=$(echo "$MODEL" | sed 's|.*/||')
    echo "" | tee -a "$LOG"
    echo "[$(date)] ----- $TAG (N=$N) -----" | tee -a "$LOG"
    $PY scripts/steer.py layer-sweep --model "$MODEL" --alpha -1.5 --n-syco "$N" 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done: $TAG" | tee -a "$LOG"
}

# Fastest first: smallest models
sweep "google/gemma-3-1b-it"           50
sweep "Qwen/Qwen2.5-0.5B-Instruct"     50
sweep "Qwen/Qwen2.5-3B-Instruct"       50
sweep "google/gemma-2-2b-it"           50
sweep "microsoft/Phi-3.5-mini-instruct" 50
sweep "Qwen/Qwen2.5-7B-Instruct"       50

echo "" | tee -a "$LOG"
echo "[$(date)] ALL DONE" | tee -a "$LOG"
