#!/bin/bash
# Priority model scan — 3 models most likely to show Llama-like spectral structure.
# N=50, alpha=-1.5, one at a time.
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
LOG="data/results/priority_scan_log.txt"

echo "[$(date)] === Priority model scan ===" | tee "$LOG"

sweep() {
    MODEL=$1
    TAG=$(echo "$MODEL" | sed 's|.*/||')
    echo "" | tee -a "$LOG"
    echo "[$(date)] ----- $TAG -----" | tee -a "$LOG"
    $PY scripts/steer.py layer-sweep --model "$MODEL" --alpha -1.5 --n-syco 50 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done: $TAG" | tee -a "$LOG"
}

sweep "Qwen/Qwen2.5-0.5B-Instruct"
sweep "Qwen/Qwen2.5-3B-Instruct"
sweep "microsoft/Phi-3.5-mini-instruct"

echo "" | tee -a "$LOG"
echo "[$(date)] ALL DONE" | tee -a "$LOG"
