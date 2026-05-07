#!/bin/bash
# Expand storage-class taxonomy to new model families.
# N=50, alpha=-1.5 fast scan. gsp environment.
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
LOG="data/results/family_expansion_log.txt"
mkdir -p data/results

echo "[$(date)] === Family expansion sweep ===" | tee "$LOG"

sweep() {
    MODEL=$1
    TAG=$(echo "$MODEL" | sed 's|.*/||')
    echo "" | tee -a "$LOG"
    echo "[$(date)] ----- $TAG -----" | tee -a "$LOG"
    $PY scripts/steer.py layer-sweep --model "$MODEL" --alpha -1.5 --n-syco 50 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done: $TAG" | tee -a "$LOG"
}

# New families not yet in the paper:
sweep "google/gemma-2-2b-it"          # Gemma 2 (different from Gemma 4)
sweep "Qwen/Qwen2.5-7B-Instruct"      # Qwen 7B (scales up existing Qwen family)
sweep "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # SmolLM family

echo "" | tee -a "$LOG"
echo "[$(date)] ALL DONE" | tee -a "$LOG"
