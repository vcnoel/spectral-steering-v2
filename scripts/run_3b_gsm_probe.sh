#!/bin/bash
# Run GSM8K+sycophancy validation on 3B L7 at three negative alphas
# N=50 GSM (fast), N=100 syco (confirm reduction holds at validation scale)
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
LOG="data/results/gsm_probe_3b_log.txt"

echo "[$(date)] Starting 3B L7 GSM8K probe" | tee "$LOG"

for ALPHA in -0.5 -1.0 -1.5; do
    ATAG=$(echo "$ALPHA" | sed 's/-/m/' | sed 's/\./p/')
    echo "" | tee -a "$LOG"
    echo "[$(date)] === alpha=$ALPHA ===" | tee -a "$LOG"
    $PY scripts/steer.py validate \
        --model "meta-llama/Llama-3.2-3B-Instruct" \
        --config "7:$ALPHA" \
        --n-syco 100 \
        --n-gsm 50 \
        --out-tag "L7_${ATAG}" 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done alpha=$ALPHA" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[$(date)] ALL DONE" | tee -a "$LOG"
