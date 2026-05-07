#!/bin/bash
PY="C:/Users/valno/miniconda3/envs/gsp/python.exe"
SCRIPT="scripts/steer.py"
LOG="data/results/llama_sweeps_log.txt"

echo "[$(date)] Starting Llama family sweep chain" | tee -a "$LOG"

run_sweep() {
    MODEL=$1; ALPHA=$2; NVAL=$3
    TAG=$(echo "$MODEL" | sed 's|.*/||')
    echo "" | tee -a "$LOG"
    echo "[$(date)] ===== $TAG | alpha=$ALPHA | N=$NVAL =====" | tee -a "$LOG"
    $PY $SCRIPT layer-sweep --model "$MODEL" --alpha "$ALPHA" --n-syco "$NVAL" 2>&1 | tee -a "$LOG"
    echo "[$(date)] Done: $TAG alpha=$ALPHA" | tee -a "$LOG"
}

# Llama 3.2 1B — fastest, run at all alphas with N=200
run_sweep "meta-llama/Llama-3.2-1B-Instruct" -1.5 200
run_sweep "meta-llama/Llama-3.2-1B-Instruct" -1.0 200
run_sweep "meta-llama/Llama-3.2-1B-Instruct" -0.5 200
run_sweep "meta-llama/Llama-3.2-1B-Instruct"  0.5 200
run_sweep "meta-llama/Llama-3.2-1B-Instruct"  1.0 200
run_sweep "meta-llama/Llama-3.2-1B-Instruct"  1.5 200

# Llama 3.2 3B — medium speed, N=100
run_sweep "meta-llama/Llama-3.2-3B-Instruct" -1.5 100
run_sweep "meta-llama/Llama-3.2-3B-Instruct" -1.0 100
run_sweep "meta-llama/Llama-3.2-3B-Instruct" -0.5 100
run_sweep "meta-llama/Llama-3.2-3B-Instruct"  0.5 100
run_sweep "meta-llama/Llama-3.2-3B-Instruct"  1.0 100
run_sweep "meta-llama/Llama-3.2-3B-Instruct"  1.5 100

# Llama 3.1 8B — slower, N=50 (we already have +1.5)
run_sweep "meta-llama/Llama-3.1-8B-Instruct" -1.5 50
run_sweep "meta-llama/Llama-3.1-8B-Instruct" -1.0 50
run_sweep "meta-llama/Llama-3.1-8B-Instruct" -0.5 50

echo "" | tee -a "$LOG"
echo "[$(date)] ALL SWEEPS COMPLETE" | tee -a "$LOG"
