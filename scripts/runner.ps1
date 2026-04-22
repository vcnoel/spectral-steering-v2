# runner.ps1
$env:PYTHONPATH = "."
$PY = "C:\Users\valno\miniconda3\envs\gemma_spectral\python.exe"
$MODEL = "google/gemma-4-E2B-it"

Write-Host "--- [1/4] Behavioral Extraction (N=200) ---"
& $PY scripts/steer.py extract --model $MODEL --n-samples 200 --load-4bit

Write-Host "--- [2/4] Structural Ablation (N=150) ---"
& $PY scripts/steer.py ablate --model $MODEL --n-samples 150 --load-4bit

Write-Host "--- [3/4] Robustness Audit (N=150) ---"
& $PY scripts/steer.py robust --model $MODEL --n-samples 150 --load-4bit

Write-Host "--- [4/4] External Benchmarks (N=150) ---"
& $PY scripts/steer.py benchmark --model $MODEL --n-samples 150 --load-4bit --truthfulqa --mmlu

Write-Host "--- WORK COMPLETE ---"
