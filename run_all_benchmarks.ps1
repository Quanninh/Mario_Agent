# run_all_benchmarks.ps1
# Parallel RL Benchmark Suite for Hoá's Midnight Escape
# Designed for 32-core CPU using 32 parallel workers

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " STARTING PARALLEL RL BENCHMARK SUITE (32-CORE CPU)       " -ForegroundColor Cyan
Write-Host " Workers: 32 | Episodes: 10,000 | Seeds: 8 (42..49)      " -ForegroundColor Cyan
Write-Host " Alpha: 0.10 -> 0.005 | Epsilon Decay: 0.999             " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TotalStartTime = Get-Date

if (-not (Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

# 1. Run Track A (Levels 1 to 5)
Write-Host "`n>>> [1/2] Launching Track A: Levels 1, 2, 3, 4, 5 (96 Jobs)..." -ForegroundColor Yellow
$TrackAStart = Get-Date
python train.py `
    --agent all `
    --mode all `
    --reward all `
    --levels 1 2 3 4 5 `
    --episodes 10000 `
    --alpha 0.10 `
    --alpha-min 0.005 `
    --epsilon-decay 0.999 `
    --seed 42 `
    --num-seeds 8 `
    --workers 32 `
    --no-progress-bar *>&1 | Tee-Object -FilePath "outputs\benchmark_trackA_levels1-5.log"
$TrackADuration = (Get-Date) - $TrackAStart
Write-Host ">>> Track A Finished in: $($TrackADuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green

# 2. Run Track B (Levels 6, 7, 8)
Write-Host "`n>>> [2/2] Launching Track B: Levels 6, 7, 8 (96 Jobs)..." -ForegroundColor Yellow
$TrackBStart = Get-Date
python train.py `
    --agent all `
    --mode all `
    --reward all `
    --levels 6 7 8 `
    --episodes 10000 `
    --alpha 0.10 `
    --alpha-min 0.005 `
    --epsilon-decay 0.999 `
    --seed 42 `
    --num-seeds 8 `
    --workers 32 `
    --no-progress-bar *>&1 | Tee-Object -FilePath "outputs\benchmark_trackB_levels6-8.log"
$TrackBDuration = (Get-Date) - $TrackBStart
Write-Host ">>> Track B Finished in: $($TrackBDuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green

$TotalDuration = (Get-Date) - $TotalStartTime
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " ALL BENCHMARK TRACKS COMPLETED SUCCESSFULLY!             " -ForegroundColor Green
Write-Host " Total Elapsed Time: $($TotalDuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green
Write-Host " Logs saved to: outputs\benchmark_trackA_levels1-5.log    " -ForegroundColor Cyan
Write-Host "                outputs\benchmark_trackB_levels6-8.log    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
