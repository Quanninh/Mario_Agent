#!/usr/bin/env bash
set -euo pipefail

cd /home/khanhteo/q_learning_2d_game/Mario_Agent
mkdir -p outputs

echo "=========================================================="
echo " STARTING PARALLEL RL BENCHMARK SUITE (32-CORE CPU)"
echo " Workers: 32 | Episodes: 10,000 | Seeds: 8 (42..49)"
echo " Alpha: 0.10 -> 0.005 | Epsilon Decay: 0.999"
echo "=========================================================="

TOTAL_START=$(date +%s)

echo ""
echo ">>> [1/2] Launching Track A: Levels 1, 2, 3, 4, 5 (96 Jobs)..."
TRACK_A_START=$(date +%s)
.venv/bin/python train.py \
    --agent all \
    --mode all \
    --reward all \
    --levels 1 2 3 4 5 \
    --episodes 10000 \
    --alpha 0.10 \
    --alpha-min 0.005 \
    --epsilon-decay 0.999 \
    --seed 42 \
    --num-seeds 8 \
    --workers 32 \
    --no-progress-bar 2>&1 | tee outputs/benchmark_trackA_levels1-5.log
TRACK_A_END=$(date +%s)
echo ">>> Track A Finished in $(( (TRACK_A_END - TRACK_A_START) / 60 ))m $(( (TRACK_A_END - TRACK_A_START) % 60 ))s."

echo ""
echo ">>> [2/2] Launching Track B: Levels 6, 7 (96 Jobs)..."
TRACK_B_START=$(date +%s)
.venv/bin/python train.py \
    --agent all \
    --mode all \
    --reward all \
    --levels 6 7 \
    --episodes 10000 \
    --alpha 0.10 \
    --alpha-min 0.005 \
    --epsilon-decay 0.999 \
    --seed 42 \
    --num-seeds 8 \
    --workers 32 \
    --no-progress-bar 2>&1 | tee outputs/benchmark_trackB_levels6-7.log
TRACK_B_END=$(date +%s)
echo ">>> Track B Finished in $(( (TRACK_B_END - TRACK_B_START) / 60 ))m $(( (TRACK_B_END - TRACK_B_START) % 60 ))s."

TOTAL_END=$(date +%s)
echo ""
echo "=========================================================="
echo " ALL BENCHMARK TRACKS COMPLETED SUCCESSFULLY!"
echo " Total Elapsed Time: $(( (TOTAL_END - TOTAL_START) / 60 ))m $(( (TOTAL_END - TOTAL_START) % 60 ))s."
echo " Logs saved to: outputs/benchmark_trackA_levels1-5.log"
echo "                outputs/benchmark_trackB_levels6-7.log"
echo "=========================================================="
