#!/usr/bin/env bash
set -euo pipefail

cd /home/khanhteo/q_learning_2d_game/Mario_Agent
mkdir -p outputs

COMMON_ARGS="--levels 1 2 3 4 5 --episodes 10000 --alpha 0.10 --alpha-min 0.005 --epsilon-decay 0.999 --no-progress-bar"

echo ">>> Resuming 28 missing Track A (Levels 1-5) jobs in parallel..."
.venv/bin/python train.py --agent tabular --mode curriculum --reward sparse --seed 45 --num-seeds 1 --workers 1 $COMMON_ARGS > outputs/resume_tab_cur_sparse_45.log 2>&1 &
.venv/bin/python train.py --agent tabular --mode curriculum --reward sparse --seed 47 --num-seeds 1 --workers 1 $COMMON_ARGS > outputs/resume_tab_cur_sparse_47.log 2>&1 &
.venv/bin/python train.py --agent hybrid --mode curriculum --reward sparse --seed 42 --num-seeds 8 --workers 8 $COMMON_ARGS > outputs/resume_hyb_cur_sparse.log 2>&1 &
.venv/bin/python train.py --agent hybrid --mode separate --reward balanced --seed 48 --num-seeds 2 --workers 2 $COMMON_ARGS > outputs/resume_hyb_sep_balanced.log 2>&1 &
.venv/bin/python train.py --agent hybrid --mode separate --reward sparse --seed 42 --num-seeds 8 --workers 8 $COMMON_ARGS > outputs/resume_hyb_sep_sparse.log 2>&1 &
.venv/bin/python train.py --agent hybrid --mode separate --reward dense --seed 42 --num-seeds 8 --workers 8 $COMMON_ARGS > outputs/resume_hyb_sep_dense.log 2>&1 &

wait
echo ">>> All 28 missing Track A jobs completed!"

echo ">>> Launching Track B: Levels 6, 7 (96 Jobs)..."
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

echo ">>> ALL DONE!"
