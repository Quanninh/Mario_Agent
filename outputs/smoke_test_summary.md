# Phase 5 Smoke Test Summary Report (Discretizer v3 & Option B Replay)

**Execution Date**: 2026-11-21  
**Environment**: Hoá's Midnight Escape (v2/v3 — 5 Progressive Graveyard Levels)  
**Command Executed**: `python train.py --smoke-test --no-progress-bar`  
**Scope**: 6 Experiments = 2 Modes (`curriculum`, `separate`) × 3 Reward Strategies (`balanced`, `sparse`, `dense`)  
**State Space**: 1,152 maximum discrete states ($2 \times 3 \times 4 \times 2 \times 3 \times 4 \times 2$)

---

## 1. Generated Artifacts & File Inventory

All 12 Member 4 data boards and all canonical checkpoints were generated and verified:

| File Name | File Path | Status |
| :--- | :--- | :---: |
| `training_board_curriculum_balanced.json` | [`outputs/training_board_curriculum_balanced.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_curriculum_balanced.json) | Verified |
| `training_metrics_curriculum_balanced.csv` | [`outputs/training_metrics_curriculum_balanced.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_curriculum_balanced.csv) | Verified |
| `training_board_curriculum_sparse.json` | [`outputs/training_board_curriculum_sparse.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_curriculum_sparse.json) | Verified |
| `training_metrics_curriculum_sparse.csv` | [`outputs/training_metrics_curriculum_sparse.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_curriculum_sparse.csv) | Verified |
| `training_board_curriculum_dense.json` | [`outputs/training_board_curriculum_dense.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_curriculum_dense.json) | Verified |
| `training_metrics_curriculum_dense.csv` | [`outputs/training_metrics_curriculum_dense.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_curriculum_dense.csv) | Verified |
| `training_board_separate_balanced.json` | [`outputs/training_board_separate_balanced.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_separate_balanced.json) | Verified |
| `training_metrics_separate_balanced.csv` | [`outputs/training_metrics_separate_balanced.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_separate_balanced.csv) | Verified |
| `training_board_separate_sparse.json` | [`outputs/training_board_separate_sparse.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_separate_sparse.json) | Verified |
| `training_metrics_separate_sparse.csv` | [`outputs/training_metrics_separate_sparse.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_separate_sparse.csv) | Verified |
| `training_board_separate_dense.json` | [`outputs/training_board_separate_dense.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_board_separate_dense.json) | Verified |
| `training_metrics_separate_dense.csv` | [`outputs/training_metrics_separate_dense.csv`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs/training_metrics_separate_dense.csv) | Verified |
| Canonical `q_table_best.{json,pkl}` | [`checkpoints/q_table_best.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/checkpoints/q_table_best.json) | Verified |
| Canonical `q_table_final.{json,pkl}` | [`checkpoints/q_table_final.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/checkpoints/q_table_final.json) | Verified |

---

## 2. Multi-Experiment Benchmark Comparison Summary

Evaluations run headlessly with $\epsilon = 0.0$ (pure greedy exploitation) vs. random baseline ($\epsilon = 1.0$):

| Mode | Reward Strategy | Level / Stage | Untrained Win% | Trained Win% | Untrained Avg R | Trained Avg R | Clear Steps | Items Collected |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **curriculum** | **balanced** | Level 1 | 50.0% | **100.0%** | +109.3 | **+821.0** | 170.0 | 0.0 |
| | | Level 2 | 0.0% | **100.0%** | -38.5 | **+1193.3** | 267.0 | 0.0 |
| | | *Level 3 (Zero-Shot)* | — | 0.0% | — | +683.3 | — | 1.0 |
| | | Level 3 | 0.0% | 0.0% | -53.0 | +711.7 | N/A | 1.0 |
| | | Level 4 | 0.0% | 0.0% | -26.1 | +459.3 | N/A | 1.0 |
| | | *Level 5 (Zero-Shot)* | — | 0.0% | — | +215.2 | — | 1.3 |
| | | Level 5 | 0.0% | 0.0% | -9.9 | +234.2 | N/A | 1.0 |
| **curriculum** | **sparse** | Level 1 | 50.0% | **100.0%** | +50.0 | +100.0 | 175.0 | 0.0 |
| | | Level 2 | 0.0% | 0.0% | -50.0 | +0.0 | N/A | 0.0 |
| | | *Level 3 (Zero-Shot)* | — | 0.0% | — | +0.0 | — | 0.0 |
| | | Level 3 | 0.0% | 0.0% | -45.0 | +0.0 | N/A | 0.0 |
| | | Level 4 | 0.0% | 0.0% | -40.0 | +0.0 | N/A | 0.0 |
| | | *Level 5 (Zero-Shot)* | — | 0.0% | — | +0.0 | — | 0.0 |
| | | Level 5 | 0.0% | 0.0% | -40.0 | +0.0 | N/A | 0.0 |
| **curriculum** | **dense** | Level 1 | 50.0% | **100.0%** | +637.0 | +921.2 | 170.0 | 0.0 |
| | | Level 2 | 0.0% | **100.0%** | +213.9 | +1376.2 | 267.0 | 0.0 |
| | | *Level 3 (Zero-Shot)* | — | 0.0% | — | +873.3 | — | 1.0 |
| | | Level 3 | 0.0% | 0.0% | +111.4 | +902.6 | N/A | 1.0 |
| | | Level 4 | 0.0% | 0.0% | +256.1 | +400.6 | N/A | 1.0 |
| | | *Level 5 (Zero-Shot)* | — | 0.0% | — | +285.6 | — | 0.0 |
| | | Level 5 | 0.0% | 0.0% | +254.8 | +502.7 | N/A | 1.7 |
| **separate** | **balanced** | Level 1 | 50.0% | **100.0%** | +109.3 | +821.0 | 170.0 | 0.0 |
| | | Level 2 | 0.0% | **100.0%** | -38.5 | +1177.3 | 281.7 | 0.0 |
| | | Level 3 | 0.0% | 0.0% | -53.0 | +122.4 | N/A | 0.0 |
| | | Level 4 | 0.0% | 0.0% | -26.1 | +473.1 | N/A | 1.7 |
| | | Level 5 | 0.0% | 0.0% | -9.9 | +264.3 | N/A | 0.0 |
| **separate** | **sparse** | Level 1 | 50.0% | **100.0%** | +50.0 | +100.0 | 175.0 | 0.0 |
| | | Level 2 | 0.0% | 0.0% | -50.0 | -50.0 | N/A | 0.0 |
| | | Level 3 | 0.0% | 0.0% | -45.0 | +0.0 | N/A | 0.0 |
| | | Level 4 | 0.0% | 0.0% | -40.0 | +10.0 | N/A | 1.0 |
| | | Level 5 | 0.0% | 0.0% | -40.0 | +0.0 | N/A | 0.0 |
| **separate** | **dense** | Level 1 | 50.0% | **100.0%** | +637.0 | +921.2 | 170.0 | 0.0 |
| | | Level 2 | 0.0% | **100.0%** | +213.9 | +1370.2 | 281.7 | 0.0 |
| | | Level 3 | 0.0% | 0.0% | +111.4 | +184.8 | N/A | 0.0 |
| | | Level 4 | 0.0% | 0.0% | +256.1 | +665.2 | N/A | 1.7 |
| | | Level 5 | 0.0% | 0.0% | +254.8 | +360.4 | N/A | 0.0 |

---

## 3. Order #5 Architectural Upgrades & Guarantees

1. **1,152-State Space & Ghost Blindness Removal**:
   - `MarioStateDiscretizer` expanded from 864 to 1,152 states with the addition of `OVERLAP` (`dist_to_ghost == 0.0`) and widening `pit_dist` lookahead to 100 px.
   - Deletion of `player_feet_y <= ghost_y` check prevents falling deaths from polluting safe sky states.
2. **High-Ghost Jump Discipline**:
   - `BalancedRewardStrategy` penalizes unnecessary jumping near high ghosts (`ghost_y < 260.0`), incentivizing Hoá to walk underneath.
3. **Linear Alpha Decay & Option B Replay**:
   - $\alpha$ decays linearly from $0.12$ to $0.02$ within each level stage.
   - Replay ratio $0.35$ refreshes previous levels' states during curriculum training in RAM without polluting the exact $2,000$ target-level episode count in `episode_history`.
