# Parallel RL Training & Benchmarking Plan
## Hoá's Midnight Escape — Tabular Q-Learning & Hybrid Q-SARSA

This execution plan guides an AI agent or automated script to perform distributed, parallel reinforcement learning training and evaluation across a **32-core CPU**.

---

## 1. Hardware & Process Budgeting

* **Available CPU**: 32 logical cores
* **Parallel Worker Pool (`-j / --workers`)**: **30 workers**
* **Core Cushion**: 2 cores reserved for Windows OS background processes, thread scheduling, and non-blocking disk I/O (JSON/pickle serialization).
* **Multi-Job Wave Math**:
  $$\text{Jobs} = 2 \text{ Agents} \times 2 \text{ Modes} \times 3 \text{ Rewards} \times 10 \text{ Seeds} = 120 \text{ Jobs}$$
  $$\text{Waves} = \frac{120 \text{ Jobs}}{30 \text{ Workers}} = 4.0 \text{ exact waves (zero tail latency or idle cores)}$$

---

## 2. Locked Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `--episodes` | `15000` | 15,000 training episodes per level stage |
| `--alpha` | `0.10` | Initial Q-learning / SARSA learning rate |
| `--alpha-min` | `0.005` | Minimum decayed learning rate floor |
| `--epsilon-decay` | `0.9995` | Multiplicative decay per episode: reaches $\epsilon_{\min}=0.01$ at ep ~9,210 |
| `--epsilon-start` | `1.0` | Initial exploration rate (100% random) |
| `--epsilon-min` | `0.01` | Exploration floor (1% exploration during final ~5,800 episodes) |
| `--gamma` | `0.99` | Discount factor |
| `--seed` | `42` | Base random seed |
| `--num-seeds` | `10` | 10 independent seeds generated: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]` |
| `--workers` | `30` | 30 parallel worker processes (`multiprocessing.Pool`) |
| `--no-progress-bar`| Active | Plain newline logging to prevent carriage return (`\r`) buffer scrambling |

---

## 3. Experiment Grid Dimensions

| Dimension | Choices | Values |
| :--- | :---: | :--- |
| **Agent (`--agent`)** | 2 | `tabular` (Tabular Q-Learning), `hybrid` (Hybrid Q-SARSA, `--hybrid-mode expected`) |
| **Mode (`--mode`)** | 2 | `curriculum` (stage transfer & replay), `separate` (fresh agent per level) |
| **Reward (`--reward`)** | 3 | `balanced` (canonical), `sparse` (terminal only), `dense` (potential shaping) |
| **Seeds (`--seeds`)** | 10 | Seeds 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 |

---

## 4. Execution Tracks & CLI Commands

### Track 0: Pre-Flight Verification (Smoke Test)
**Objective**: Verify Python interpreter, multiprocessing pool spawning, and file write permissions in < 30 seconds before launching large runs.
```powershell
python train.py --smoke-test -j 8 --agent all --mode all --reward all --num-seeds 2
```

---

### Track A: Primary Curriculum Stages (Levels 1 to 5)
* **Target Levels**: `--levels 1 2 3 4 5`
* **Focus**: Foundational mechanics — Quiet Graveyard, Pits, Whispering Ghosts, Nightmare Catacombs, and Valley of the Wisp Maidens.
* **Volume**: 120 jobs (75,000 episodes/job).
* **Estimated Runtime**: **~14 to 18 minutes** across 30 workers.

**Execution Command**:
```powershell
python train.py `
  --agent all `
  --mode all `
  --reward all `
  --levels 1 2 3 4 5 `
  --episodes 15000 `
  --alpha 0.10 `
  --alpha-min 0.005 `
  --epsilon-decay 0.9995 `
  --seed 42 `
  --num-seeds 10 `
  --workers 30 `
  --no-progress-bar
```

---

### Track B: Advanced Ambush & Master Gauntlet (Levels 6, 7, 8)
* **Target Levels**: `--levels 6 7 8`
* **Focus**: Complex hazard navigation — Awakening of Corpse Demon (Quỷ Nhập Tràng ambushes), Cataclysm of the Undead (Jiangshi / Cương Thi bounds), and The Celestial Purgatory (7,200px gauntlet with hanging slide tunnels and aerial catwalks).
* **Volume**: 120 jobs (45,000 episodes/job).
* **Estimated Runtime**: **~18 to 25 minutes** across 30 workers.

**Execution Command**:
```powershell
python train.py `
  --agent all `
  --mode all `
  --reward all `
  --levels 6 7 8 `
  --episodes 15000 `
  --alpha 0.10 `
  --alpha-min 0.005 `
  --epsilon-decay 0.9995 `
  --seed 42 `
  --num-seeds 10 `
  --workers 30 `
  --no-progress-bar
```

---

### Track C: Complete Grand Slam (Levels 1 through 8)
* **Target Levels**: `--levels 1 2 3 4 5 6 7 8`
* **Focus**: Full 8-stage end-to-end curriculum transfer evaluation.
* **Volume**: 120 jobs (120,000 episodes/job).
* **Estimated Runtime**: **~30 to 45 minutes** across 30 workers.

**Execution Command**:
```powershell
python train.py `
  --agent all `
  --mode all `
  --reward all `
  --levels 1 2 3 4 5 6 7 8 `
  --episodes 15000 `
  --alpha 0.10 `
  --alpha-min 0.005 `
  --epsilon-decay 0.9995 `
  --seed 42 `
  --num-seeds 10 `
  --workers 30 `
  --no-progress-bar
```

---

## 5. Automated PowerShell Script (`run_all_benchmarks.ps1`)

To execute Track A followed by Track B with automatic error handling, console output logging, and execution timing, save and run:

```powershell
# run_all_benchmarks.ps1
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " STARTING PARALLEL RL BENCHMARK SUITE (32-CORE CPU)       " -ForegroundColor Cyan
Write-Host " Workers: 30 | Episodes: 15,000 | Seeds: 10 (42..51)     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TotalStartTime = Get-Date

# 1. Run Track A (Levels 1 to 5)
Write-Host "`n>>> [1/2] Launching Track A: Levels 1, 2, 3, 4, 5 (120 Jobs)..." -ForegroundColor Yellow
$TrackAStart = Get-Date
python train.py `
    --agent all `
    --mode all `
    --reward all `
    --levels 1 2 3 4 5 `
    --episodes 15000 `
    --alpha 0.10 `
    --alpha-min 0.005 `
    --epsilon-decay 0.9995 `
    --seed 42 `
    --num-seeds 10 `
    --workers 30 `
    --no-progress-bar *>&1 | Tee-Object -FilePath "outputs\benchmark_trackA_levels1-5.log"
$TrackADuration = (Get-Date) - $TrackAStart
Write-Host ">>> Track A Finished in: $($TrackADuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green

# 2. Run Track B (Levels 6, 7, 8)
Write-Host "`n>>> [2/2] Launching Track B: Levels 6, 7, 8 (120 Jobs)..." -ForegroundColor Yellow
$TrackBStart = Get-Date
python train.py `
    --agent all `
    --mode all `
    --reward all `
    --levels 6 7 8 `
    --episodes 15000 `
    --alpha 0.10 `
    --alpha-min 0.005 `
    --epsilon-decay 0.9995 `
    --seed 42 `
    --num-seeds 10 `
    --workers 30 `
    --no-progress-bar *>&1 | Tee-Object -FilePath "outputs\benchmark_trackB_levels6-8.log"
$TrackBDuration = (Get-Date) - $TrackBStart
Write-Host ">>> Track B Finished in: $($TrackBDuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green

$TotalDuration = (Get-Date) - $TotalStartTime
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " ALL BENCHMARK TRACKS COMPLETED SUCCESSFULLY!             " -ForegroundColor Green
Write-Host " Total Elapsed Time: $($TotalDuration.TotalMinutes.ToString('F2')) minutes." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
```

---

## 6. Output Files & Artifact Structure

Every job produces independent artifacts, completely avoiding file contention:

1. **Per-Run Checkpoints (`checkpoints/`)**:
   - `checkpoints/<agent_tag>/<mode>_<reward>_seed<seed>/q_table_best.pkl`
   - `checkpoints/<agent_tag>/<mode>_<reward>_seed<seed>/q_table_best.json`
   - `checkpoints/<agent_tag>/<mode>_<reward>_seed<seed>/q_table_final.pkl`
2. **Per-Run CSV Learning Metrics (`outputs/`)**:
   - `outputs/training_metrics_<agent_tag>_<mode>_<reward>_seed<seed>.csv`
   - Columns: `global_episode`, `level_episode`, `level_id`, `total_reward`, `steps`, `max_x_reached`, `progress_pct`, `obstacle_hits`, `items_collected`, `epsilon`, `won`, `outcome`, `rolling_win_rate_100`.
3. **Per-Run Evaluation JSON Data Boards (`outputs/`)**:
   - `outputs/training_board_<agent_tag>_<mode>_<reward>_seed<seed>.json`
   - Full evaluation summary across all tested levels (win rate, clear steps, items collected, untrained vs. trained).
4. **Multi-Seed Statistical Summary Table**:
   - Printed at completion of each track by `train.py`. Computes:
     - `Trained Win% (Mean ± Std)`
     - `Trained Reward (Mean ± Std)`
     - `Clear Steps (Mean)`

---

## 7. Instructions for Subagents / Executing Agents

When executing this plan:
1. Always start by verifying with **Track 0 (Smoke Test)**. Ensure exit code is 0.
2. Run **Track A** first (`--levels 1 2 3 4 5`). Verify outputs are populated in `outputs/` and `checkpoints/`.
3. Run **Track B** next (`--levels 6 7 8`).
4. Inspect the summary comparison tables in the generated log files:
   - `outputs/benchmark_trackA_levels1-5.log`
   - `outputs/benchmark_trackB_levels6-8.log`
5. Report the statistical delta between `tabular` vs `hybrid` across `balanced`, `sparse`, and `dense` rewards.
