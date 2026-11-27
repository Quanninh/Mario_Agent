# L4–L5 Observability, Bug Fixes, Eligibility Traces & Stage-Scaled Training Report

**Scope:** Bug fixes, discretizer refinements (`110,592`-state 12-tuple), `balanced` reward sync + early double-jump penalty, optional replacing eligibility traces (`Q(λ)` / `Expected SARSA(λ)`), stage-scaled schedule with `2400`-step L5 cap, and local laptop validation batch (`--lambda 0.65`).

> Strict boundaries honored: `env/`, `core/base_env.py`, `DenseRewardStrategy`, and `SparseRewardStrategy` remained untouched.

---

## 1. Stage 1: Unit & Invariant Verification

### 1.1 Automated Test Suite (`65/65` Passing)
Verified via:
```bash
python -m unittest discover -s tests -p "test_*.py"
# Ran 65 tests in 0.677s — OK
```
Includes **14 new invariant checks** in `tests/test_l4_l5_upgrades.py`:
- **Discretizer (`discretizer/state_discretizer.py`)**:
  - Fire-warning-aware `pit_dist_bin`: `pit_fire_state == 1` at `50.0 px -> 3`, safe pit at `25.0 px -> 2`, far pit at `80.0 px` with embers `-> 1`.
  - Double-jump-safe `vy_bin`: airborne `-10.0 -> 1`, airborne apex `-1.5 -> 0`, airborne falling `4.0 -> -1`, grounded `-> 0`.
  - 3-bin `laser_bin`: `70.0 px -> 1`, `20.0 px -> 2`, `100.0 px -> 0`.
  - `discretizer.state_space_size == 110592` (`12`-tuple contract preserved).
- **Balanced Reward (`rewards/balanced_reward.py`)**:
  - Mid-air `RIGHT_JUMP` with `can_double_jump=True`, `is_crouching=False`, `velocity_y=-10.0` incurs `-3.0` (`early_double_jump_penalty`); apex jump at `velocity_y=-1.0` incurs `0.0`.
  - Grounded jumps at `dist_pit=55.0` (`<= 65.0`) or `dist_laser=75.0` (`<= 85.0`) are exempted; open ground grounded jump still incurs `-15.0`.
  - `DenseRewardStrategy` and `SparseRewardStrategy` verified untouched.
- **Agents (`agent/tabular_q_agent.py`, `agent/hybrid_q_sarsa_agent.py`)**:
  - Greedy `select_action()` (`training=False`) uses read-only `q_table.get(state)` and never inflates `len(q_table)` on unseen states.
  - Removed overwritten `update_hybrid = update` alias; `HybridQSARSAAgent.update_hybrid(s, a, r, s2, a2, done)` works positionally without swapping `a2` and `done`.
  - Replacing traces (`trace_decay=0.65`) back-propagate terminal rewards along multi-step trajectories in 1 episode, Watkins's `Q(λ)` cuts prior traces on strictly suboptimal exploratory actions, `reset_traces()` clears traces between episodes, and `"trace_decay"` serializes/loads backward-compatibly.
- **Trainer & Playback (`training/trainer.py`, `train.py`, `play_agent.py`)**:
  - Explicit 8-class outcome classification (`win`, `ghost_death`, `fire_death`, `laser_death`, `wisp_death`, `monster_death`, `pit_death`, `timeout`) and `outcome_counts` aggregation in `evaluate_agent()`.
  - `max_steps_cap=2400` terminates stalling Level 5 episodes at step `2400` with `outcome == "timeout"` and applies `timeout_penalty = -50.0`.
  - Curriculum replay covers all prior levels `levels[:lvl_idx]` (`lvl_idx > 0`) with cached `replay_envs` closed cleanly in `finally`, `rewarm_epsilon_ghost` starts at `lvl >= 3`, and `post_curriculum_retention` / `best_checkpoint_retention` populate cleanly.
  - `play_agent.py` uses `env.is_running` and clears `explicit_checkpoint = None` on `F1–F3`, `T`, `H/A`, and `separate`-mode level swaps (`1..8`).

### 1.2 100-Episode Smoke Test
Executed end-to-end across all 4 `balanced` runs (`--episodes 100 --lambda 0.65 --max-steps-cap 2400`) confirming `state_space_max = 110592`, `max_steps_cap = 2400`, `outcome_counts`, `post_curriculum_retention`, `best_checkpoint_retention`, and clean checkpoint exports (including `hybrid_expected` reaching `100.0%` L2 and `2.0%` L3 even within 100 episodes).

---

## 2. Stage 2: Level 4 A/B Check ($\lambda = 0.0$ vs. $\lambda = 0.65$)

Executed on **Level 4 only** in `separate` mode (`--levels 4 --episodes 3500 --reward balanced --max-steps-cap 2400`, seed `42`):

| Agent | $\lambda$ | Untrained Win% / Avg R | Final Eval Win% / Avg R | Best Checkpoint Win% / Avg R | Best CP Clear Steps / Items | L4 Eval `outcome_counts` | Train Wins / Max Roll50 / States |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`tabular`** | `0.0` | `0.0%` / `+111.26` | `0.0%` / `+627.90` | `0.0%` / `+1164.90` | N/A / `4.0` | Final: `ghost_death: 50`<br>Best: `laser_death: 50` | `0/3500` / `0.0%` / `288` (`max_x=1355.1`, last500 avgR `710.5`, prog `39.3%`) |
| **`tabular`** | `0.65` | `0.0%` / `+111.26` | `0.0%` / `+784.90` | `0.0%` / `+809.10` | N/A / `1.2` | Final: `pit_death: 50`<br>Best: `fire_death: 40, ghost_death: 8, laser_death: 2` | `0/3500` / `0.0%` / `265` (`max_x=1254.0`, last500 avgR `673.3`, prog `39.3%`) |
| **`hybrid_expected`** | `0.0` | `0.0%` / `+111.26` | `0.0%` / `+264.80` | `0.0%` / `+664.43` | N/A / `1.0` | Final: `ghost_death: 50`<br>Best: `fire_death: 50` | `0/3500` / `0.0%` / `216` (`max_x=1076.0`, last500 avgR `277.3`, prog `19.0%`) |
| **`hybrid_expected`** | **`0.65`** | `0.0%` / `+111.26` | `0.0%` / **`+1600.97`** | **`26.0%` (`13/50`)** / **`+1722.06`** | **`671.0`** / **`4.0`** | Final: `ghost_death: 50`<br>Best: `win: 13, ghost_death: 37` | **`12/3500`** / **`10.0%`** / `361` (`max_x=2032.0`, last500 avgR `1103.7`, prog `62.3%`) |

- **Selected Winning Value: `--lambda 0.65`** (`Expected SARSA(0.65)` improved Level 4 Best Checkpoint win rate from `0.0%` to `26.0%`, increased Final Eval reward from `+264.80` to `+1600.97`, lifted late-stage training progress from `19.0%` to `62.3%`, and improved `tabular` Final Eval reward from `+627.90` to `+784.90`).

---

## 3. Stage 3: Full 4-Run `balanced` Benchmark (L1–L5, $\lambda = 0.65$)

Executed with stage-scaled schedule `{1: 800, 2: 1200, 3: 1800, 4: 3500, 5: 4500}` (`11,800` episodes per run), auto-computed per-stage `epsilon_decay`, `--lambda 0.65`, and `--max-steps-cap 2400`:

### Output Boards Generated
- `outputs/training_board_tabular_curriculum_balanced.json`
- `outputs/training_board_tabular_separate_balanced.json`
- `outputs/training_board_hybrid_expected_curriculum_balanced.json`
- `outputs/training_board_hybrid_expected_separate_balanced.json`

### 3.1 Per-Stage Completion Summary (50 Greedy Eval Episodes Immediately After Each Level Stage)

| Agent | Mode | Level | Untrained Win% / Avg R | Stage Eval Win% | Stage Avg R | Clear Steps | Items | Train Wins / Max Progress | States Discovered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **`tabular`** | `curriculum` | L1 | `6.67%` / `+190.8` | **`100.0%`** | `+927.90` | `231.0` | `0.0` | `786/800` (`100.0%`, `x=882.8`) | `27` |
| `tabular` | `curriculum` | L2 | `0.0%` / `+96.4` | **`100.0%`** | `+1157.20` | `328.0` | `0.0` | `553/1200` (`100.0%`, `x=1082.8`) | `65` |
| `tabular` | `curriculum` | L3 | `0.0%` / `+126.4` | **`100.0%`** | `+1375.90` | `359.0` | `3.0` | `329/1800` (`100.0%`, `x=1202.4`) | `177` |
| `tabular` | `curriculum` | L4 | `0.0%` / `+111.3` | `0.0%` | `+973.30` | N/A | `3.0` | `0/3500` (`90.28%`, `x=1834.0`) | `356` |
| `tabular` | `curriculum` | L5 | `0.0%` / `+40.2` | `0.0%` | `+225.46` | N/A | `1.0` | `0/4500` (`31.50%`, `x=801.6`) | `873` |
| **`tabular`** | `separate` | L1 | `6.67%` / `+190.8` | **`100.0%`** | `+927.90` | `231.0` | `0.0` | `786/800` (`100.0%`, `x=882.8`) | `27` |
| `tabular` | `separate` | L2 | `0.0%` / `+96.4` | **`100.0%`** | `+1052.90` | `291.0` | `0.0` | `539/1200` (`100.0%`, `x=1078.8`) | `61` |
| `tabular` | `separate` | L3 | `0.0%` / `+126.4` | `0.0%` *(Best CP: **100%**)* | `+598.20` | N/A *(`406.0`)* | `2.0` *(`3.0`)* | `35/1800` (`100.0%`, `x=1202.8`) | `165` |
| `tabular` | `separate` | L4 | `0.0%` / `+111.3` | `0.0%` | `+784.90` | N/A | `0.0` | `0/3500` (`60.93%`, `x=1254.0`) | `265` |
| `tabular` | `separate` | L5 | `0.0%` / `+40.2` | `0.0%` | `+307.80` | N/A | `1.0` | `0/4500` (`26.66%`, `x=686.0`) | `665` |
| **`hybrid_expected`** | `curriculum` | L1 | `6.67%` / `+190.8` | **`100.0%`** | `+890.30` | `207.0` | `0.0` | `788/800` (`100.0%`, `x=879.7`) | `25` |
| `hybrid_expected` | `curriculum` | L2 | `0.0%` / `+96.4` | **`100.0%`** | `+1166.00` | `300.0` | `0.0` | `611/1200` (`100.0%`, `x=1079.6`) | `62` |
| `hybrid_expected` | `curriculum` | L3 | `0.0%` / `+126.4` | `0.0%` *(Post: **100%**)* | `+881.30` | N/A *(`390.0`)* | `2.0` | `204/1800` (`100.0%`, `x=1202.0`) | `170` |
| `hybrid_expected` | `curriculum` | L4 | `0.0%` / `+111.3` | `0.0%` | `+310.90` | N/A | `1.0` | `0/3500` (`61.44%`, `x=1264.0`) | `281` |
| `hybrid_expected` | `curriculum` | L5 | `0.0%` / `+40.2` | `0.0%` | `+294.90` | N/A | `1.0` | `0/4500` (`31.70%`, `x=806.4`) | `678` |
| **`hybrid_expected`** | `separate` | L1 | `6.67%` / `+190.8` | **`100.0%`** | `+890.30` | `207.0` | `0.0` | `788/800` (`100.0%`, `x=879.7`) | `25` |
| `hybrid_expected` | `separate` | L2 | `0.0%` / `+96.4` | `0.0%` | `+551.40` | N/A | `0.0` | `187/1200` (`100.0%`, `x=1079.8`) | `64` |
| `hybrid_expected` | `separate` | L3 | `0.0%` / `+126.4` | `0.0%` *(Best CP: **100%**)* | `+673.50` | N/A *(`360.0`)* | `2.0` | `105/1800` (`100.0%`, `x=1202.8`) | `168` |
| `hybrid_expected` | `separate` | L4 | `0.0%` / `+111.3` | `0.0%` *(Best CP: **14.0%**)* | `+1600.97` | N/A *(`671.0`)* | `4.0` | **`12/3500`** (`100.0%`, `x=2032.0`) | `361` |
| `hybrid_expected` | `separate` | L5 | `0.0%` / `+40.2` | `0.0%` | `+300.80` | N/A | `1.0` | `0/4500` (`21.54%`, `x=564.0`) | `639` |

**Zero-Shot Curriculum Transfers:**
- `tabular curriculum`: `level_3_zero_shot`: `0.0%` (`+531.87 R`, `ghost_death: 47, fire_death: 1, pit_death: 2`), `level_5_zero_shot`: `0.0%` (`+258.89 R`, `pit_death: 26, laser_death: 13, ghost_death: 11`)
- `hybrid_expected curriculum`: `level_3_zero_shot`: `0.0%` (`+543.47 R`, `ghost_death: 43, fire_death: 6, pit_death: 1`), `level_5_zero_shot`: `0.0%` (`+225.45 R`, `ghost_death: 20, laser_death: 15, pit_death: 15`)

### 3.2 Curriculum Retention Matrices (`post_curriculum_retention` & `best_checkpoint_retention`, 50 Greedy Eps/Level)

| Agent | Retention Matrix | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`tabular`** | `post_curriculum_retention` | `0.0%` (`+69.86 R`) | `0.0%` (`-49.08 R`) | `0.0%` (`-49.08 R`) | `0.0%` (`-79.20 R`) | `0.0%` (`+225.46 R`) |
| **`tabular`** | `best_checkpoint_retention` | **`100.0%`** (`+1007.60 R`, `214.0 st`) | **`100.0%`** (`+1194.60 R`, `312.0 st`) | `0.0%` (`+496.30 R`) | `0.0%` (`+353.20 R`) | `0.0%` (`+247.90 R`) |
| **`hybrid_expected`** | `post_curriculum_retention` | **`100.0%`** (`+904.40 R`, `216.0 st`) | **`100.0%`** (`+1145.70 R`, `293.0 st`) | **`100.0%`** (`+1295.00 R`, `390.0 st`, `2.0 it`) | `0.0%` (`+258.10 R`) | `0.0%` (`+294.90 R`) |
| **`hybrid_expected`** | `best_checkpoint_retention` | **`100.0%`** (`+999.10 R`, `213.0 st`) | **`100.0%`** (`+1184.10 R`, `299.0 st`) | **`100.0%`** (`+1353.70 R`, `403.0 st`, `2.0 it`) | `0.0%` (`+305.10 R`) | `0.0%` (`+281.20 R`) |

### 3.3 Exact Level 4 & Level 5 `outcome_counts` Breakdown (50 Evaluated Episodes Each)

| Run | Metric Source | Level | `win` | `ghost_death` | `fire_death` | `laser_death` | `wisp_death` | `monster_death` | `pit_death` | `timeout` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tabular curriculum` | Stage L4 Eval | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| `tabular curriculum` | `post_curriculum_retention` | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 |
| `tabular curriculum` | `best_checkpoint_retention` | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `tabular curriculum` | `q_table_lvl4_best` *(diag)* | 4 | 0 | 45 | 0 | 5 | 0 | 0 | 0 | 0 |
| `tabular curriculum` | Stage L5 / `post_curriculum_retention` | 5 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | 0 |
| `tabular curriculum` | `best_checkpoint_retention` / `q_table_lvl5_best` | 5 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | 0 |
| `tabular separate` | Stage L4 Eval | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| `tabular separate` | `q_table_lvl4_best` *(diag)* | 4 | 0 | 2 | 45 | 3 | 0 | 0 | 0 | 0 |
| `tabular separate` | Stage L5 Eval | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| `tabular separate` | `q_table_lvl5_best` *(diag)* | 5 | 0 | 34 | 0 | 1 | 15 | 0 | 0 | 0 |
| `hybrid_expected curriculum` | Stage L4 Eval | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected curriculum` | `post_curriculum_retention` | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected curriculum` | `best_checkpoint_retention` | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected curriculum` | `q_table_lvl4_best` *(diag)* | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| `hybrid_expected curriculum` | Stage L5 / `post_curriculum_retention` | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 |
| `hybrid_expected curriculum` | `best_checkpoint_retention` / `q_table_lvl5_best` | 5 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | 0 |
| `hybrid_expected separate` | Stage L4 Eval | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected separate` | `q_table_lvl4_best` *(diag)* | 4 | 7 | 43 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected separate` | Stage L5 Eval | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 |
| `hybrid_expected separate` | `q_table_lvl5_best` *(diag)* | 5 | 0 | 31 | 0 | 19 | 0 | 0 | 0 | 0 |

### 3.4 Shared Schedule & Decay Metadata Across All 4 Runs
- `state_space_max`: `110592`
- `trace_decay`: `0.65`
- `max_steps_cap`: `2400`
- `episodes_schedule`: `{"level_1": 800, "level_2": 1200, "level_3": 1800, "level_4": 3500, "level_5": 4500}`
- `epsilon_decay_per_level`:
  - `curriculum`: `{"level_1": 0.994173, "level_2": 0.997445, "level_3": 0.998182, "level_4": 0.999065, "level_5": 0.999272}`
  - `separate`: `{"level_1": 0.994173, "level_2": 0.996111, "level_3": 0.997406, "level_4": 0.998665, "level_5": 0.998962}`

---

## 4. Synchronized Specifications & Documentation
- `specs/PRD.md`
- `specs/ARCHITECTURE.md`
- `specs/TASKS.md`
- `specs/PROGRESS.md`
- `docs/MEMBER_3_HANDOFF_REPORT.md`
- `docs/TRAINING_AND_DEMO_GUIDE.md`

---

## 5. Reproduction Commands
```bash
# Run Stage 1 unit & invariant tests
python -m unittest discover -s tests -p "test_*.py" -v

# Re-run winning 4-run balanced benchmark
python train.py --agent all --mode all --reward balanced --lambda 0.65 --max-steps-cap 2400 --no-progress-bar
```
