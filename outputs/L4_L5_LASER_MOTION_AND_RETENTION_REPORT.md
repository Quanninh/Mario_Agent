# L4–L5 Pit-Margin (24 px), Motion-Aware Laser Bin, Non-Zero Trace Floor (0.65 -> 0.35), Alpha Cooling & Curriculum Retention Guard Report

**Scope:** Re-calibrated `pit_immediate = 24.0` px (`6`-tick takeoff horizon), world-frame motion-aware 3-bin `laser_bin` (`laser_imminent_distance = 20.0` px + `discretizer.reset()`), non-zero trace floor (`λ: 0.65 -> 0.35`), sharper `alpha_min = 0.005`, linear-weighted curriculum replay, `q_table_curriculum_best.{pkl,json}` retention guard, read-only verification of the `env/game_env.py` `±85 px` vertical ghost-FOV filter, and 4-run local validation batch (`14,800` eps/run).

> Strict boundaries honored: `env/`, `core/base_env.py`, `DenseRewardStrategy`, `SparseRewardStrategy`, `BalancedRewardStrategy` core weights, and the 12-tuple / `55,296`-state capacity remained untouched.

---

## 1. Stage 1: Unit & Invariant Verification

### 1.1 Automated Test Suite (`74/74` Passing)
Verified via:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
# Ran 74 tests in 0.878s — OK
```
- Updated `pit_immediate` (`24.0` px) and motion-aware `laser_bin` tests in `tests/test_l4_l5_upgrades.py` and `tests/test_pit_margin_and_trace_decay.py`.
- Added **5 new invariant & regression tests** in `tests/test_laser_motion_and_retention.py`:
  1. `test_level5_ground_ghost_not_occluded_by_sky_maiden`: read-only check on `create_env(5)` at `player_x in (270, 285, 300), player_y = 288` confirming `raw.ghost_y > 200.0` (user's `±85 px` vertical FOV patch prevents Sky Maiden 1 from occluding Base Ghost 1).
  2. `test_pit_margin_24_and_motion_aware_laser_bin`: verifies `20.0 px -> 2`, `30.0 px -> 1`, `50.0 px + fire=1 -> 3`, world-frame 4-tick laser sequence (`1 -> 2` receding -> `1` closing -> `1` imminent `<= 20 px`), `reset()` clearing `_prev_laser_world_x`, and `55296 / 110592` state cardinality.
  3. `test_non_zero_trace_floor_alpha_cooling_and_zero_fast_path`: verifies `λ: 0.65 -> 0.35` (reaching `0.35` across eps `76..100`), `alpha: 0.12 -> 0.005`, zero-`λ` fast path staying `0.0`, and metadata fields (`trace_decay_start`, `trace_decay_min`, `trace_decay_schedule`, `alpha_min`).
  4. `test_curriculum_retention_guard_locks_peak_stage_against_later_collapse`: verifies `q_table_curriculum_best.pkl` and `curriculum_best_checkpoint_stage` stay locked at the peak multi-level stage (`2`) when Stage 3 collapses.
  5. `test_trainer_resets_discretizer_between_episodes`: verifies `Trainer._run_episode()` resets `_prev_laser_world_x` before every episode.

### 1.2 100-Episode Balanced Smoke Test
Executed across all 4 `balanced` runs (`--episodes 100 --lambda 0.65 --lambda-min 0.35 --alpha-min 0.005 --eval-interval 25 --rewarm-epsilon-hard 0.50 --max-steps-cap 2400`) into temporary dirs (verified and cleaned afterward), confirming valid metadata (`state_space_max = 55296`, `trace_decay_start = 0.65`, `trace_decay_min = 0.35`, `trace_decay_schedule = "linear_to_0.35_at_75pct"`, `alpha_min = 0.005`) and `curriculum_best_retention`.

---

## 2. Stage 2: Full 4-Run `balanced` Benchmark (L1–L5, `λ: 0.65 -> 0.35`, `alpha_min = 0.005`)

Executed concurrently (`seed=42`, `{1: 800, 2: 1200, 3: 1800, 4: 5000, 5: 6000}` = `14,800` episodes/run, `--lambda 0.65`, `--lambda-min 0.35`, `--alpha-min 0.005`, `--eval-interval 25`, `--rewarm-epsilon-hard 0.50`, `--max-steps-cap 2400`, `enable_wisp_sensor = False`):

### Output Files Generated
- `outputs/training_board_{tabular,hybrid_expected}_{curriculum,separate}_balanced.json` (4 JSON boards)
- `outputs/training_metrics_{tabular,hybrid_expected}_{curriculum,separate}_balanced.csv` (4 CSV logs)

### 2.1 Per-Stage Final & Best Checkpoint Summary (`50` Greedy Eval Episodes Each)

| Agent | Mode | Lvl | Untr Win% / Avg R | Final Win% / Avg R (Steps, Items) | Best CP Win% / Avg R (Steps, Items) | Train Wins / Max R50 / Max Prog | States |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`tabular`** | `curriculum` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+949.60` (`200.0`, `0.0`) | **`100.0%`** / `+955.50` (`233.0`, `0.0`) | `787/800` / `100.0%` / `100.0%` | `27` |
| `tabular` | `curriculum` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1166.40` (`246.0`, `0.0`) | **`100.0%`** / `+1183.90` (`291.0`, `0.0`) | `616/1200` / `96.0%` / `100.0%` | `63` |
| `tabular` | `curriculum` | L3 | `0.0%` / `+126.42` | `0.0%` / `+691.20` (`N/A`, `2.0`) | **`100.0%`** / `+1295.40` (`402.0`, `0.0`) | `144/1800` / `50.0%` / `100.0%` | `164` |
| `tabular` | `curriculum` | L4 | `0.0%` / `+111.26` | `0.0%` / `+1116.40` (`N/A`, `2.0`) | **`26.0%`** / **`+1575.86`** (`643.3`, `3.48`) | **`9/5000`** / `6.0%` / **`100.0%`** | `388` |
| `tabular` | `curriculum` | L5 | `0.0%` / `+33.67` | `0.0%` / `+200.80` (`N/A`, `1.0`) | `0.0%` / `+307.14` (`N/A`, `1.0`) | `0/6000` / `0.0%` / `35.44%` | `624` |
| **`tabular`** | `separate` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+949.60` (`200.0`, `0.0`) | **`100.0%`** / `+955.50` (`233.0`, `0.0`) | `787/800` / `100.0%` / `100.0%` | `27` |
| `tabular` | `separate` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1072.40` (`276.0`, `0.0`) | **`100.0%`** / `+1138.70` (`333.0`, `0.0`) | `574/1200` / `98.0%` / `100.0%` | `57` |
| `tabular` | `separate` | L3 | `0.0%` / `+126.42` | **`100.0%`** / **`+1287.00`** (`380.0`, `1.0`) | **`100.0%`** / **`+1287.00`** (`380.0`, `1.0`) | `398/1800` / `92.0%` / `100.0%` | `155` |
| `tabular` | `separate` | L4 | `0.0%` / `+111.26` | `0.0%` / `+915.90` (`N/A`, `1.0`) | `0.0%` / `+1154.11` (`N/A`, `2.0`) | `0/5000` / `0.0%` / `64.27%` | `298` |
| `tabular` | `separate` | L5 | `0.0%` / `+33.67` | `0.0%` / `+302.00` (`N/A`, `1.0`) | `0.0%` / `+346.47` (`N/A`, `1.0`) | `0/6000` / `0.0%` / `33.78%` | `432` |
| **`hybrid_expected`** | `curriculum` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+907.80` (`212.0`, `0.0`) | **`100.0%`** / `+946.80` (`212.0`, `0.0`) | `788/800` / `100.0%` / `100.0%` | `26` |
| `hybrid_expected` | `curriculum` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1099.10` (`339.0`, `0.0`) | **`100.0%`** / `+1113.40` (`346.0`, `0.0`) | `616/1200` / `100.0%` / `100.0%` | `58` |
| `hybrid_expected` | `curriculum` | L3 | `0.0%` / `+126.42` | `0.0%` / `+658.70` (`N/A`, `2.0`) | **`100.0%`** / `+1320.98` (`373.0`, `2.0`) | `279/1800` / `34.0%` / `100.0%` | `167` |
| `hybrid_expected` | `curriculum` | L4 | `0.0%` / `+111.26` | `0.0%` / `+359.60` (`N/A`, `2.0`) | `0.0%` / `+894.10` (`N/A`, `3.0`) | `0/5000` / `0.0%` / `82.13%` | `289` |
| `hybrid_expected` | `curriculum` | L5 | `0.0%` / `+33.67` | `0.0%` / `+288.20` (`N/A`, `0.0`) | `0.0%` / `+170.50` (`N/A`, `0.0`) | `0/6000` / `0.0%` / `34.18%` | `565` |
| **`hybrid_expected`** | `separate` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+907.80` (`212.0`, `0.0`) | **`100.0%`** / `+946.80` (`212.0`, `0.0`) | `788/800` / `100.0%` / `100.0%` | `26` |
| `hybrid_expected` | `separate` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1109.00` (`300.0`, `0.0`) | **`100.0%`** / `+1109.00` (`300.0`, `0.0`) | `283/1200` / `74.0%` / `100.0%` | `63` |
| `hybrid_expected` | `separate` | L3 | `0.0%` / `+126.42` | `0.0%` / `+671.60` (`N/A`, `2.0`) | `0.0%` / `+870.10` (`N/A`, `2.0`) | `109/1800` / `30.0%` / `100.0%` | `147` |
| `hybrid_expected` | `separate` | L4 | `0.0%` / `+111.26` | `0.0%` / `+752.50` (`N/A`, `1.0`) | `0.0%` / `+901.30` (`N/A`, `1.8`) | `0/5000` / `0.0%` / `60.10%` | `215` |
| `hybrid_expected` | `separate` | L5 | `0.0%` / `+33.67` | `0.0%` / `+295.70` (`N/A`, `1.0`) | `0.0%` / `+315.31` (`N/A`, `1.22`) | `0/6000` / `0.0%` / `31.89%` | `432` |

**Zero-Shot Curriculum Transfers (`50` eps):**
- `tabular curriculum`: `level_3_zero_shot`: **`2.0%`** (`1/50` win, `+552.08 R`, `1.0` it, `ghost: 46, fire: 2, pit: 1`); `level_5_zero_shot`: `0.0%` (`+225.39 R`, `1.0` it, `ghost: 35, laser: 13, pit: 2`).
- `hybrid_expected curriculum`: `level_3_zero_shot`: `0.0%` (`+477.66 R`, `1.02` it, `ghost: 47, fire: 2, pit: 1`); `level_5_zero_shot`: `0.0%` (`+174.92 R`, `0.96` it, `ghost: 20, laser: 13, pit: 13, wisp: 4`).

### 2.2 Curriculum Retention Matrices (`50` Greedy Eval Episodes Per Level)

| Agent | Retention Matrix | Stage | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tabular`** | `post_curriculum_retention` | Final (L5) | **`100.0%`** (`+1001.1 R`, `253.0 st`) | `0.0%` (`+105.2 R`, `timeout: 50`) | `0.0%` (`+105.2 R`, `timeout: 50`) | `0.0%` (`+210.7 R`, `ghost: 50`) | `0.0%` (`+200.8 R`, `ghost: 50`) |
| **`tabular`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+1000.3 R`, `201.0 st`) | **`100.0%`** (`+1189.9 R`, `279.0 st`) | **`100.0%`** (`+1359.4 R`, `352.0 st`) | `0.0%` (`+298.4 R`, `fire: 50`) | `0.0%` (`+307.14 R`, `pit: 32, ghost: 18`) |
| **`tabular`** | **`curriculum_best_retention`** | **Stage 3** | **`100.0%`** (`+1008.1 R`, `182.0 st`) | **`100.0%`** (`+1192.0 R`, `304.0 st`) | **`100.0%`** (`+1295.4 R`, `402.0 st`) | `0.0%` (`+274.3 R`, `fire: 38, ghost: 9, pit: 3`) | `0.0%` (`+205.7 R`, `laser: 23, wisp: 13, ghost: 8, pit: 5, fire: 1`) |
| **`hybrid_expected`** | `post_curriculum_retention` | Final (L5) | **`100.0%`** (`+1003.5 R`, `229.0 st`) | `0.0%` (`+578.9 R`, `pit: 50`) | `0.0%` (`+520.6 R`, `ghost: 50`) | `0.0%` (`+251.2 R`, `ghost: 50`) | `0.0%` (`+288.2 R`, `pit: 50`) |
| **`hybrid_expected`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+1009.5 R`, `177.0 st`) | **`100.0%`** (`+1201.4 R`, `278.0 st`) | **`100.0%`** (`+1368.5 R`, `351.0 st`) | `0.0%` (`+1019.5 R`, `fire: 50`) | `0.0%` (`+170.5 R`, `ghost: 50`) |
| **`hybrid_expected`** | **`curriculum_best_retention`** | **Stage 3** | **`100.0%`** (`+970.9 R`, `211.0 st`) | **`100.0%`** (`+1157.2 R`, `288.0 st`) | **`100.0%`** (`+1321.0 R`, `373.0 st`) | `0.0%` (`+313.4 R`, `ghost: 50`) | `0.0%` (`+126.0 R`, `ghost: 41, wisp: 8, laser: 1`) |

### 2.3 Level 4 & Level 5 `outcome_counts` & `max_x_reached` Diagnostics

| Run | Source | Lvl | `win` | `ghost` | `fire` | `laser` | `wisp` | `pit` | `timeout` | Diagnostics (`max_x`, Probe & Last-500) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `tabular curr` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | — |
| `tabular curr` | Best CP (`50`) | 4 | **13** | 15 | 0 | 22 | 0 | **0** | 0 | `10`-ep probe `q_table_lvl4_best`: `min/med/max = 1317.2 / 1789.1 / 2026.4` (`1` win, cleared Twin Lasers) |
| `tabular curr` | Train (`5000`) | 4 | **9** | 1578 | 729 | 653 | 0 | 1996 | 35 | `max_x` med/p90/max: `456.0 / 1222.0 / 2031.6`, last500 R `+937.97` (`50.59%`) |
| `tabular curr` | Final Eval (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `10`-ep probe `q_table_final`: `max_x = 329.2` |
| `tabular curr` | Best CP (`50`) | 5 | 0 | 18 | 0 | 0 | 0 | 32 | 0 | `10`-ep probe `q_table_lvl5_best`: `min/med/max = 412.0 / 412.0 / 473.2` |
| `tabular curr` | Train (`6000`) | 5 | 0 | 3210 | 9 | 1013 | 35 | 1733 | 0 | `max_x` med/p90/max: `392.0 / 452.0 / 895.6`, last500 R `+233.63` (`13.96%`) |
| `tabular sep` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep probe `q_table_lvl4_final`: `max_x = 1076.0` |
| `tabular sep` | Best CP (`50`) | 4 | 0 | 42 | 0 | 8 | 0 | **0** | 0 | `10`-ep probe `q_table_lvl4_best`: `min/med/max = 1322.4 / 1329.2 / 1333.2` (`+1154.11 R`) |
| `tabular sep` | Train (`5000`) | 4 | 0 | 1545 | 1027 | 119 | 0 | 2285 | 24 | `max_x` med/p90/max: `446.4 / 1008.8 / 1320.0`, last500 R `+724.65` (`40.44%`) |
| `tabular sep` | Final Eval (`50`) | 5 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep probe `q_table_lvl5_final`: `max_x = 426.0` |
| `tabular sep` | Best CP (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `10`-ep probe `q_table_lvl5_best`: `min/med/max = 482.0 / 490.0 / 498.0` (cleared Pit 1 `450`) |
| `tabular sep` | Train (`6000`) | 5 | 0 | 2933 | 0 | 885 | 113 | 2069 | 0 | `max_x` med/p90/max: `380.0 / 426.0 / 856.0`, last500 R `+272.06` (`14.64%`) |
| `hybrid curr` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep probe `q_table_final`: `max_x = 374.0` |
| `hybrid curr` | Best CP (`50`) | 4 | 0 | 0 | 0 | 50 | 0 | **0** | 0 | `10`-ep probe `q_table_lvl4_best`: `max_x = 1150.0` (`+894.1 R`) |
| `hybrid curr` | Train (`5000`) | 4 | 0 | 2007 | 372 | 9 | 0 | 2602 | 10 | `max_x` med/p90/max: `432.8 / 456.0 / 1672.8`, last500 R `+334.25` (`20.02%`) |
| `hybrid curr` | Final Eval (`50`) | 5 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep probe `q_table_final`: `max_x = 426.0` |
| `hybrid curr` | Best CP (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `10`-ep probe `q_table_lvl5_best`: `max_x = 308.8` |
| `hybrid curr` | Train (`6000`) | 5 | 0 | 1899 | 4 | 691 | 11 | 3395 | 0 | `max_x` med/p90/max: `416.0 / 428.8 / 865.6`, last500 R `+292.12` (`15.92%`) |
| `hybrid sep` | Final Eval (`50`) | 4 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | `10`-ep probe `q_table_lvl4_final`: `max_x = 936.0` |
| `hybrid sep` | Best CP (`50`) | 4 | 0 | 6 | 3 | 1 | 0 | 40 | 0 | `10`-ep probe `q_table_lvl4_best`: `min/med/max = 944.8 / 1076.0 / 1076.0` |
| `hybrid sep` | Train (`5000`) | 4 | 0 | 1367 | 1942 | 2 | 0 | 1689 | 0 | `max_x` med/p90/max: `458.0 / 936.0 / 1237.6`, last500 R `+698.32` (`41.81%`) |
| `hybrid sep` | Final Eval (`50`) | 5 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep probe `q_table_lvl5_final`: `max_x = 426.0` |
| `hybrid sep` | Best CP (`50`) | 5 | 0 | 1 | 0 | 37 | 11 | 1 | 0 | `10`-ep probe `q_table_lvl5_best`: **`min/med/max = 364.4 / 622.9 / 628.8`** (cleared Pit 1, LaserGhost 1 & Obs 2) |
| `hybrid sep` | Train (`6000`) | 5 | 0 | 2596 | 0 | 1492 | 16 | 1896 | 0 | `max_x` med/p90/max: `378.0 / 426.0 / 811.0`, last500 R `+271.57` (`14.66%`) |

### 2.4 Diagnostic Summary
1. **`tabular curriculum` Breakthrough Across L3–L4:** Achieved **`2.0%` L3 zero-shot**, **`100.0%` L3 BestCP** (`+1295.4 R`), **`26.0%` L4 BestCP** (`13/50` wins, `+1575.86 R`, `0/50` pit/fire deaths, `9/5000` train wins, `max_x = 2031.6`; 10-ep BestCP median `1789.1` clearing the Twin Laser Scissors `1150..1400`), AND locked **Stage 3** `q_table_curriculum_best.pkl` at **`100% / 100% / 100%`** on L1–L3 while `best_checkpoint_retention` also retained `100% / 100% / 100%` on L1–L3.
2. **Non-Zero Trace Floor + Alpha Cooling Fixed `tabular separate` L3 Final Collapse:** `tabular separate` achieved **`100.0%` Final AND `100.0%` BestCP** on L1, L2, AND L3 (`+1287.0 R`, `380.0` steps, `398/1800` train wins, `maxR50 = 92.0%`), and cleared L4 Pits 1–3 (`max_x = 1322.4..1333.2`, `0/50` pit deaths). Meanwhile `hybrid_expected separate` saw `109/1800` L3 train wins (`maxR50 = 30.0%`) without a `100%` greedy checkpoint (`+870.1 R`), while `hybrid_expected curriculum` hit **`100.0%` L3 BestCP** (`279/1800` train wins) and **`100% / 100% / 100%`** Stage-3 `curriculum_best_retention`.
3. **Level 5 Progress Beyond `x = 426` After Vertical Ghost-FOV Fix:** All 4 runs crossed Pit 1 (`380..450`) in training (`max_x = 811.0..895.6`, `31.9%–35.4%` progress). In greedy `BestCP` eval, `tabular curriculum` reached `412..473.2`, `tabular separate` cleared Pit 1 cleanly to `482..498`, and **`hybrid_expected separate` cleared Base Ghost 1, Pit 1, LaserGhost 1, and Obs 2 to median `622.9` (`max = 628.8`, `1/50` ghost & `1/50` pit death)** before low `WispGhost` (`610..740`) and `LaserGhost` 1 (`400..530`) stopped greedy runs.

---

## 3. Reproduction Commands
```bash
# Run Stage 1 unit & invariant tests (74/74 OK)
python -m unittest discover -s tests -p "test_*.py" -v

# Re-run Stage 2 4-run balanced benchmark
python train.py --agent all --mode all --reward balanced --lambda 0.65 --lambda-min 0.35 --alpha-min 0.005 --eval-interval 25 --rewarm-epsilon-hard 0.50 --max-steps-cap 2400 --no-progress-bar
```
