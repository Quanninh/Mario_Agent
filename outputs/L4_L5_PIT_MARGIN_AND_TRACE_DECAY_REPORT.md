# L4–L5 Pit-Margin Calibration, Wisp Noise Suppression, Decaying Eligibility Traces & Stage-Best Carryover Report

**Scope:** Discretizer `pit_immediate = 16.0` px & default `enable_wisp_sensor = False` (`55,296` states), `BalancedRewardStrategy` `wisp_threat_distance = -1.0` sync, per-stage linear eligibility trace decay (`λ = 0.65 -> 0.0` at 75%), curriculum `q_table_lvl{N}_best` carryover, `best_checkpoint_*` per-level tracking, tighter `eval_interval = 25`, 3-tier curriculum rewarm (`0.30 / 0.35 / 0.50`), expanded schedule `{1: 800, 2: 1200, 3: 1800, 4: 5000, 5: 6000}`, and 4-run local validation batch.

> Strict boundaries honored: `env/`, `core/base_env.py`, `DenseRewardStrategy`, `SparseRewardStrategy`, and the 12-tuple `DiscreteState` contract remained untouched.

---

## 1. Stage 1: Unit & Invariant Verification

### 1.1 Automated Test Suite (`69/69` Passing)
Verified via:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
# Ran 69 tests in 0.733s — OK
```
- Updated existing pit/wisp boundary tests in `tests/test_agent_discretizer_rewards.py` and `tests/test_l4_l5_upgrades.py` for `pit_immediate = 16.0` px, `enable_wisp_sensor = False` (`state_space_size == 55296`, `110592` when `True`), and `wisp_threat_distance = -1.0`.
- Added **4 new Phase 9 invariant checks** in `tests/test_pit_margin_and_trace_decay.py`:
  - `test_discretizer_pit_margins_and_wisp_suppression`: verifies `25.0 px` maps to `FAR_SAFE (1)`, `12.0 px` maps to `IMMEDIATE_SAFE (2)`, fire embers at `50.0 px` still map to `FIRE_DANGER (3)`, `wisp_bin == 0` when disabled vs `1/2` when enabled, and dynamic `state_space_size` (`55296` vs `110592`).
  - `test_balanced_reward_wisp_exemption_and_baselines`: verifies overhead wisps no longer exempt grounded jumps by default (`-15.0` penalty), while re-enabling `wisp_threat_distance=65.0` exempts it, and `DenseRewardStrategy` / `SparseRewardStrategy` remain untouched.
  - `test_linear_trace_decay_schedule_and_curriculum_carryover`: verifies `trace_decay` decays linearly from `0.65` to `0.0` at 75% of each stage (clearing active traces), resets to `0.65` on the next level, applies 3-tier curriculum rewarm (`0.30 / 0.35 / 0.50`), and logs metadata (`trace_decay_start`, `trace_decay_schedule`, `enable_wisp_sensor`, `rewarm_epsilon_hard`).
  - `test_carryover_uses_stage_best_not_drifted_final_and_separate_has_best_keys`: verifies curriculum mode loads `q_table_lvl1_best.pkl` Q-weights into Level 2 rather than drifted final weights, and `separate` mode records `best_checkpoint_*` keys in `evaluation_summary`.

### 1.2 100-Episode Balanced Smoke Test
Executed end-to-end across all 4 `balanced` runs (`--episodes 100 --lambda 0.65 --eval-interval 25 --max-steps-cap 2400`) into temporary output/checkpoint directories (verified and cleaned afterward):
- Confirmed metadata: `state_space_max = 55296`, `enable_wisp_sensor = False`, `trace_decay_start = 0.65`, `trace_decay_schedule = "linear_to_0_at_75pct"`, `rewarm_epsilon_hard = 0.5`.
- Confirmed `best_checkpoint_*` keys across all 5 levels in all 4 boards and `best_checkpoint_retention` in curriculum boards; `hybrid_expected` (`curriculum` & `separate`) hit `100.0%` L1 and `100.0%` L2 win rates within 100 episodes.

---

## 2. Stage 2: Full 4-Run `balanced` Benchmark (L1–L5, $\lambda: 0.65 \to 0.0$)

Executed concurrently on local laptop (`seed=42`, schedule `{1: 800, 2: 1200, 3: 1800, 4: 5000, 5: 6000}` = `14,800` episodes/run, `--lambda 0.65`, `--eval-interval 25`, `--rewarm-epsilon-hard 0.50`, `--max-steps-cap 2400`, `enable_wisp_sensor = False`):

### Output Boards Generated
- `outputs/training_board_tabular_curriculum_balanced.json`
- `outputs/training_board_tabular_separate_balanced.json`
- `outputs/training_board_hybrid_expected_curriculum_balanced.json`
- `outputs/training_board_hybrid_expected_separate_balanced.json`

### 2.1 Per-Stage Final & Best Checkpoint Summary (`50` Greedy Eval Episodes Each)

| Agent | Mode | Lvl | Untr Win% / Avg R | Final Win% / Avg R (Steps, Items) | Best CP Win% / Avg R (Steps, Items) | Train Wins / Max R50 / Max Prog | States |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`tabular`** | `curriculum` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+955.60` (`200.0`, `0.0`) | **`100.0%`** / `+955.80` (`250.0`, `0.0`) | `787/800` / `100.0%` / `100.0%` | `27` |
| `tabular` | `curriculum` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1185.40` (`262.0`, `0.0`) | **`100.0%`** / `+1188.48` (`270.0`, `0.0`) | `264/1200` / `50.0%` / `100.0%` | `65` |
| `tabular` | `curriculum` | L3 | `0.0%` / `+126.42` | `0.0%` / `+636.50` (`N/A`, `1.0`) | `2.0%` / `+845.78` (`492.0`, `1.26`) | `28/1800` / `12.0%` / `100.0%` | `175` |
| `tabular` | `curriculum` | L4 | `0.0%` / `+111.26` | `0.0%` / `+366.80` (`N/A`, `2.0`) | `0.0%` / `+584.20` (`N/A`, `1.0`) | `0/5000` / `0.0%` / `84.14%` | `340` |
| `tabular` | `curriculum` | L5 | `0.0%` / `+33.67` | `0.0%` / `+287.50` (`N/A`, `1.0`) | `0.0%` / `+160.40` (`N/A`, `0.0`) | `0/6000` / `0.0%` / `34.53%` | `593` |
| **`tabular`** | `separate` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+955.60` (`200.0`, `0.0`) | **`100.0%`** / `+955.80` (`250.0`, `0.0`) | `787/800` / `100.0%` / `100.0%` | `27` |
| `tabular` | `separate` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1138.70` (`303.0`, `0.0`) | **`100.0%`** / `+1138.70` (`303.0`, `0.0`) | `295/1200` / `74.0%` / `100.0%` | `63` |
| `tabular` | `separate` | L3 | `0.0%` / `+126.42` | `0.0%` / `+4.80` (`N/A`, `0.0`) | **`100.0%`** / **`+1358.60`** (`414.0`, `3.0`) | `116/1800` / `42.0%` / `100.0%` | `169` |
| `tabular` | `separate` | L4 | `0.0%` / `+111.26` | `0.0%` / **`+1168.80`** (`N/A`, `2.0`) | `0.0%` / **`+1188.70`** (`N/A`, `3.0`) | `0/5000` / `0.0%` / `64.15%` | `288` |
| `tabular` | `separate` | L5 | `0.0%` / `+33.67` | `0.0%` / `+225.60` (`N/A`, `1.0`) | `0.0%` / **`+462.22`** (`N/A`, `1.76`) | `0/6000` / `0.0%` / `33.23%` | `527` |
| **`hybrid_expected`** | `curriculum` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+908.30` (`207.0`, `0.0`) | **`100.0%`** / `+943.90` (`211.0`, `0.0`) | `789/800` / `100.0%` / `100.0%` | `27` |
| `hybrid_expected` | `curriculum` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1131.10` (`319.0`, `0.0`) | **`100.0%`** / `+1131.10` (`319.0`, `0.0`) | `398/1200` / `100.0%` / `100.0%` | `59` |
| `hybrid_expected` | `curriculum` | L3 | `0.0%` / `+126.42` | `0.0%` / `+520.90` (`N/A`, `1.0`) | **`100.0%`** / `+1240.50` (`727.0`, `2.0`) | `9/1800` / `4.0%` / `100.0%` | `164` |
| `hybrid_expected` | `curriculum` | L4 | `0.0%` / `+111.26` | `0.0%` / `+374.10` (`N/A`, `2.0`) | `0.0%` / `+754.26` (`N/A`, `1.0`) | `0/5000` / `0.0%` / `62.77%` | `313` |
| `hybrid_expected` | `curriculum` | L5 | `0.0%` / `+33.67` | `0.0%` / `+273.40` (`N/A`, `1.0`) | `0.0%` / `+284.89` (`N/A`, `1.0`) | `0/6000` / `0.0%` / `25.83%` | `600` |
| **`hybrid_expected`** | `separate` | L1 | `6.67%` / `+190.84` | **`100.0%`** / `+908.30` (`207.0`, `0.0`) | **`100.0%`** / `+943.90` (`211.0`, `0.0`) | `789/800` / `100.0%` / `100.0%` | `27` |
| `hybrid_expected` | `separate` | L2 | `0.0%` / `+96.42` | **`100.0%`** / `+1073.90` (`291.0`, `0.0`) | **`100.0%`** / `+1137.40` (`346.0`, `0.0`) | `521/1200` / `94.0%` / `100.0%` | `61` |
| `hybrid_expected` | `separate` | L3 | `0.0%` / `+126.42` | `0.0%` / `+676.60` (`N/A`, `2.0`) | **`100.0%`** / `+1331.00` (`390.0`, `2.0`) | `127/1800` / `34.0%` / `100.0%` | `164` |
| `hybrid_expected` | `separate` | L4 | `0.0%` / `+111.26` | `0.0%` / `+953.42` (`N/A`, `2.0`) | `0.0%` / `+950.76` (`N/A`, `2.0`) | `0/5000` / `0.0%` / `61.67%` | `225` |
| `hybrid_expected` | `separate` | L5 | `0.0%` / `+33.67` | `0.0%` / `+230.00` (`N/A`, `1.0`) | `0.0%` / `+329.15` (`N/A`, `1.0`) | `0/6000` / `0.0%` / `25.57%` | `435` |

**Zero-Shot Curriculum Transfers (`50` eps):**
- `tabular curriculum`: `level_3_zero_shot`: `0.0%` (`+555.52 R`, `2.0 it`, `ghost_death: 46, fire_death: 4`); `level_5_zero_shot`: `0.0%` (`+212.13 R`, `0.76 it`, `ghost_death: 26, pit_death: 24`).
- `hybrid_expected curriculum`: `level_3_zero_shot`: `0.0%` (`+537.45 R`, `2.04 it`, `ghost_death: 44, fire_death: 5, pit_death: 1`); `level_5_zero_shot`: `0.0%` (`+170.79 R`, `0.94 it`, `ghost_death: 44, laser_death: 5, wisp_death: 1`).

### 2.2 Curriculum Retention Matrices (`50` Greedy Eval Episodes Per Level)

| Agent | Retention Matrix | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`tabular`** | `post_curriculum_retention` | `0.0%` (`+71.2 R`, `timeout: 50`) | `0.0%` (`-27.2 R`, `timeout: 50`) | `0.0%` (`-27.2 R`, `timeout: 50`) | `0.0%` (`-102.64 R`, `timeout: 50`) | `0.0%` (`+287.5 R`, `laser: 50`) |
| **`tabular`** | `best_checkpoint_retention` | **`100.0%`** (`+1010.4 R`, `188.0 st`) | **`100.0%`** (`+1203.6 R`, `284.0 st`) | `14.0%` (`+933.46 R`, `367.7 st`, `2.02 it`) | `0.0%` (`+276.2 R`, `pit: 50`) | `0.0%` (`+160.4 R`, `ghost: 50`) |
| **`hybrid_expected`** | `post_curriculum_retention` | **`100.0%`** (`+1001.2 R`, `224.0 st`) | **`100.0%`** (`+1192.6 R`, `280.0 st`) | `0.0%` (`+536.7 R`, `ghost: 50`) | `0.0%` (`+246.8 R`, `pit: 50`) | `0.0%` (`+273.4 R`, `pit: 50`) |
| **`hybrid_expected`** | `best_checkpoint_retention` | **`100.0%`** (`+999.3 R`, `211.0 st`) | **`100.0%`** (`+1193.0 R`, `302.0 st`) | **`100.0%`** (`+1381.5 R`, `395.0 st`, `3.0 it`) | `0.0%` (`+347.3 R`, `pit: 50`) | `0.0%` (`+284.89 R`, `ghost: 23, pit: 16, laser: 11`) |

### 2.3 Level 4 & Level 5 `outcome_counts` & Training `max_x_reached` Diagnostics

| Run | Source | Lvl | `win` | `ghost` | `fire` | `laser` | `wisp` | `pit` | `timeout` | Diagnostics (`max_x`, Last-500) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `tabular curr` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | — |
| `tabular curr` | Best CP (`50`) | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 744.0` |
| `tabular curr` | Train (`5000`) | 4 | 0 | 1990 | 917 | 264 | 0 | 1742 | 87 | `max_x` med/p90/max: `442.0 / 944.9 / 1712.6`, last500 R `+503.74` (`29.39%`) |
| `tabular curr` | Final Eval (`50`) | 5 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 415.6` |
| `tabular curr` | Best CP (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 304.8` |
| `tabular curr` | Train (`6000`) | 5 | 0 | 3251 | 2 | 931 | 36 | 1780 | 0 | `max_x` med/p90/max: `383.6 / 426.0 / 874.0`, last500 R `+265.68` (`14.24%`) |
| `tabular sep` | Final Eval (`50`) | 4 | 0 | 0 | 0 | **50** | 0 | **0** | 0 | Probe (`10` eps): `max_x = 1310.0` (cleared Pits 1–3; hit Twin Lasers) |
| `tabular sep` | Best CP (`50`) | 4 | 0 | 0 | 0 | **50** | 0 | **0** | 0 | Probe (`10` eps): `max_x = 1300.4` (`+1188.7 R`, `3.0` items) |
| `tabular sep` | Train (`5000`) | 4 | 0 | 2155 | 1280 | 253 | 0 | 1268 | 44 | `max_x` med/p90/max: `426.0 / 860.0 / 1317.6`, last500 R `+574.03` (`33.68%`) |
| `tabular sep` | Final Eval (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 337.2` |
| `tabular sep` | Best CP (`50`) | 5 | 0 | 38 | 0 | 12 | **0** | **0** | 0 | Probe (`10` eps): `max_x = 644.0` (`+462.22 R`, `1.76` items) |
| `tabular sep` | Train (`6000`) | 5 | 0 | 3760 | 2 | 751 | 86 | 1401 | 0 | `max_x` med/p90/max: `392.0 / 453.2 / 842.8`, last500 R `+273.15` (`14.49%`) |
| `hybrid curr` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | Probe (`10` eps): `max_x = 434.0` |
| `hybrid curr` | Best CP (`50`) | 4 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 924.0` (max `932.0`) |
| `hybrid curr` | Train (`5000`) | 4 | 0 | 2046 | 464 | 28 | 0 | 2448 | 14 | `max_x` med/p90/max: `436.8 / 701.6 / 1290.4`, last500 R `+499.80` (`27.71%`) |
| `hybrid curr` | Final Eval (`50`) | 5 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | Probe (`10` eps): `max_x = 392.0` |
| `hybrid curr` | Best CP (`50`) | 5 | 0 | 23 | 0 | 11 | 0 | 16 | 0 | Probe (`10` eps): `max_x = 409.2` (max `421.2`) |
| `hybrid curr` | Train (`6000`) | 5 | 0 | 3016 | 0 | 701 | 5 | 2278 | 0 | `max_x` med/p90/max: `388.0 / 426.0 / 666.4`, last500 R `+276.21` (`14.61%`) |
| `hybrid sep` | Final Eval (`50`) | 4 | 0 | 26 | 0 | 19 | 0 | 5 | 0 | Probe (`10` eps): med `max_x = 1130.0` (range `1118.0..1134.0`) |
| `hybrid sep` | Best CP (`50`) | 4 | 0 | 23 | 0 | 23 | 0 | 4 | 0 | Probe (`10` eps): med `max_x = 1134.0` (`+950.76 R`, `2.0` items) |
| `hybrid sep` | Train (`5000`) | 4 | 0 | 1681 | 922 | 2 | 0 | 2395 | 0 | `max_x` med/p90/max: `442.8 / 800.4 / 1268.7`, last500 R `+382.40` (`22.82%`) |
| `hybrid sep` | Final Eval (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | Probe (`10` eps): `max_x = 348.4` |
| `hybrid sep` | Best CP (`50`) | 5 | 0 | 49 | 0 | 1 | **0** | **0** | 0 | Probe (`10` eps): `max_x = 480.0` (`+329.15 R`, `1.0` items) |
| `hybrid sep` | Train (`6000`) | 5 | 0 | 3522 | 0 | 302 | 18 | 2158 | 0 | `max_x` med/p90/max: `340.4 / 426.0 / 660.0`, last500 R `+227.19` (`12.60%`) |

### 2.4 Diagnostic Conclusions (Improvements vs. Remaining Bottlenecks)
1. **Verified Improvements:**
   - **State Compression:** Suppressing overhead wisp noise reduced Level 5 discovered states across every configuration (`873 -> 593`, `665 -> 527`, `678 -> 600`, `639 -> 435` — `-31.9%` on `hybrid_expected separate`), while `wisp_death` remained `0/50` across all Final/BestCP greedy evaluations.
   - **Level 2 & Level 3 Stabilization:** `hybrid_expected separate` Level 2 rose from `0.0%` to **`100.0%`** Final/BestCP (`+1073.9 / +1137.4 R`); **3 of 4 runs (`tabular separate`, `hybrid_expected curriculum`, `hybrid_expected separate`)** achieved **`100.0%` BestCP on Level 3** (`+1358.6 / +1240.5 / +1331.0 R`); and `hybrid_expected curriculum` achieved **`100.0% / 100.0% / 100.0%`** on Levels 1–3 in `best_checkpoint_retention`.
   - **Pit-Margin Improvement on Level 4:** `tabular separate` cleared Pits 1–3 (`x = 400..480, 880..960, 1020..1100`) with **`0/50` pit deaths** and **`0/50` fire deaths** (`Best R +809.1 -> +1188.7`, `Final R +784.9 -> +1168.8`, `max_x = 1300.4..1310.0`), and `hybrid_expected separate` dropped pit deaths to `4/50` BestCP (`max_x = 1134.0`).
2. **Remaining Bottlenecks & Regressions:**
   - **L4 Twin `LaserGhost` Gauntlet & Early Trace Decay Cutoff:** Unlike the prior constant `λ = 0.65` run where `hybrid_expected separate` reached `14%–26%` BestCP on L4, decaying `λ -> 0.0` at 75% (`ep 3750`) truncated late-stage multi-step credit assignment across `x = 1150..1400`, where two overlapping out-of-phase `LaserGhost`s (`patrol 1150..1360` & `1220..1400`, `rot = ±0.045`) create beam-angle timing hazards that the proximity-only `laser_bin` does not distinguish (`50/50 laser_death` in `tabular separate`; `23 laser + 23 ghost` in `hybrid_expected separate`).
   - **Curriculum Late-Stage Overwrite:** Neither curriculum run solved L4 (`0%`), so L4/L5 drift with `35%` replay eroded `tabular curriculum` L1–L4 into `timeout: 50` in `post_curriculum_retention` and `hybrid_expected curriculum` L3 from `100%` BestCP to `0%` final.
   - **Level 5 Early Corridor Congestion (`x = 320..644`):** Ground `Ghost` (`patrol 260..360`), Pit 1 (`380..450`), `LaserGhost` (`patrol 400..530`), and low `WispGhost` (`patrol 610..740`) stack in the opening `450 px`, capping all greedy evaluations at `max_x = 304.8..644.0`.

---

## 3. Reproduction Commands
```bash
# Run Stage 1 unit & invariant tests (69/69 OK)
python -m unittest discover -s tests -p "test_*.py" -v

# Re-run Stage 2 4-run balanced benchmark
python train.py --agent all --mode all --reward balanced --lambda 0.65 --eval-interval 25 --rewarm-epsilon-hard 0.50 --max-steps-cap 2400 --no-progress-bar
```
