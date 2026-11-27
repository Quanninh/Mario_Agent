# L4–L5 Overhead Ghost Fix, Laser Jump Suppression & Polish-Tail Anchor Report (Phase 11 vs. Phase 10)

**Spec:** `specs/PHASE_11_OVERHEAD_LASER_AND_POLISH_ANCHOR.md`  
**Scope:** Standing-head reference for airborne `ghost_alt_bin` (`discretizer/state_discretizer.py`), grounded laser jump penalty (`rewards/balanced_reward.py`: default `laser_threat_distance = -1.0`), Polish-Tail `85%` best-weight onset anchor + mid-polish anti-regression rollbacks with `alpha` halving (`training/trainer.py`, `train.py`: `--polish-anchor-best`), expanded L5 budget `{1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 10000}` (`20,800` eps/run), and `5` new tests (`tests/test_phase11_overhead_laser_and_polish.py`).

> Strict Section 1 boundaries honored: `env/`, `core/base_env.py`, `core/interfaces.py`, `rewards/dense_reward.py`, and `rewards/sparse_reward.py` remained untouched; in `rewards/balanced_reward.py` only default `laser_threat_distance` (`-1.0`) and its non-negative gate changed; exact 12-tuple `DiscreteState` shape and `state_space_size == 55296` (`enable_wisp_sensor = False`) / `110592` (`True`) preserved.

---

## 1. Stage 1 & Step 2 Verification Summary

* **Full Unit Test Suite:**
  ```bash
  python -m unittest discover -s tests -p "test_*.py" -v
  # Ran 85 tests in 0.984s — OK (0 failures, 0 skipped)
  ```
  All `80` prior tests (with clean default updates in `tests/test_l4_l5_upgrades.py`, `tests/test_agent_discretizer_rewards.py`, and `tests/test_pit_margin_and_trace_decay.py`) + `5` new Phase 11 tests passed cleanly.
* **100-Episode Balanced Smoke Test (`Step 2`):** Exited `0` across all `4` configurations into temporary `outputs/p11_*` directories (cleaned after verification), confirming `polish_anchor_best = True`, `state_space_max = 55296`, `polish_anchor_reloaded`, and `polish_regression_rollbacks` active (`hybrid_expected separate` hit `100.0%` L2 in `100` eps; anti-regression rollbacks fired on `tabular curriculum` L3 and `tabular separate` L4).

---

## 2. Step 4 Acceptance Gate Scorecard (Phase 10 vs. Phase 11)

| Gate | Target Criterion | Phase 10 Result | Phase 11 Result | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **L1–L3** | `100.0%` Final AND `100.0%` BestCP across all 4 runs (`12/12` + `12/12`) | `12/12` BestCP, `9/12` Final (`tab_sep` L3 `0%` collapse; `tab_curr`/`hyb_curr` L3 `0%`) | **`tab_sep` & `hyb_sep`: `100% / 100% / 100%` Final & BestCP (`6/6`)**; **`tab_curr` L3 Final jumped `0% -> 44.0%`**; `hyb_curr` end-Stage-3 `0%` (yet **BOTH curriculum runs hit `100.0%` L1–L3 in retention!**) | **PARTIAL PASS** (`10/12` Final `100%`, `10/12` BestCP `100%`) |
| **L4** | Non-zero eval wins on `>= 2` runs (`BestCP >= 30%`, `Final >= 20%`); `x=1258` laser deaths eliminated; `tab_sep` preserves `x>=1826` | `0%` eval on all 4 runs; `0` train wins; `tab_curr` & `hyb_sep` `50/50 laser_death` at `1258–1262`; `tab_sep` `+60 R` timeout collapse | **`hyb_curr` hit `100.0%` Final AND `100.0%` BestCP (`+2185.0 R`, `620 st`, `4.0` it)** AND **Stage-5 `tab_curr` hit `100.0%` (`+2215.7 R`, `623 st`, `4.0` it)**! All **4/4** runs scored train wins (`39/10/93/36`, peaks `2029.6–2029.8`) & **`0/200` Final eval laser deaths**; `tab_sep` held `+1317.7 R` (`1517.7` probe) | **PARTIAL PASS** (`2/2` Curriculum L4=`100%` by Stage 5; `1/4` end-Stage-4 eval `>=20%`) |
| **L5** | `x=482` overhead `Laser Sentry 1` deaths eliminated; median `max_x > 650` & peak `max_x > 1,200` (`>= 1` run) | Train med `356–414`, peak `812–936`; `hyb_curr` `last500 med = 482` (`ghost: 50`); BestCP probes `428–480` | **`x=482` eval `ghost_death` eliminated on `hyb_sep` (`766.0` probe, `last500 med = 664.0 > 650`, `50/50 fire@760`) & `tab_curr` BestCP (`884.7` med, `max 902.7`, `+743.9 R`)**; full 10k train med `384–417`, peak `916.5` (`< 1,200`) | **PARTIAL / FAIL** (`>650` in BestCP probes & `last500`; peak `916.5 < 1,200`) |
| **Curriculum Retention** | `curriculum_best_checkpoint_stage >= 4` on both runs with `>= 95%` on L1–L3 | Stage `5` on both (`100%/100%/100%` L1–L3; L4 `0%`: `+1095.1 / +778.5`) | **Stage `5` on BOTH runs with `100.0% / 100.0% / 100.0% / 100.0%` across L1, L2, L3, AND L4** (`+2215.7 / +2237.7 R` on L4, **AND `100%` across L1–L4 even in `post_curriculum_retention`!**) | **PASS** (`2/2` Runs, L1–L4 `100%`) |

---

## 3. Side-by-Side Per-Stage Comparison (Phase 10 vs. Phase 11, `50` Greedy Eval Episodes)

Phase 10 schedule: `{1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 8000}` (`18,800` eps). Phase 11 schedule: `{1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 10000}` (`20,800` eps).

### 3.1 `tabular` (`curriculum` & `separate`)

| Mode | Lvl | P10 Final% / BestCP% (Final R / Best R) | P11 Final% / BestCP% (Final R / Best R) | P11 Steps (Final/Best) / Items | P11 BestEp / Reload / Rollbacks | P11 Train Wins / MaxR50 / MaxProg | P11 States (Best / Final) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `curriculum` | L1 | `100.0%` / `100.0%` (`+955.6` / `+966.0`) | **`100.0%`** / **`100.0%`** (`+919.6` / `+966.0`) | `200.0` / `214.0` / `0.0` | `425` / `True` / `0` | `787/800` / `100.0%` / `100.0%` | `27` / `27` |
| `curriculum` | L2 | `100.0%` / `100.0%` (`+1075.6` / `+1056.9`) | **`100.0%`** / **`100.0%`** (`+1056.9` / `+1056.9`) | `609.8` / `609.8` / `0.0` | `50` / `True` / `8` | `427/1200` / `92.0%` / `100.0%` | `58` / `58` |
| `curriculum` | L3 | `0.0%` / `100.0%` (`+470.6` / `+1324.1`) | **`44.0%`** / **`44.0%`** (**`+1065.9`** / `+1065.9`) | `387.3` / `387.3` / `1.44` | `800` / `True` / `8` | `183/1800` / `48.0%` / `100.0%` | `160` / `160` |
| `curriculum` | L4 | `0.0%` / `0.0%` (`+1061.5` / `+1113.5`) | `0.0%` / `0.0%` (`+905.0` / `+905.0`) *(100% by St.5)* | `N/A` / `N/A` / `0.0` | `4475` / `True` / `3` | **`39/7000`** / **`10.0%`** / **`100.0%`** | `383` / `387` |
| `curriculum` | L5 | `0.0%` / `0.0%` (`+286.4` / `+300.7`) | `0.0%` / `0.0%` (`+250.8` / **`+743.9`**) | `N/A` / `N/A` / `1.0 / 2.0` | `9025` / `True` / `1` | `0/10000` / `0.0%` / `36.32%` | `645` / `653` |
| `separate` | L1 | `100.0%` / `100.0%` (`+955.6` / `+966.0`) | **`100.0%`** / **`100.0%`** (`+919.6` / `+966.0`) | `200.0` / `214.0` / `0.0` | `425` / `True` / `0` | `787/800` / `100.0%` / `100.0%` | `27` / `27` |
| `separate` | L2 | `100.0%` / `100.0%` (`+1056.4` / `+1056.4`) | **`100.0%`** / **`100.0%`** (`+1056.4` / `+1056.4`) | `316.0` / `316.0` / `0.0` | `550` / `False` / `0` | `525/1200` / `100.0%` / `100.0%` | `62` / `63` |
| `separate` | L3 | `0.0%` / `100.0%` (`+21.0` / `+1269.1`) | **`100.0%`** / **`100.0%`** (**`+1321.4`** / **`+1321.4`**) | `396.0` / `396.0` / `2.0` | `1425` / `True` / `11` | `89/1800` / `20.0%` / `100.0%` | `159` / `159` |
| `separate` | L4 | `0.0%` / `0.0%` (`+60.0` / `+1705.0`) | `0.0%` / `0.0%` (**`+1317.7`** / `+1317.7`) | `N/A` / `N/A` / `2.0` | `6600` / `True` / `34` | **`10/7000`** / **`6.0%`** / **`100.0%`** | `259` / `259` |
| `separate` | L5 | `0.0%` / `0.0%` (`+1.7` / `+343.9`) | `0.0%` / `0.0%` (**`+418.4`** / **`+418.4`**) | `N/A` / `N/A` / **`1.84`** | `2450` / `True` / `60` | `0/10000` / `0.0%` / `26.03%` | `316` / `316` |

### 3.2 `hybrid_expected` (`curriculum` & `separate`)

| Mode | Lvl | P10 Final% / BestCP% (Final R / Best R) | P11 Final% / BestCP% (Final R / Best R) | P11 Steps (Final/Best) / Items | P11 BestEp / Reload / Rollbacks | P11 Train Wins / MaxR50 / MaxProg | P11 States (Best / Final) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `curriculum` | L1 | `100.0%` / `100.0%` (`+907.8` / `+946.8`) | **`100.0%`** / **`100.0%`** (**`+946.8`** / `+946.8`) | `212.0` / `212.0` / `0.0` | `25` / `True` / `0` | `788/800` / `100.0%` / `100.0%` | `25` / `25` |
| `curriculum` | L2 | `100.0%` / `100.0%` (`+1113.6` / `+1113.6`) | **`100.0%`** / **`100.0%`** (`+1113.6` / `+1113.6`) | `284.0` / `284.0` / `0.0` | `400` / `False` / `0` | `627/1200` / `100.0%` / `100.0%` | `58` / `59` |
| `curriculum` | L3 | `0.0%` / `100.0%` (`+533.4` / `+1246.2`) | `0.0%` / `0.0%` (`+623.8` / `+839.6`) *(100% by St.5)* | `N/A` / `N/A` / `2.0 / 1.0` | `550` / `True` / `0` | `47/1800` / `12.0%` / `100.0%` | `157` / `158` |
| `curriculum` | L4 | `0.0%` / `0.0%` (`+940.7` / `+750.5`) | **`100.0%`** / **`100.0%`** (**`+2185.0`** / **`+2185.0`**) | `620.0` / `620.0` / **`4.0`** | `4700` / `True` / `15` | **`93/7000`** / **`18.0%`** / **`100.0%`** | `349` / `349` |
| `curriculum` | L5 | `0.0%` / `0.0%` (`+344.3` / `+320.0`) | `0.0%` / `0.0%` (`+303.1` / `+303.1`) | `N/A` / `N/A` / `1.0` | `8550` / `True` / `1` | `0/10000` / `0.0%` / `36.31%` | `609` / `615` |
| `separate` | L1 | `100.0%` / `100.0%` (`+907.8` / `+946.8`) | **`100.0%`** / **`100.0%`** (**`+946.8`** / `+946.8`) | `212.0` / `212.0` / `0.0` | `25` / `True` / `0` | `788/800` / `100.0%` / `100.0%` | `25` / `25` |
| `separate` | L2 | `100.0%` / `100.0%` (`+1133.4` / `+1133.4`) | **`100.0%`** / **`100.0%`** (`+1133.4` / `+1133.4`) | `266.0` / `266.0` / `0.0` | `450` / `False` / `0` | `572/1200` / `100.0%` / `100.0%` | `57` / `60` |
| `separate` | L3 | `100.0%` / `100.0%` (`+1335.2` / `+1335.2`) | **`100.0%`** / **`100.0%`** (`+1241.3` / `+1241.3`) | `357.0` / `357.0` / `2.0` | `800` / `True` / `1` | `379/1800` / `70.0%` / `100.0%` | `150` / `155` |
| `separate` | L4 | `0.0%` / `0.0%` (`+1107.4` / `+1166.2`) | `0.0%` / `0.0%` (**`+1620.1`** / **`+1620.2`**) | `N/A` / `N/A` / **`4.0`** | `6050` / `True` / `5` | **`36/7000`** / **`6.0%`** / **`100.0%`** | `264` / `273` |
| `separate` | L5 | `0.0%` / `0.0%` (`+299.6` / `+339.7`) | `0.0%` / `0.0%` (**`+567.1`** / **`+583.2`**) | `N/A` / `N/A` / `1.0` | `9975` / `True` / `29` | `0/10000` / `0.0%` / `33.78%` | `374` / `374` |

**Zero-Shot Curriculum Transfers (`50` eps):**
* `tabular curriculum`: `level_3_zero_shot`: `0.0%` (`+461.31 R`, `2.0` it, `ghost: 48, pit: 2`); `level_5_zero_shot`: `0.0%` (`+197.35 R`, `0.9` it, `ghost: 23, pit: 12, wisp: 9, laser: 5, fire: 1`).
* `hybrid_expected curriculum`: `level_3_zero_shot`: `0.0%` (`+551.10 R`, `2.02` it, `ghost: 44, fire: 3, pit: 3`); `level_5_zero_shot`: `0.0%` (`+252.62 R`, `1.0` it, `laser: 19, ghost: 17, pit: 14`).

---

## 4. Curriculum Retention Matrices (`50` Greedy Eval Episodes Per Level)

| Agent | Retention Matrix | Stage | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tabular`** | `post_curriculum_retention` | Final (L5) | **`100.0%`** (`+1000.1 R`, `203 st`) | **`100.0%`** (`+1179.6 R`, `284 st`) | **`100.0%`** (`+1360.0 R`, `382 st`) | **`100.0%`** (`+2215.7 R`, `623 st`, `4 it`) | `0.0%` (`+250.8 R`, `laser: 50`) |
| **`tabular`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+1000.1 R`, `203 st`) | **`100.0%`** (`+1179.6 R`, `284 st`) | **`100.0%`** (`+1360.0 R`, `382 st`) | **`100.0%`** (`+2215.7 R`, `623 st`, `4 it`) | `0.0%` (**`+743.9 R`**, `laser: 49, ghost: 1`) |
| **`tabular`** | **`curriculum_best_retention`** | **Stage 5** | **`100.0%`** (`+1000.1 R`, `203 st`) | **`100.0%`** (`+1179.6 R`, `284 st`) | **`100.0%`** (`+1360.0 R`, `382 st`) | **`100.0%`** (`+2215.7 R`, `623 st`, `4 it`) | `0.0%` (**`+739.6 R`**, `laser: 49, ghost: 1`) |
| **`hybrid_expected`** | `post_curriculum_retention` | Final (L5) | **`100.0%`** (`+986.3 R`, `217 st`) | **`100.0%`** (`+1168.8 R`, `300 st`) | **`100.0%`** (`+1373.8 R`, `356 st`) | **`100.0%`** (`+2237.7 R`, `653 st`, `5 it`) | `0.0%` (`+303.1 R`, `pit: 50`) |
| **`hybrid_expected`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+986.3 R`, `217 st`) | **`100.0%`** (`+1168.8 R`, `300 st`) | **`100.0%`** (`+1373.8 R`, `356 st`) | **`100.0%`** (`+2237.7 R`, `653 st`, `5 it`) | `0.0%` (`+303.1 R`, `pit: 50`) |
| **`hybrid_expected`** | **`curriculum_best_retention`** | **Stage 5** | **`100.0%`** (`+986.3 R`, `217 st`) | **`100.0%`** (`+1168.8 R`, `300 st`) | **`100.0%`** (`+1373.8 R`, `356 st`) | **`100.0%`** (`+2237.7 R`, `653 st`, `5 it`) | `0.0%` (`+303.1 R`, `pit: 50`) |

**Zero catastrophic forgetting across all 4 cleared levels:** Every retention matrix (`post_curriculum_retention`, `best_checkpoint_retention`, AND Stage-5 `curriculum_best_retention`) on **both** `tabular` and `hybrid_expected` retained **`100.0% / 100.0% / 100.0% / 100.0%`** across **Levels 1, 2, 3, AND 4** simultaneously after `10,000` episodes of Level 5 training.

---

## 5. Level 4 & Level 5 `outcome_counts`, `max_x_reached`, `top_death_clusters` & 10-Ep Probes

| Run | Source | Lvl | `win` | `ghost` | `fire` | `laser` | `wisp` | `pit` | `timeout` | `max_x` & Top Death Cluster Diagnostics |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `tab curr` | Stage 4 Final/Best | 4 | 0 | **0** | **0** | **0** | 0 | 50 | 0 | `lvl4_best` probe `1061.4`; **Stage-5 Final & `curr_best` probes `2026.0` (`10/10` win, `+2215.7 R`)** |
| `tab curr` | Train (`7000`) | 4 | **39** | 2308 | 1322 | 389 | 0 | 2884 | 58 | `med/p90/peak`: `442.0 / 1125.4 / 2029.7`; `last500 med = 1061.4` (`+923.7 R`); top cluster `pit x=1060 (294)` |
| `tab curr` | Stage 5 Final/Best | 5 | 0 | 0 / 1 | 0 | 50 / 49 | 0 | **0** | 0 | **`lvl5_best` probe `874.7 / 884.7 / 902.7`** (`+744.8 R`, `laser: 9, ghost: 1`); `final`: `361.2` |
| `tab curr` | Train (`10000`) | 5 | 0 | 3724 | 23 | 2209 | 103 | 3941 | 0 | `med/p90/peak`: `404.4 / 453.2 / 916.5`; `last500 med = 365.2`; top cluster `laser x=360 (155)` |
| `tab sep` | Stage 4 Final/Best | 4 | 0 | **0** | **0** | **0** | 0 | **0** | 50 | `lvl4_best` & `lvl4_final` probes: **`1517.7`** (`timeout: 10`, `+1317.7 R`, cleared Twin Lasers) |
| `tab sep` | Train (`7000`) | 4 | **10** | 2268 | 1788 | 415 | 0 | 2453 | 66 | `med/p90/peak`: `424.0 / 1076.0 / 2029.8`; `last500 med = 805.3` (`+680.8 R`) |
| `tab sep` | Stage 5 Final/Best | 5 | 0 | 50 | **0** | **0** | 0 | **0** | 0 | `lvl5_best` probe **`635.6` (`max = 667.6`, `+418.0 R`)**; `lvl5_final` **`634.5`** (`+422.6 R`, zero collapse!) |
| `tab sep` | Train (`10000`) | 5 | 0 | 4402 | 0 | 2168 | 206 | 3224 | 0 | `med/p90/peak`: `384.0 / 426.0 / 671.1`; `last500 med = 395.7`; top cluster `laser x=400 (149)` |
| `hyb curr` | Stage 4 Final/Best | 4 | **50** | **0** | **0** | **0** | 0 | **0** | 0 | **`lvl4_best` probe `2028.0` (`+2185.0 R`, `620 st`); Stage-5 `final`/`curr_best` `2030.0` (`+2237.7 R`, `5.0` it)** |
| `hyb curr` | Train (`7000`) | 4 | **93** | 2586 | 415 | 330 | 0 | 3574 | 2 | `med/p90/peak`: `456.0 / 1244.0 / 2029.6`; **`last500 med = 1242.8` (`+1136.2 R`)** |
| `hyb curr` | Stage 5 Final/Best | 5 | 0 | **0** | 0 | **0** | 0 | 50 | 0 | `lvl5_best` / `final` probes `426.0` (`pit_death: 10`, `+303.1 R`) |
| `hyb curr` | Train (`10000`) | 5 | 0 | 3263 | 40 | 1168 | 43 | 5486 | 0 | `med/p90/peak`: `417.2 / 428.0 / 916.4`; top cluster `pit x=420 (456)` |
| `hyb sep` | Stage 4 Final/Best | 4 | 0 | **0** | 50 | **0** | 0 | **0** | 0 | `lvl4_best` / `final` probes **`1704.0`** (`fire: 10`, `+1620.2 / +1620.1 R`, `4.0` it, Pit 4 fire) |
| `hyb sep` | Train (`7000`) | 4 | **36** | 3301 | 1376 | 546 | 0 | 1741 | 0 | `med/p90/peak`: `447.6 / 1348.8 / 2029.8`; **`last500 med = 1696.0` (`+1298.4 R`)** |
| `hyb sep` | Stage 5 Final/Best | 5 | 0 | **0** | 50 | **0** | 0 | **0** | 0 | `lvl5_best` probe **`766.0`** (`+583.0 R`); `final` **`760.0`** (`+566.2 R`, Pit 2 hellfire, `0` ghost/laser!) |
| `hyb sep` | Train (`10000`) | 5 | 0 | 4838 | 188 | 857 | 40 | 4077 | 0 | `med/p90/peak`: `410.8 / 522.8 / 856.0`; **`last500 med = 664.0`** (`+379.4 R`) |

---

## 6. Root-Cause Findings & Impact of Phase 11 Changes

1. **Phase 11A + 11B Completely Unlocked L4 Twin Laser Scissors (`x = 1220..1360`):**
   * Fixing airborne `ghost_alt_bin` to compare against `standing_head_y` (`Phase 11A`) + penalizing grounded laser jumps by default (`laser_threat_distance = -1.0`, `Phase 11B`) **eliminated `100%` (`0/200`) of Final eval `laser_death` on Level 4 across all 4 runs** (down from `100/200` in Phase 10).
   * All **4/4 runs scored Level 4 training wins (`39`, `10`, `93`, `36`)**, `hybrid_expected curriculum` achieved **`100.0%` Final AND BestCP (`+2185.0 R`)** at Stage 4 and **`100% / 100% / 100% / 100%`** through Stage 5 (`+2237.7 R`, `5.0` items), `tabular curriculum` reached **`100.0%` L4 (`+2215.7 R`)** by Stage 5, and `hybrid_expected separate` pushed `last500` median to **`1696.0`** (`+1620.2 R` eval right at Pit 4 fire `x = 1704.0`).
2. **Phase 11C Polish-Tail Anchor + Rollbacks Eliminated `tabular separate` Collapses Completely:**
   * Where Phase 10's `tabular separate` suffered catastrophic polish-tail stalls on L3 (`+21.0 R`), L4 (`+60.0 R`), and L5 (`+1.7 R`), Phase 11C triggered active rollbacks (`11` on L3, `34` on L4, `60` on L5), yielding **`100.0%` L3 Final (`+1321.4 R`)**, **`+1317.7 R` L4 Final** (`probe = 1517.7`), and **`+418.4 R` L5 Final** (`probe = 634.5`, matching BestCP `+418.4 R`).
   * *Note on Curriculum Stage 3/4 Eval vs. Stage 5 Retention:* During curriculum periodic evals, `stage_score` averages across all unlocked levels (`1..lvl`). When L3/L4 policy changes trade a slight dip on L1–L2 for L3–L4 breakthroughs (or L4 Pit 4 timing), end-of-Stage-3/4 snapshots can temporarily trail (`tab_curr` L3 `44%`, `tab_curr` L4 `0%` at `4475`, `hyb_curr` L3 `0%`), yet continued replay in Stages 4–5 drove **both curriculum runs to `100.0% / 100.0% / 100.0% / 100.0%` on L1–L4**.
3. **Level 5 Progress Beyond `Laser Sentry 1` (`x = 460`) and Remaining Bottlenecks (`> 1,200`):**
   * Eliminating airborne `Laser Sentry 1` flips unlocked clean Stage-5 greedy runs across `Base Ghost 1`, `Pit 1 (380..450)`, `Laser Sentry 1 (400..530)`, and `Obstacle 2 (580)`: `tabular separate` reached `635.6` (`Ground Maiden 1`), `hybrid_expected separate` reached `766.0` (`last500 med = 664.0 > 650`, `0/50` ghost & laser deaths in eval, stopping at `Pit 2` fire `x = 760..835`), and `tabular curriculum` BestCP crossed `Pit 2` cleanly to **`884.7` (`max = 902.7`, `+743.9 R`)** before **`Laser Sentry 2` (`840..960`, `49/50 laser_death`)**.
   * Reaching `> 1,200` on L5 now stalls at **`Pit 2` active hellfire (`760..835`) + `Laser Sentry 2` (`840..960`)** because very few late training episodes reach `x > 850` consistently (`med = 384–417`, `p90 = 426–523`).

---

## 7. Reproduction Commands

```bash
# Step 1: Full Unit Test Suite (85/85 OK)
python -m unittest discover -s tests -p "test_*.py" -v

# Step 3: Full 4-Run Balanced Benchmark (20,800 eps/run)
python train.py --agent all --mode all --reward balanced \
  --lambda 0.65 --lambda-min 0.35 --lambda-polish-zero \
  --polish-anchor-best --alpha-min 0.003 --eval-interval 25 \
  --rewarm-epsilon-hard 0.45 --rewarm-epsilon-l5 0.40 --max-steps-cap 2400 \
  --no-progress-bar
```
