# L4–L5 Convergence, Hazard-Safe Hybrid & Two-Phase Polish Report (Phase 10 vs. Phase 9)

**Spec:** `specs/PHASE_10_L4_L5_CONVERGENCE_AND_HYBRID_FIX.md.md`  
**Scope:** Airborne pit-distance disambiguation (`state_discretizer.py`), temporal hellfire prewarn cycle guard (`48`-tick / `50 px`, `68`-tick safety cap), laser `15 px` swap glitch filter (`28 px` imminent threshold), hazard-safe `Expected SARSA` gating + Watkins trace cuts (`HybridQSARSAAgent`), two-phase trace schedule + polish tail (`λ: 0.65 -> 0.35 -> 0.0`, `eps -> 0.005`, `alpha -> 0.003`), `18,800`-episode schedule (`800/1200/1800/7000/8000`), 4-tier curriculum rewarm (`0.30/0.35/0.45/0.40`), multi-level curriculum promotion + adaptive replay, and diagnostic logging (`best_checkpoint_episode`, `top_death_clusters`).

> Strict Section 1 boundaries honored: `env/`, `core/base_env.py`, `core/interfaces.py`, `rewards/dense_reward.py`, `rewards/sparse_reward.py`, and core `BalancedRewardStrategy` weights/penalties remained untouched. Exact 12-tuple `DiscreteState` shape and `state_space_size == 55296` (`enable_wisp_sensor = False`) / `110592` (`True`) preserved.

---

## 1. Stage 1 & Step 2 Verification Summary

* **Full Unit Test Suite:**
  ```bash
  python -m unittest discover -s tests -p "test_*.py" -v
  # Ran 80 tests in 1.302s — OK (0 failures, 0 skipped)
  ```
  All `74` prior tests + `6` new tests in `tests/test_phase10_convergence.py` passed after updating prior threshold assertions cleanly in `tests/test_pit_margin_and_trace_decay.py` and `tests/test_laser_motion_and_retention.py`.
* **100-Episode Balanced Smoke Test (`Step 2`):** Exited `0` across all `4` configurations into temporary directories (cleaned after verification), confirming `trace_decay_schedule = "two_phase_0.65_to_0.35_polish_to_0"`, `polish_epsilon_min = 0.005`, `alpha_min = 0.003`, `state_space_max = 55296`, `rewarm_epsilon_hard = 0.45`, `rewarm_epsilon_l5 = 0.4`, `best_checkpoint_episode`, and `top_death_clusters`.

---

## 2. Step 4 Acceptance Gate Scorecard (Phase 9 vs. Phase 10)

| Gate | Target Criterion | Phase 9 Result | Phase 10 Result | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **L1–L3** | `100.0%` Final AND `100.0%` BestCP across all 4 runs | L1–L2 `100/100` (4/4); L3 BestCP `100%` (3/4, `hyb_sep` `0%`); L3 Final `100%` (`tab_sep` only) | L1–L2 `100/100` (4/4); **L3 BestCP `100.0%` (4/4)**; **`hyb_sep` L3 `100.0% / 100.0%`**; Final `0.0%` on `tab_curr`, `hyb_curr`, `tab_sep` | **PARTIAL FAIL** (`12/12` BestCP, `9/12` Final) |
| **L4 (`tabular curr`)** | BestCP `>= 50%` & Final `>= 30%` | BestCP `26.0%` (`13/50`), Final `0.0%`, `9/5000` train wins, peak `2031.6` | BestCP `0.0%` (`+1113.5 R`, `50/50 laser_death`), Final `0.0%` (`+1061.5 R`), `0/7000` wins, peak `1835.2` (Note `tab_sep` BestCP hit `+1705.0 R`, `1826.0`) | **FAIL** |
| **L4 (`hybrid_expected`)** | Break `x = 456 / 936` pit/fire lock; `median max_x >= 1,200`, non-zero wins | `hyb_sep`: `50/50 fire` Final (`x=936`), `40/50 pit` BestCP (`med 458.0`, `+901.3 R`); `hyb_curr`: `med 432.8`, Final `+359.6 R` (`x=374`) | **`hyb_sep` broke Pit 1, Ghost 1–2, Fire Pit & Pit 3 (`0/50` pit/fire/ghost in eval)**: `last500 max_x med = 1245.2`, Final/BestCP `50/50` to Twin Lasers (`1258.4..1262.0`, `+1166.2 R`); `0` wins, full-stage train `med 454–456` | **PARTIAL** (Lock Broken, `0` Wins) |
| **L5** | Median `max_x > 650` and peak `max_x > 1,200` | Train med `378.0..416.0`, peak `811.0..895.6`; `hyb_sep` BestCP probe `622.9` | Train med `356.0..414.0`, **`hyb_curr` `last500 med = 482.0`** (`+428.3 R`), peak `936.0` (`tab_curr`) / `906.8` (`hyb_curr`); BestCP/Final probes `426.0..499.7` | **FAIL** |
| **Curriculum Retention** | `curriculum_best_checkpoint_stage >= 4` with `>= 95%` win rate on L1–L3 | Locked at **Stage `3`** on both `tab_curr` and `hyb_curr` (`100%/100%/100%` L1–L3; L4 `+274.3 / +313.4`) | **Stage `5`** on BOTH `tab_curr` & `hyb_curr` while retaining **`100.0% / 100.0% / 100.0%`** on L1–L3 (AND raising L4 retention to **`+1095.1 R` / `+778.5 R`**) | **PASS** (`2/2` Runs) |

---

## 3. Side-by-Side Per-Stage Comparison (Phase 9 vs. Phase 10, `50` Greedy Eval Episodes)

Phase 9 baseline: `{1: 800, 2: 1200, 3: 1800, 4: 5000, 5: 6000}` (`14,800` eps). Phase 10: `{1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 8000}` (`18,800` eps).

### 3.1 `tabular` (`curriculum` & `separate`)

| Mode | Lvl | P9 Final% / BestCP% (Best R) | P10 Final% / BestCP% (Final R / Best R) | P10 Final Steps / Best Steps / Items | P10 BestCP Ep | P10 Train Wins / MaxR50 / MaxProg | P10 States (Best / Final) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `curriculum` | L1 | `100.0%` / `100.0%` (`+955.5`) | **`100.0%`** / **`100.0%`** (`+955.6` / `+966.0`) | `200.0` / `214.0` / `0.0` | `425` | `787/800` / `100.0%` / `100.0%` | `27` / `27` |
| `curriculum` | L2 | `100.0%` / `100.0%` (`+1183.9`) | **`100.0%`** / **`100.0%`** (`+1075.6` / `+1056.9`) | `304.0` / `609.8` / `0.0` | `50` | `571/1200` / `94.0%` / `100.0%` | `58` / `63` |
| `curriculum` | L3 | `0.0%` / `100.0%` (`+1295.4`) | `0.0%` / **`100.0%`** (`+470.6` / **`+1324.1`**) | `N/A` / `351.0` / `0.0 / 1.0` | `1550` | `142/1800` / `50.0%` / `100.0%` | `160` / `161` |
| `curriculum` | L4 | `0.0%` / `26.0%` (`+1575.9`) | `0.0%` / `0.0%` (`+1061.5` / `+1113.5`) | `N/A` / `N/A` / `2.0` | `6550` | `0/7000` / `0.0%` / `88.54%` | `349` / `363` |
| `curriculum` | L5 | `0.0%` / `0.0%` (`+307.1`) | `0.0%` / `0.0%` (`+286.4` / `+300.7`) | `N/A` / `N/A` / `0.0 / 1.0` | `4650` | `0/8000` / `0.0%` / `37.13%` | `612` / `624` |
| `separate` | L1 | `100.0%` / `100.0%` (`+955.5`) | **`100.0%`** / **`100.0%`** (`+955.6` / `+966.0`) | `200.0` / `214.0` / `0.0` | `425` | `787/800` / `100.0%` / `100.0%` | `27` / `27` |
| `separate` | L2 | `100.0%` / `100.0%` (`+1138.7`) | **`100.0%`** / **`100.0%`** (`+1056.4` / `+1056.4`) | `316.0` / `316.0` / `0.0` | `550` | `525/1200` / `100.0%` / `100.0%` | `62` / `63` |
| `separate` | L3 | `100.0%` / `100.0%` (`+1287.0`) | `0.0%` / **`100.0%`** (`+21.0` / **`+1269.1`**) | `N/A` / `349.0` / `1.0` | `1675` | `40/1800` / `26.0%` / `100.0%` | `157` / `157` |
| `separate` | L4 | `0.0%` / `0.0%` (`+1154.1`) | `0.0%` / `0.0%` (`+60.0` / **`+1705.0`**) | `N/A` / `N/A` / `1.0 / 4.0` | `5425` | `0/7000` / `0.0%` / **`90.34%`** | `346` / `351` |
| `separate` | L5 | `0.0%` / `0.0%` (`+346.5`) | `0.0%` / `0.0%` (`+1.7` / `+343.9`) | `N/A` / `N/A` / `0.0 / 1.0` | `6900` | `0/8000` / `0.0%` / `33.80%` | `479` / `493` |

### 3.2 `hybrid_expected` (`curriculum` & `separate`)

| Mode | Lvl | P9 Final% / BestCP% (Best R) | P10 Final% / BestCP% (Final R / Best R) | P10 Final Steps / Best Steps / Items | P10 BestCP Ep | P10 Train Wins / MaxR50 / MaxProg | P10 States (Best / Final) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `curriculum` | L1 | `100.0%` / `100.0%` (`+946.8`) | **`100.0%`** / **`100.0%`** (`+907.8` / `+946.8`) | `212.0` / `212.0` / `0.0` | `25` | `788/800` / `100.0%` / `100.0%` | `25` / `26` |
| `curriculum` | L2 | `100.0%` / `100.0%` (`+1113.4`) | **`100.0%`** / **`100.0%`** (`+1113.6` / `+1113.6`) | `284.0` / `284.0` / `0.0` | `400` | `627/1200` / `100.0%` / `100.0%` | `58` / `59` |
| `curriculum` | L3 | `0.0%` / `100.0%` (`+1321.0`) | `0.0%` / **`100.0%`** (`+533.4` / **`+1246.2`**) | `N/A` / `398.0` / `0.0 / 2.0` | `725` | `153/1800` / `52.0%` / `100.0%` | `149` / `161` |
| `curriculum` | L4 | `0.0%` / `0.0%` (`+894.1`) | `0.0%` / `0.0%` (**`+940.7`** / `+750.5`) | `N/A` / `N/A` / `1.0` | `3700` | `0/7000` / `0.0%` / `82.73%` | `255` / `324` |
| `curriculum` | L5 | `0.0%` / `0.0%` (`+170.5`) | `0.0%` / `0.0%` (**`+344.3`** / **`+320.0`**) | `N/A` / `N/A` / `1.0` | `5325` | `0/8000` / `0.0%` / `35.91%` | `531` / `607` |
| `separate` | L1 | `100.0%` / `100.0%` (`+946.8`) | **`100.0%`** / **`100.0%`** (`+907.8` / `+946.8`) | `212.0` / `212.0` / `0.0` | `25` | `788/800` / `100.0%` / `100.0%` | `25` / `26` |
| `separate` | L2 | `100.0%` / `100.0%` (`+1109.0`) | **`100.0%`** / **`100.0%`** (`+1133.4` / `+1133.4`) | `266.0` / `266.0` / `0.0` | `450` | `572/1200` / `100.0%` / `100.0%` | `57` / `60` |
| `separate` | L3 | `0.0%` / `0.0%` (`+870.1`) | **`100.0%`** / **`100.0%`** (**`+1335.2`** / **`+1335.2`**) | `378.0` / `378.0` / `3.0` | `1200` | **`422/1800`** / **`80.0%`** / `100.0%` | `147` / `147` |
| `separate` | L4 | `0.0%` / `0.0%` (`+901.3`) | `0.0%` / `0.0%` (**`+1107.4`** / **`+1166.2`**) | `N/A` / `N/A` / `2.0 / 3.0` | `4300` | `0/7000` / `0.0%` / `63.60%` | `229` / `271` |
| `separate` | L5 | `0.0%` / `0.0%` (`+315.3`) | `0.0%` / `0.0%` (`+299.6` / **`+339.7`**) | `N/A` / `N/A` / `1.0` | `3850` | `0/8000` / `0.0%` / `31.94%` | `369` / `401` |

**Zero-Shot Curriculum Transfers (`50` eps):**
* `tabular curriculum`: `level_3_zero_shot`: `0.0%` (`+461.31 R`, `2.0` items, `ghost: 48, pit: 2` vs. Phase 9 `2.0%` / `+552.08 R`); `level_5_zero_shot`: `0.0%` (**`+297.43 R`**, `1.04` items, `ghost: 38, laser: 7, fire: 3, wisp: 2` vs. Phase 9 `+225.39 R`).
* `hybrid_expected curriculum`: `level_3_zero_shot`: `0.0%` (**`+551.10 R`**, `2.02` items, `ghost: 44, fire: 3, pit: 3` vs. Phase 9 `+477.66 R`); `level_5_zero_shot`: `0.0%` (**`+214.24 R`**, `0.84` items, `ghost: 41, pit: 9` vs. Phase 9 `+174.92 R`).

---

## 4. Curriculum Retention Matrices (`50` Greedy Eval Episodes Per Level)

| Agent | Retention Matrix | P9 Stage -> P10 Stage | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tabular`** | `post_curriculum_retention` | Final (L5) -> Final (L5) | **`100.0%`** (`+1002.9 R`, `235 st`) | **`100.0%`** (`+1187.0 R`, `302 st`) | `0.0%` (`+706.8 R`, `pit: 50`) | `0.0%` (`+283.1 R`, `pit: 50`) | `0.0%` (`+286.4 R`, `pit: 50`) |
| **`tabular`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+1006.3 R`, `201 st`) | **`100.0%`** (`+1188.6 R`, `300 st`) | **`100.0%`** (`+1338.1 R`, `349 st`) | `0.0%` (**`+1095.1 R`**, `laser: 50`) | `0.0%` (`+300.7 R`, `pit: 50`) |
| **`tabular`** | **`curriculum_best_retention`** | **Stage 3 -> Stage 5** | **`100.0%`** (`+1006.3 R`, `201 st`) | **`100.0%`** (`+1188.6 R`, `300 st`) | **`100.0%`** (`+1338.1 R`, `349 st`) | `0.0%` (**`+1095.1 R`**, `laser: 50`) | `0.0%` (**`+300.7 R`**, `pit: 50`) |
| **`hybrid_expected`** | `post_curriculum_retention` | Final (L5) -> Final (L5) | **`100.0%`** (`+960.8 R`, `210 st`) | `0.0%` (`-2229.0 R`, `timeout: 50`) | `0.0%` (`+333.8 R`, `ghost: 50`) | `0.0%` (`+340.3 R`, `fire: 50`) | `0.0%` (`+344.3 R`, `ghost: 50`) |
| **`hybrid_expected`** | `best_checkpoint_retention` | `q_table_best` | **`100.0%`** (`+1000.3 R`, `201 st`) | **`100.0%`** (`+1195.0 R`, `296 st`) | **`100.0%`** (`+1383.3 R`, `377 st`) | `0.0%` (**`+778.5 R`**, `fire: 50`) | `0.0%` (**`+320.0 R`**, `ghost: 50`) |
| **`hybrid_expected`** | **`curriculum_best_retention`** | **Stage 3 -> Stage 5** | **`100.0%`** (`+1000.3 R`, `201 st`) | **`100.0%`** (`+1195.0 R`, `296 st`) | **`100.0%`** (`+1383.3 R`, `377 st`) | `0.0%` (**`+778.5 R`**, `fire: 50`) | `0.0%` (**`+320.0 R`**, `ghost: 50`) |

Both curriculum runs promoted `curriculum_best_checkpoint_stage` from **Stage `3` (Phase 9) $\to$ Stage `5` (Phase 10)** while holding **`100.0% / 100.0% / 100.0%`** across L1–L3, and improving retained L4 reward from `+274.3 -> +1095.1` (`tabular`) and `+313.4 -> +778.5` (`hybrid_expected`). Notably, `tabular curriculum`'s final L5 agent (`post_curriculum_retention`) also jumped from `0.0%` to **`100.0%`** on L2!

---

## 5. Level 4 & Level 5 `outcome_counts`, `max_x_reached`, `top_death_clusters` & 10-Ep Checkpoint Probes

| Run | Source | Lvl | `win` | `ghost` | `fire` | `laser` | `wisp` | `pit` | `timeout` | `max_x` & Top Death Cluster Diagnostics |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `tab curr` | Final Eval (`50`) | 4 | 0 | **0** | **0** | 50 | 0 | **0** | 0 | `10`-ep probe `q_table_final`: `429.6` (`pit: 10`) |
| `tab curr` | Best CP (`50`) | 4 | 0 | **0** | **0** | 50 | 0 | **0** | 0 | `10`-ep `q_table_lvl4_best`: `1262.4` (`laser: 10`); `curr_best`: `1260.8` (`laser: 10`) |
| `tab curr` | Train (`7000`) | 4 | 0 | 1971 | 1886 | 456 | 0 | 2621 | 66 | `max_x` med/p90/max: `435.6 / 958.4 / 1835.2`; `last500` med **`996.8`** (`+785.6 R`); top clusters: `pit x=450 (94)`, `laser x=1260 (46)`, `fire x=950 (25)` |
| `tab curr` | Final / Best (`50`) | 5 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | `10`-ep `best`: `428.0`; `final`: `426.0` |
| `tab curr` | Train (`8000`) | 5 | 0 | 3815 | 37 | 1068 | 26 | 3054 | 0 | `max_x` med/p90/max: `392.0 / 450.4 / 936.0`; `last500` med `426.0` (`+262.4 R`); top clusters: `pit x=420 (285)`, `ghost x=380 (47)` |
| `tab sep` | Final Eval (`50`) | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | `10`-ep `q_table_lvl4_final`: `288.0` (`timeout: 10`, `+60.0 R` — polish collapse) |
| `tab sep` | Best CP (`50`) | 4 | 0 | 50 | **0** | **0** | 0 | **0** | 0 | **`10`-ep `q_table_lvl4_best`: `1826.0` (`+1705.0 R`, `4.0` items — cleared Twin Lasers!)** |
| `tab sep` | Train (`7000`) | 4 | 0 | 2730 | 1764 | 525 | 0 | 1905 | 76 | `max_x` med/p90/max: `428.0 / 1148.0 / 1835.2`; Phase 2 (`4000..5950`) `+779.6 R` -> Polish `+275.8 R` |
| `tab sep` | Final / Best (`50`) | 5 | 0 | 0 / 49 | 0 | 0 | 50 / 0 | 0 / 1 | 0 | `10`-ep `best`: `476.0 / 480.0 / 499.7` (`+339.3 R`); `final`: `181.2` (`wisp: 10`, `+1.7 R`) |
| `tab sep` | Train (`8000`) | 5 | 0 | 4418 | 9 | 2152 | 226 | 1195 | 0 | `max_x` med/p90/max: `356.0 / 426.0 / 856.4`; `last500` med `353.7` (`+206.6 R`) |
| `hyb curr` | Final / Best (`50`) | 4 | 0 | **0** | 0 / 50 | **0** | 0 | 50 / 0 | 0 | `10`-ep `best`: `898.0` (`fire: 10`); `curr_best`: `948.0`; `final`: `430.0` (yet 50-ep Final `+940.7 R`!) |
| `hyb curr` | Train (`7000`) | 4 | 0 | 2974 | 665 | 479 | 0 | 2849 | 33 | `max_x` med/p90/max: `454.0 / 1061.6 / 1684.8`; `last500` med **`1061.6`** (`+861.8 R`); top cluster: `pit x=1060 (192)`, `laser x=1230 (46)` |
| `hyb curr` | Final / Best (`50`) | 5 | 0 | 50 | **0** | **0** | **0** | **0** | 0 | `10`-ep `best`: `456.0` (`ghost: 10`); `final`: **`482.0`** (`ghost: 10`, `+344.3 R`, cleared Pit 1!) |
| `hyb curr` | Train (`8000`) | 5 | 0 | 3592 | 75 | 1122 | 38 | 3173 | 0 | `max_x` med/p90/max: `414.0 / 478.0 / 906.8`; `last500` med **`482.0`** (`+428.3 R`); top cluster: `ghost x=480 (221)` |
| `hyb sep` | Final / Best (`50`) | 4 | 0 | **0** | **0** | 50 | 0 | **0** | 0 | `10`-ep `best`: **`1262.0`** (`laser: 10`, `3.0` items); `final`: **`1258.4`** (`laser: 10`, `2.0` items) |
| `hyb sep` | Train (`7000`) | 4 | 0 | 2774 | 687 | 1532 | 0 | 2007 | 0 | `max_x` med/p90/max: `456.0 / 1241.2 / 1306.8`; **`last500` med `1245.2`** (`+1033.5 R`); top cluster: `laser x=1250 (202)` |
| `hyb sep` | Final / Best (`50`) | 5 | 0 | 0 / 50 | 0 | 0 | 0 | 50 / 0 | 0 | `10`-ep `best`: `480.0` (`ghost: 10`, `+339.7 R`); `final`: `426.0` (`pit: 10`) |
| `hyb sep` | Train (`8000`) | 5 | 0 | 3220 | 0 | 574 | 30 | 4176 | 0 | `max_x` med/p90/max: `408.4 / 426.0 / 812.0`; `last500` med `426.0` (`+283.1 R`) |

---

## 6. Root-Cause Diagnosis

1. **Why `hybrid_expected` Transformed L3 & L4 While `tabular separate` Collapsed in the Polish Tail:**
   * Across all polish tails (`85%–100%` of each stage, where `λ -> 0`, `eps -> 0.005`, `alpha -> 0.003`), **`hybrid_expected separate` monotonically gained performance**: L3 climbed `5.8% (ep 0..1200) -> 52.4% (1200..1530) -> 66.3% (1530..1800)` to **`100.0%` Final AND `100.0%` BestCP** (`+1335.2 R`, `422/1800` wins), and L4 climbed `+275.1 -> +778.4 -> +1003.1 R` (`last500 med max_x = 1245.2`, `0/50` pit/fire/ghost deaths). Hazard-gated Expected SARSA + Watkins trace cutting prevented rare late hazard deaths from corrupting optimal non-hazard Q-values.
   * Conversely, **pure `tabular separate` (`Q(λ)`) collapsed in the polish tail on L3, L4, and L5** (`L3 BestCP 100.0%` at ep `1675` $\to$ Final `+21.0 R` `timeout: 50`; `L4 BestCP +1705.0 R` / `1826.0` at ep `5425` $\to$ Polish `+275.8 R`, ep `6961..6989` timeouts at `x = 312..391` $\to$ Final `+60.0 R`; `L5 BestCP +343.9 R` at ep `6900` $\to$ Final `+1.7 R` `wisp_death`). Without hazard-gated smoothing, decaying `epsilon` down to `0.005` locked a single bad Ghost 1 / early encounter into a sub-idle stall (`q[idle] > q[right]`).
2. **Why L4 Twin Laser Scissors (`x = 1220..1360`, `laser_length = 95 px`) Blocked Phase 10 Wins:**
   * Both `tabular curriculum` (`50/50 laser_death` at `1260.8..1262.4`) and `hybrid_expected separate` (`50/50 laser_death` at `1258.4..1262.0`) reliably cleared Ghost 1, Pit 1 (`400..480`), Ghost 2, Pit 2 (`880..960`), and Pit 3 (`1020..1100`), arriving deterministically inside the dual counter-rotating `LaserGhost` crossfire (`env/map_levels.py`: `x=1200`, `1150..1360` and `x=1320`, `1220..1400`).
   * Because `laser_bin` encodes scalar `dist_to_laser` (`<= 28 px` imminent) and world-x motion without encoding **beam angle / active sweep direction or relative left/right offset**, greedy policies arriving when a `95 px` beam sweeps through ground level have no temporal/angular cue to pause outside `x = 1150` vs. sprint underneath.
3. **Why L5 Stayed Below `median max_x > 650` Despite Monotonic Late Gains:**
   * On L5 (`env/map_levels.py`), `Pit 1 (380..450)` immediately adjoins **`Laser Sentry 1` (`x = 460`, patrol `400..530`)**. Agents either clipped Pit 1 (`x_bin = 420`, `285–289` cluster deaths) or landed across Pit 1 directly into `Laser Sentry 1` / ground ghost (`x = 480..499.7`). However, `hybrid_expected curriculum`'s 1000-episode mean `max_x` climbed monotonically every window (`351.8 -> ... -> 424.0 -> 543.4`, `last500 med = 482.0`, Final probe `482.0`, `+344.3 R`), indicating `8000` episodes at `rewarm = 0.40` was still actively ascending at cutoff.
4. **Why L3 Curriculum Final Dropped Despite `100%` BestCP & `Stage 5` Retention:**
   * With `1800` total episodes and `replay_ratio = 0.35`, only `~1,170` L3 episodes executed while L1/L2 replays continued alongside `rewarm = 0.35`. Both curriculum runs still found **`100.0%` BestCP** (`ep 1550` / `ep 725`) and locked **Stage `5` `q_table_curriculum_best` at `100% / 100% / 100%`** on L1–L3, showing that saving/deploying `q_table_curriculum_best.pkl` & `q_table_lvl3_best.pkl` solves L3 convergence reliably, whereas single-point `ep 1800` final weights remain sensitive to noisy final updates.

---

## 7. Reproduction Commands

```bash
# Step 1: Full Unit Test Suite (80/80 OK)
python -m unittest discover -s tests -p "test_*.py" -v

# Step 3: Full 4-Run Balanced Benchmark (18,800 eps/run)
python train.py --agent all --mode all --reward balanced \
  --lambda 0.65 --lambda-min 0.35 --lambda-polish-zero \
  --alpha-min 0.003 --eval-interval 25 --rewarm-epsilon-hard 0.45 \
  --rewarm-epsilon-l5 0.40 --max-steps-cap 2400 --no-progress-bar
```
