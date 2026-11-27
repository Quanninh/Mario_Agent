# L4–L6 Integration, Order Comparison & L5 Convergence Report (Phase 12: Experiment A vs. Experiment B)

**Spec:** `specs/PHASE_12_L6_INTEGRATION_AND_L5_CONVERGENCE.md`  
**Scope:** Decoupled retention-safe `stage_best_q_table` vs. `curriculum_best_q_table` (`12A`), main-phase anti-collapse recovery guard (`12B`), uncapped `epsilon_decay <= 0.99998` targeting `0.015` at `85%` polish + scaled 6-level schedule `{1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 30000, 6: 3500}` (`44,300` eps/run) & `>10k` eval cadence scaling (`12C`), order-agnostic curriculum + `separate` per-level determinism + `QuyNhapTrang` threat sync `monster_threat_distance = 95.0` (`12D`), and `8` new unit tests (`tests/test_phase12_l6_and_l5_convergence.py`).

> Strict Section 1 boundaries honored: `env/`, `core/base_env.py`, `core/interfaces.py`, `rewards/dense_reward.py`, `rewards/sparse_reward.py`, `agent/tabular_q_agent.py`, and `agent/hybrid_q_sarsa_agent.py` received zero Phase 12 edits. Only `monster_threat_distance` changed (`75.0 -> 95.0`) in `discretizer/state_discretizer.py` and `rewards/balanced_reward.py`; the 12-tuple `DiscreteState` (`55,296` / `110,592` states) and all Phase 11 reward weights remained intact.

---

## 1. Unit Tests & Smoke Verification

* **Full Unit Test Suite:**
  ```bash
  python -m unittest discover -s tests -p "test_*.py" -v
  # Ran 93 tests in 4.153s — OK (0 failures, 0 skips)
  ```
  All `85` prior tests (updating `test_phase11_overhead_laser_and_polish.py` for the 6-level `DEFAULT_LEVEL_EPISODES` schedule) plus `8` new Phase 12 tests (`tests/test_phase12_l6_and_l5_convergence.py`) passed cleanly.
* **6-Level Smoke Tests (`Step 2`):**
  * **Smoke A (`--levels 1 2 3 4 6 5`, `--agent all --mode all`):** Exited `0`; verified `level_order == [1, 2, 3, 4, 6, 5]`, `monster_threat_distance == 95.0`, `main_collapse_guard == True`, `rewarm_epsilon_l6 == 0.4`, dynamic zero-shot keys (`level_3_zero_shot`, `level_6_zero_shot`, `level_5_zero_shot`), and ordered 6-level retention dicts.
  * **Smoke B (`--levels 1 2 3 4 5 6`, `--agent tabular --mode curriculum`):** Exited `0`; verified `level_order == [1, 2, 3, 4, 5, 6]`, zero-shot keys (`level_3_zero_shot`, `level_5_zero_shot`, `level_6_zero_shot`), and ordered 6-level retention dicts. Temporary `p12_smoke_*` folders were removed before full benchmarks.

---

## 2. Section 5 Acceptance Gate Scorecard (`Experiment A` & `Experiment B`)

| Acceptance Gate | Target Criterion | Observed Result (`Experiment A` + `Experiment B`) | Verdict |
| :--- | :--- | :--- | :---: |
| **L1–L4 Retention (Curriculum)** | Both `tabular` and `hybrid_expected` retain `100.0%` across L1–L4 in `curriculum_best_retention`. | **`hybrid_expected`** retained **`100.0% / 100.0% / 100.0% / 100.0%`** across L1–L4 in `curriculum_best_retention` (Stage 4) in **both Exp A and Exp B** (`+2054.3 R`, `677 st`). **`tabular`** retained **`100.0% / 100.0% / 100.0% / 0.0%`** (`probe_x = 1811.2`, `+1688.0 R`, ghost near goal). | **PARTIAL** (`1/2` agents `100%` L1–L4) |
| **Level 6 Mastery** | `>= 80%` Final AND BestCP on `>= 3/4` Exp A runs; `QuyNhapTrang` `monster_death < 10%`; zero L1–L4 forgetting after L6. | **`monster_death` passed on all runs:** **`0/50` (`0.0%`)** in **every** Final and BestCP eval across Exp A and Exp B (`0.14%–0.77%` in Exp A training; `24.8%–24.1%` during Exp B exploratory search before polish). **Eval win rate:** Exp A hit `0.0%` across all 4 runs (`4` train wins `hyb_curr`, `1` `tab_curr`), whereas **Exp B `hybrid_expected` (`L5 -> L6`) hit `100.0%` Final AND `100.0%` BestCP (`+1898.4 R`, `107` train wins)** (`tab_curr` `9` train wins, `0%` eval). L1–L4 forgetting occurred in `hyb_curr` during L6 (Exp A) and L5 (Exp B). | **PARTIAL** (`0%` eval in Exp A; **`100%` `hyb_curr` in Exp B**; `0/50` monster deaths) |
| **Level 5 Pit-1 Escape** | Zero runs stuck at `x = 424–426` (`pit_death: 50`) in BestCP eval; no Pit-1 anchor override. | **All 6/6 runs escaped Pit 1:** BestCP `probe_x` = `821.6`, `839.3`, `736.6`, **`1271.2`** (Exp A) and `836.0`, `640.0` (Exp B). Zero Pit-1-locked anchors. | **PASS** (`6/6` runs) |
| **Level 5 Deep Progress** | `>= 2/4` runs BestCP `probe_x > 1,000`; `>= 1/4` peak `max_x > 1,500` (or wins); `>= 2/4` `last500` median `max_x > 700`. | **`1/4` Exp A** reached `probe_x > 1,000` (`tabular separate`: **`1271.2`**, `BestCP @ ep 26050`) and `last500 med > 700` (**`817.7`**). Peaks reached `1284.4` (`tab_sep`), `1208.0` (`tab_curr`), `1184.0` (`ExpB hyb_curr`), `1023.6` (`hyb_sep`), `1012.0` (`ExpB tab_curr`), `887.3` (`hyb_curr`) (`< 1,500`, `0` L5 train wins). | **MISS / PARTIAL** (`1/4` >1k, `0/4` >1.5k) |
| **Order Comparison (`A` vs. `B`)** | Side-by-side comparison of L6 win rate, L5 progress, zero-shot transfer, and 6-level retention. | **L5 before L6 (`Exp B`) unlocked `100.0%` L6 mastery on `hybrid_expected` (`0% -> 100%`, `+1898.4 R`, `107` vs. `4` wins) and `9x` more L6 train wins on `tabular` (`9` vs. `1`), while `L6 before L5` (`Exp A`) produced stronger L5 exploration (`probe_x 839.3 / 821.6` vs. `836.0 / 640.0`, peaks `1208.0` vs. `1012.0`).** | **COMPLETE** |

---

## 3. Side-by-Side Results: Experiment A (`1->2->3->4->6->5`) vs. Experiment B (`1->2->3->4->5->6`)

Schedule across all runs: `L1: 800`, `L2: 1200`, `L3: 1800`, `L4: 7000`, `L5: 30000`, `L6: 3500` (`44,300` eps/run). Evaluation uses `50` greedy episodes per checkpoint.

### 3.1 Per-Stage Benchmark Summary (Levels 1–6)

| Run (`Exp / Agent / Mode`) | Lvl | Untr% (`Untr R`) | Final% (`Final R`) | BestCP% (`Best R` @ `BestEp`) | `BestProbeX` | Steps (`Fin/Best`) / Items | `Recov` / `Anc` / `Rollb` | Train Wins (`N`) / `Last500 MedX` / `PeakX` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ExpA / tabular / curr`** | L1 | `6.7%` (`+190.8`) | **`100.0%`** (`+925.6`) | **`100.0%`** (`+968.9` @ `325`) | `877.6` | `200.0/197.0` / `0.0` | `0` / `True` / `0` | `787/800` / `879.3` / `882.8` |
| **`ExpA / tabular / curr`** | L2 | `0.0%` (`+96.4`) | **`100.0%`** (`+1167.4`) | **`100.0%`** (`+1167.4` @ `275`) | `1078.8` | `284.0/284.0` / `0.0` | `0` / `True` / `0` | `574/1200` / `1078.8` / `1082.8` |
| **`ExpA / tabular / curr`** | L3 | `0.0%` (`+126.4`) | **`100.0%`** (`+1374.4`) | **`100.0%`** (`+1391.8` @ `1100`) | `1196.0` | `344.0/382.0` / `2.0/3.0` | `0` / `True` / `5` | `251/1800` / `1044.0` / `1202.4` |
| **`ExpA / tabular / curr`** | L4 | `0.0%` (`+111.3`) | `0.0%` (`+1688.0`) | `0.0%` (`+1688.0` @ `6675`) | `1811.2` | `N/A` / `3.0` | `4` / `True` / `22` | `0/7000` / `1240.2` / `1830.0` |
| **`ExpA / tabular / curr`** | L6 | `0.0%` (`+61.4`) | `0.0%` (`+921.3`) | `0.0%` (`+922.8` @ `2075`) | `1006.3` | `N/A` / `3.0` | `0` / `True` / `11` | **`1/3500`** / **`981.6`** / `1699.6` |
| **`ExpA / tabular / curr`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+731.7`) | `0.0%` (`+731.7` @ `15100`) | `839.3` | `N/A` / `2.0` | `4` / `True` / `175` | `0/30000` / `467.3` / `1208.0` |
| **`ExpA / hyb_exp / curr`** | L1 | `6.7%` (`+190.8`) | **`100.0%`** (`+890.3`) | **`100.0%`** (`+890.3` @ `50`) | `878.0` | `207.0/207.0` / `0.0` | `0` / `False` / `0` | `789/800` / `878.0` / `881.6` |
| **`ExpA / hyb_exp / curr`** | L2 | `0.0%` (`+96.4`) | **`100.0%`** (`+1153.1`) | **`100.0%`** (`+1153.1` @ `775`) | `1076.0` | `279.0/279.0` / `0.0` | `0` / `False` / `0` | `437/1200` / `1076.0` / `1080.8` |
| **`ExpA / hyb_exp / curr`** | L3 | `0.0%` (`+126.4`) | **`100.0%`** (`+1342.7`) | **`100.0%`** (`+1342.7` @ `1225`) | `1199.2` | `365.0/365.0` / `2.0` | `0` / `True` / `2` | `249/1800` / `698.0` / `1201.7` |
| **`ExpA / hyb_exp / curr`** | L4 | `0.0%` (`+111.3`) | **`100.0%`** (**`+2054.3`**) | **`100.0%`** (**`+2054.3`** @ `6950`) | **`2026.0`** | `677.0/677.0` / `4.0` | `4` / `True` / `32` | **`61/7000`** / **`1326.2`** / `2029.2` |
| **`ExpA / hyb_exp / curr`** | L6 | `0.0%` (`+61.4`) | `0.0%` (`+670.9`) | `0.0%` (`+670.9` @ `1500`) | `748.0` | `N/A` / `3.0` | `0` / `True` / `15` | **`4/3500`** / `608.0` / `1698.8` |
| **`ExpA / hyb_exp / curr`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+642.6`) | `0.0%` (`+642.6` @ `27425`) | `821.6` | `N/A` / `1.0` | `4` / `True` / `138` | `0/30000` / `522.8` / `887.3` |
| **`ExpA / tabular / sep`** | L1 | `6.7%` (`+190.8`) | **`100.0%`** (`+925.6`) | **`100.0%`** (`+968.9` @ `325`) | `877.6` | `200.0/197.0` / `0.0` | `0` / `True` / `0` | `787/800` / `879.3` / `882.8` |
| **`ExpA / tabular / sep`** | L2 | `0.0%` (`+96.4`) | **`100.0%`** (`+1109.0`) | **`100.0%`** (`+1109.0` @ `475`) | `1078.0` | `290.0/290.0` / `0.0` | `0` / `False` / `0` | `507/1200` / `1078.0` / `1078.8` |
| **`ExpA / tabular / sep`** | L3 | `0.0%` (`+126.4`) | **`100.0%`** (`+1267.3`) | **`100.0%`** (`+1267.3` @ `1300`) | `1196.0` | `367.0/367.0` / `1.0` | `0` / `True` / `10` | `175/1800` / `728.0` / `1199.9` |
| **`ExpA / tabular / sep`** | L4 | `0.0%` (`+111.3`) | `0.0%` (`+1154.4`) | `0.0%` (`+1158.6` @ `4450`) | `1287.4` | `N/A` / `3.0` | `4` / `True` / `41` | **`11/7000`** / `706.4` / `2029.1` |
| **`ExpA / tabular / sep`** | L6 | `0.0%` (`+61.4`) | `0.0%` (`+946.6`) | `0.0%` (`+946.6` @ `2500`) | `1064.0` | `N/A` / `2.0` | `0` / `True` / `1` | `0/3500` / **`964.0`** / `1068.0` |
| **`ExpA / tabular / sep`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+1118.4`) | `0.0%` (**`+1118.4`** @ `26050`) | **`1271.2`** | `N/A` / `2.0` | `4` / `True` / `85` | `0/30000` / **`817.7`** / **`1284.4`** |
| **`ExpA / hyb_exp / sep`** | L1 | `6.7%` (`+190.8`) | **`100.0%`** (`+890.3`) | **`100.0%`** (`+890.3` @ `50`) | `878.0` | `207.0/207.0` / `0.0` | `0` / `False` / `0` | `789/800` / `878.0` / `881.6` |
| **`ExpA / hyb_exp / sep`** | L2 | `0.0%` (`+96.4`) | **`100.0%`** (`+1125.9`) | **`100.0%`** (`+1125.9` @ `725`) | `1076.0` | `281.0/281.0` / `0.0` | `0` / `False` / `0` | `513/1200` / `1076.0` / `1078.8` |
| **`ExpA / hyb_exp / sep`** | L3 | `0.0%` (`+126.4`) | `40.0%` (`+1031.4`) | `40.0%` (`+1030.7` @ `800`) | `1153.8` | `340.7/342.4` / `1.4` | `0` / `True` / `11` | `112/1800` / `791.8` / `1200.0` |
| **`ExpA / hyb_exp / sep`** | L4 | `0.0%` (`+111.3`) | `0.0%` (`+827.5`) | `0.0%` (`+827.5` @ `4000`) | `936.0` | `N/A` / `2.0` | `4` / `True` / `42` | `0/7000` / `846.2` / `1316.8` |
| **`ExpA / hyb_exp / sep`** | L6 | `0.0%` (`+61.4`) | `0.0%` (`+790.0`) | `0.0%` (`+790.0` @ `3450`) | `888.0` | `N/A` / `3.0` | `0` / `True` / `19` | `0/3500` / `410.0` / `1346.0` |
| **`ExpA / hyb_exp / sep`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+599.9`) | `0.0%` (`+599.9` @ `26275`) | `736.6` | `N/A` / `2.0` | `4` / `True` / `146` | `0/30000` / `650.8` / `1023.6` |
| **`ExpB / tabular / curr`** | L1–L4 | *(identical to ExpA `tab_curr`)* | `100/100/100/0%` | `100/100/100/0%` (`L4 @ 6675`) | `L4: 1811.2` | `L4 items: 3.0` | `L4: 4/True/22` | `L4: 0/7000 / 1240.2 / 1830.0` |
| **`ExpB / tabular / curr`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+716.2`) | `0.0%` (`+716.2` @ `26100`) | `836.0` | `N/A` / `2.0` | `4` / `True` / `130` | `0/30000` / `444.8` / `1012.0` |
| **`ExpB / tabular / curr`** | L6 | `0.0%` (`+61.4`) | `0.0%` (`+671.1`) | `0.0%` (`+671.1` @ `3200`) | `751.0` | `N/A` / `3.0` | `0` / `True` / `11` | **`9/3500`** / `485.6` / `1699.6` |
| **`ExpB / hyb_exp / curr`** | L1–L4 | *(identical to ExpA `hyb_curr`)* | **`100/100/100/100%`** | **`100/100/100/100%`** (`L4 @ 6950`) | `L4: 2026.0` | `L4: 677.0 / 4.0` | `L4: 4/True/32` | `L4: 61/7000 / 1326.2 / 2029.2` |
| **`ExpB / hyb_exp / curr`** | L5 | `0.0%` (`+33.7`) | `0.0%` (`+486.8`) | `0.0%` (`+486.8` @ `28200`) | `640.0` | `N/A` / `2.0` | `0` / `True` / `169` | `0/30000` / `333.2` / `1184.0` |
| **`ExpB / hyb_exp / curr`** | L6 | `0.0%` (`+61.4`) | **`100.0%`** (**`+1898.4`**) | **`100.0%`** (**`+1898.4`** @ `3050`) | **`1697.2`** | **`448.0/448.0`** / **`4.0`** | `0` / `True` / `12` | **`107/3500`** / **`828.8`** (`98` wins) / `1702.8` |

### 3.2 Zero-Shot Curriculum Transfers (`50` Greedy Episodes)

| Experiment & Agent | Probe | Preceded By | Win% | Avg Reward | Avg `max_x` | Avg Items | Outcome Breakdown (`50` eps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ExpA / tabular` | `L3 ZeroShot` | L2 | `0.0%` | `+555.45` | `680.7` | `2.02` | `ghost: 46, fire: 3, pit: 1` |
| `ExpA / tabular` | `L6 ZeroShot` | L4 | `0.0%` | `+216.96` | `348.6` | `1.26` | `ghost: 27, pit: 14, monster: 9` |
| `ExpA / tabular` | `L5 ZeroShot` | L6 | `0.0%` | `+177.52` | `295.1` | `1.00` | `ghost: 50` |
| `ExpA / hyb_exp` | `L3 ZeroShot` | L2 | `0.0%` | `+523.50` | `674.0` | `1.10` | `ghost: 46, fire: 2, pit: 2` |
| `ExpA / hyb_exp` | `L6 ZeroShot` | L4 | `0.0%` | `+231.19` | `383.4` | `1.78` | `pit: 32, ghost: 11, monster: 7` |
| `ExpA / hyb_exp` | `L5 ZeroShot` | L6 | `0.0%` | `+203.08` | `314.1` | `1.00` | `ghost: 50` |
| `ExpB / tabular` | `L5 ZeroShot` | L4 | `0.0%` | `+169.71` | `325.7` | `0.92` | `ghost: 27, pit: 12, wisp: 8, laser: 3` |
| `ExpB / tabular` | `L6 ZeroShot` | L5 | `0.0%` | `+149.50` | `283.2` | `0.00` | `ghost: 50` |
| `ExpB / hyb_exp` | `L5 ZeroShot` | L4 | `0.0%` | `+172.71` | `330.2` | `0.92` | `ghost: 34, pit: 9, laser: 7` |
| **`ExpB / hyb_exp`** | **`L6 ZeroShot`** | **L5** | `0.0%` | **`+282.78`** | **`420.4`** | `1.48` | **`ghost: 0, fire: 0, laser: 0`**, `monster: 35, pit: 15` |

> **Order Transfer Takeaway (`L5 -> L6` vs. `L4 -> L6`):** Pre-training on 30,000 episodes of Level 5 lifted `hybrid_expected`'s zero-shot L6 opening (`0` early ghost deaths, `max_x = 420.4`, `+282.78 R`) and unlocked **`100.0%` Final AND BestCP mastery (`107` wins, `+1898.4 R`)**, whereas `L4 -> L6` in Experiment A only reached `4` training wins and `0.0%` eval (`probe_x = 748.0`). Conversely, training L6 first in Experiment A gave higher subsequent L5 exploration peaks (`tab_curr`: `1208.0` vs. `1012.0`; `hyb_curr`: `probe_x = 821.6` vs. `640.0`).

---

## 4. 6-Level Curriculum Retention Matrices (`50` Greedy Episodes / Level)

| Exp & Agent | Retention Summary | Source Stage / File | L1 | L2 | L3 | L4 | L6 (order pos) | L5 (order pos) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ExpA / tab`** | `post_curriculum` | Final (`L5`) | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`662.8`) | `0.0%` (`462.0`) | `0.0%` (`839.3`) |
| **`ExpA / tab`** | `best_checkpoint` | `q_table_best` | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`1746.8`) | `0.0%` (`406.0`) | `0.0%` (`856.0`) |
| **`ExpA / tab`** | **`curriculum_best`** | **Stage 4** | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`1811.2`) | `0.0%` (`354.9`) | `0.0%` (`310.4`) |
| **`ExpA / hyb`** | `post_curriculum` | Final (`L5`) | **`100.0%`** | `0.0%` (`528.8`) | `0.0%` (`1036.0`) | `0.0%` (`408.0`) | `0.0%` (`287.2`) | `0.0%` (`821.6`) |
| **`ExpA / hyb`** | `best_checkpoint` | `q_table_best` | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`1615.0`) | `0.0%` (`257.2`) | `0.0%` (`289.2`) |
| **`ExpA / hyb`** | **`curriculum_best`** | **Stage 4** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** (`2026.0`) | `0.0%` (`393.3`) | `0.0%` (`326.6`) |
| **`ExpB / tab`** | `post_curriculum` | Final (`L6`) | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`464.4`) | `0.0%` (`751.0`) | `0.0%` (`303.2`) |
| **`ExpB / tab`** | `best_checkpoint` | `q_table_best` | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`1220.4`) | `0.0%` (`309.2`) | `0.0%` (`421.2`) |
| **`ExpB / tab`** | **`curriculum_best`** | **Stage 4** | **`100.0%`** | **`100.0%`** | **`100.0%`** | `0.0%` (`1811.2`) | `0.0%` (`351.1`) | `0.0%` (`318.5`) |
| **`ExpB / hyb`** | `post_curriculum` | Final (`L6`) | **`100.0%`** | **`100.0%`** | `0.0%` (`736.8`) | `0.0%` (`385.6`) | **`100.0%`** (`1697.2`) | `0.0%` (`324.8`) |
| **`ExpB / hyb`** | `best_checkpoint` | `q_table_best` | **`100.0%`** | **`100.0%`** | `0.0%` (`1078.1`) | `0.0%` (`405.6`) | **`100.0%`** (`1700.2`) | `0.0%` (`351.3`) |
| **`ExpB / hyb`** | **`curriculum_best`** | **Stage 4** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** (`2026.0`) | `0.0%` (`381.8`) | `0.0%` (`322.6`) |

* **Checkpoint Diagnostics Across Stages:** On `hybrid_expected curriculum`, `q_table_lvl4_best.pkl` (`282` states) retained `{1: 100%, 2: 100%, 3: 100%, 4: 100%}` in both runs. However, entering either L6 (`Exp A`: `q_table_lvl6_best.pkl`, `392` states -> `{1: 100%, 2: 0%, 3: 0%, 4: 0%}`) or L5 (`Exp B`: `q_table_lvl5_best.pkl`, `576` states -> `{1: 100%, 2: 0%, 3: 0%, 4: 0%}`) wiped L2–L4 before any `(1, ...)` anchor survived, so carryover passed degraded weights onward while `curriculum_best` remained safely locked at Stage 4 (`_should_promote_curriculum_best` non-regression guard blocked overwriting Stage 4). Meanwhile, `tabular curriculum` preserved **`100% / 100% / 100%` on L1–L3** through every checkpoint (`Stage 4`, `Stage 5`, `Stage 6`, and `final`) in **both** experiments.

---

## 5. Level 4, 5 & 6 Outcome Breakdown & Top Death Clusters

| Run | Source | Lvl | `win` | `ghost` | `fire` | `laser` | `wisp` | `mon` | `pit` | `timeout` | Top Death Clusters / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ExpA tab curr` | BestCP Eval (`50`) | 4 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | `probe_x = 1811.2`; Ghost near L4 goal (`last500` train laser cluster `1240 (124)`) |
| `ExpA hyb curr` | BestCP Eval (`50`) | 4 | **50** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `probe_x = 2026.0`, `61` train wins (`ep 6552`, `last500 med_x = 1326.2`) |
| `ExpA tab curr` | BestCP Eval (`50`) | 6 | 0 | 17 | 0 | 33 | 0 | **0** | 0 | 0 | `probe_x = 1006.3`; Laser Sentry at `x=960` (`last500 laser 1010 (105)`), `1` train win |
| `ExpA hyb curr` | BestCP Eval (`50`) | 6 | 0 | 0 | 0 | 0 | 0 | **0** | 50 | 0 | `probe_x = 748.0`; Pit 2 (`700..775`), `4` train wins (`peak 1698.8`) |
| `ExpA tab sep` | BestCP Eval (`50`) | 6 | 0 | 0 | 0 | 50 | 0 | **0** | 0 | 0 | `probe_x = 1064.0`; Laser Sentry edge / Pit 3 (`last500 laser 1060 (219)`) |
| `ExpA hyb sep` | BestCP Eval (`50`) | 6 | 0 | 0 | 0 | 50 | 0 | **0** | 0 | 0 | `probe_x = 888.0`; Laser Sentry approach (`last500 pit 380 (99)`, `laser 880 (35)`) |
| `ExpB tab curr` | BestCP Eval (`50`) | 6 | 0 | 0 | 0 | 0 | 0 | **0** | 50 | 0 | `probe_x = 751.0`; Pit 2 (`700..775`), `9` train wins (`peak 1699.6`, `wisp 470 (85)`) |
| **`ExpB hyb curr`** | **BestCP Eval (`50`)** | **6** | **50** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **`probe_x = 1697.2` (`+1898.4 R`, `448 st`, `4 it`), `107` train wins (`98` in `last500`)** |
| `ExpA tab sep` | BestCP Eval (`50`) | 5 | 0 | 0 | 0 | **50** | 0 | 0 | 0 | 0 | **`probe_x = 1271.2`** (`+1118.4 R`), cleared Pit 2–3, stopped at Laser Sentry 3 (`1270–1280`) |
| `ExpA tab curr` | BestCP Eval (`50`) | 5 | 0 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | `probe_x = 839.3`; Wisp hit past Pit 2 (`830`), train `peak = 1208.0` (`@ep 21261`) |
| `ExpA hyb curr` | BestCP Eval (`50`) | 5 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `probe_x = 821.6`; Pit 2 active hellfire (`810`), `recov = 4`, `rollb = 138` |
| `ExpA hyb sep` | BestCP Eval (`50`) | 5 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `probe_x = 736.6`; Pit 2 hellfire approach (`730`), train `peak = 1023.6`, `rollb = 146` |
| `ExpB tab curr` | BestCP Eval (`50`) | 5 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | `probe_x = 836.0`; Pit 2 hellfire (`820–830`), train `peak = 1012.0`, `rollb = 130` |
| `ExpB hyb curr` | BestCP Eval (`50`) | 5 | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | `probe_x = 640.0`; `recov = 0` (`<400 R` gate), `rollb = 169`, train `peak = 1184.0` |

---

## 6. Root-Cause Findings & Bottlenecks Identified for Phase 13

1. **Polish Rollback Thrashing (`rollb = 85–175` of `180` polish checks on L5; `22–42` of `42` on L4):**
   * On `0%`-win L5 runs, periodic 10-episode eval variance causes frequent dips `< 0.75 * best_avg_reward`, triggering `_handle_mid_polish_regression` (`training/trainer.py`) on `47%–97%` of all L5 polish evaluations (`175/180` `ExpA tab_curr`, `169/180` `ExpB hyb_curr`, `146/180` `hyb_sep`, `138/180` `ExpA hyb_curr`, `130/180` `ExpB tab_curr`, `85/180` `tab_sep`). Each rollback resets weights and halves `alpha` until it pins at `0.0015`, freezing late polish improvement while rolling back to mid-run snapshots (`ep 15100`).
2. **Zero-Warning `LaserGhost` Rotational Sweeps (`dist_to_laser: -1.0 -> 0.0` in 1 frame):**
   * `env/game_env.py` only populates `dist_to_laser >= 0` when the beam's Y-box overlaps `[player.y, player.bottom]`. Rollout traces of BestCP L6 (`hyb_sep @ 884`, `tab_curr @ 1004`, `tab_sep @ 1060`) and L5 (`tab_sep @ 1271`) show `laser_bin == 0` (`-1.0`) right up to `0.0` on the frame the rotating arm sweeps through Hoá while crouching underneath or leaping past the spinner.
3. **`hybrid_expected` Early L2–L4 Forgetting in Post-Stage-4 Transitions:**
   * Entering either L6 (`Exp A`) or L5 (`Exp B`) with `rewarm = 0.40`, `lambda = 0.65`, and Expected SARSA updates erased L2–L4 (`100% -> 0%`) inside early training before any `retention_ok == 1` stage checkpoint was recorded, so subsequent stages inherited degraded prior weights.
4. **`MainPhaseCollapseGuard` Early Cap Exhaustion & `>= 400 R` Gate Miss:**
   * All runs that triggered the guard hit the 4-recovery limit early (`recov = 4` across all L4 and 5/6 L5 stages), disabling protection for the remainder of `25%–85%` (`>15,000` episodes on L5), while `ExpB hyb_curr` L5 never crossed `400.0 R` before polish (`recov = 0`, stalling at `x = 333.2` `last500`).

---

## 7. Reproduction Commands

```bash
# Unit Tests (93/93 OK)
python -m unittest discover -s tests -p "test_*.py" -v

# Experiment A (L6 before L5: 1 -> 2 -> 3 -> 4 -> 6 -> 5, all 4 runs)
python train.py --agent all --mode all --reward balanced --levels 1 2 3 4 6 5 --lambda 0.65 --lambda-min 0.35 --lambda-polish-zero --polish-anchor-best --alpha-min 0.003 --eval-interval 25 --rewarm-epsilon-hard 0.45 --rewarm-epsilon-l6 0.40 --rewarm-epsilon-l5 0.40 --max-steps-cap 2400 --outputs-dir outputs/p12_l6_first --checkpoints-dir checkpoints/p12_l6_first --no-progress-bar

# Experiment B (Sequential: 1 -> 2 -> 3 -> 4 -> 5 -> 6, curriculum)
python train.py --agent all --mode curriculum --reward balanced --levels 1 2 3 4 5 6 --lambda 0.65 --lambda-min 0.35 --lambda-polish-zero --polish-anchor-best --alpha-min 0.003 --eval-interval 25 --rewarm-epsilon-hard 0.45 --rewarm-epsilon-l5 0.40 --rewarm-epsilon-l6 0.40 --max-steps-cap 2400 --outputs-dir outputs/p12_seq --checkpoints-dir checkpoints/p12_seq --no-progress-bar
```
