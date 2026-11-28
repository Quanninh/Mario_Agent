# Section 7 Evaluation Plots & Markdown Tables Implementation Plan

> **For Executing Agent:** Follow this plan task-by-task using strict Test-Driven Development (RED → GREEN → REFACTOR/VERIFY). Do NOT modify training scripts, checkpoints, or `evaluation/*/data/*.csv` files.

**Goal:** Replace all legacy evaluation figure generators in `evaluation/generate_curriculum_evaluation.py` with the 4 Section 7 paper figures (`fig1`–`fig4`) across Levels 1–7, clean stale figure files from `evaluation/` and `springer_lncs/figures/`, and generate `evaluation/paper_evaluation_tables.md` containing clearly labeled `[PLACEHOLDER]` template tables for Subsections 7.1–7.5.

**Architecture:** Single-module Matplotlib evaluation pipeline (`evaluation/generate_curriculum_evaluation.py`) reading existing split CSVs from `evaluation/tabular/data/` and `evaluation/hybrid_expected/data/`, using shared stage-divider, multi-seed aggregation, and track-aware pre-training step helpers (`Track A: L1–L5`, `Track B: L6–L7`), plus a pure-Python Markdown template generator for the upcoming 100-seed deterministic checkpoint evaluation.

**Tech Stack:** Python 3.13, `numpy`, `pandas`, `matplotlib`, `unittest` (standard library).

---

## Files Touched

- `tests/test_evaluation_plots.py` *(New)* — Unit tests using temporary synthetic CSV fixtures for track accounting, Figs. 1–4 generation, stale-figure cleanup, and Markdown table template output.
- `evaluation/generate_curriculum_evaluation.py` *(Modified)* — Replace legacy figure functions (`plot_figure_1_win_rate`, `plot_figure_2_reward_trajectories`, `plot_figure_3_sample_efficiency`, `plot_per_level_comparisons`, `plot_cross_arch_sample_efficiency`) and legacy summary generation with `fig1`–`fig4`, `cleanup_stale_figures`, and `generate_paper_evaluation_tables`.
- `evaluation/fig1_reward_trajectories.{pdf,png}` *(Generated)*
- `evaluation/fig2_balanced_overview.{pdf,png}` *(Generated)*
- `evaluation/fig3_spotlight_l3_l4_l6.{pdf,png}` *(Generated)*
- `evaluation/fig4_sample_efficiency.{pdf,png}` *(Generated)*
- `evaluation/paper_evaluation_tables.md` *(Generated)*
- `springer_lncs/figures/fig1_reward_trajectories.pdf` ... `fig4_sample_efficiency.pdf` *(Synced; stale figures removed)*

---

## Shared Test Fixture Scaffolding (Create Once in Task 1)

All unit tests in `tests/test_evaluation_plots.py` share an in-memory/temp-directory CSV builder that produces realistic 7-level datasets (`curriculum` & `separate` × `balanced`, `dense`, `sparse` × `2` seeds) without touching real `evaluation/*/data/` files.

---

### Task 1: Update `LEVEL_IDS` to 1–7, Add Track-Aware Pre-Training Helper & Shared Plot Helpers

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Create `tests/test_evaluation_plots.py` with synthetic store helper `_build_synthetic_store()` and `test_track_pretraining_and_levels`:

```python
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation.generate_curriculum_evaluation as gce


def _make_level_df(level_id: int, n_eps: int = 120, win_after: int = 40, steps_per_ep: int = 100, start_global_steps: int = 0) -> pd.DataFrame:
    rows = []
    cum_lvl = 0
    cum_glob = start_global_steps
    for ep in range(1, n_eps + 1):
        won = ep >= win_after
        cum_lvl += steps_per_ep
        cum_glob += steps_per_ep
        rows.append({
            "global_episode": (level_id - 1) * n_eps + ep,
            "level_episode": ep,
            "level_id": level_id,
            "total_reward": 200.0 if won else -20.0,
            "steps": steps_per_ep,
            "max_x_reached": 900.0 if won else 300.0,
            "progress_pct": 100.0 if won else 35.0,
            "obstacle_hits": 0 if won else 3,
            "items_collected": 1 if won else 0,
            "items_total": 1,
            "epsilon": max(0.01, 1.0 - ep / n_eps),
            "won": won,
            "outcome": "win" if won else "timeout",
            "rolling_win_rate_100": 100.0 if won else 0.0,
            "global_cumulative_steps": cum_glob,
            "level_cumulative_steps": cum_lvl,
        })
    return pd.DataFrame(rows)


def _build_synthetic_store() -> dict:
    store = {}
    for mode in ("curriculum", "separate"):
        for strat in ("balanced", "dense", "sparse"):
            for seed in ("seed42", "seed43"):
                track_a_steps = 0
                track_b_steps = 0
                for lvl in range(1, 8):
                    win_after = 30 if (strat == "balanced" and lvl in (1, 2, 3, 6)) else 9999
                    if mode == "curriculum":
                        base = track_a_steps if lvl <= 5 else track_b_steps
                    else:
                        base = 0
                    df = _make_level_df(lvl, n_eps=120, win_after=win_after, steps_per_ep=100, start_global_steps=base)
                    store[(mode, strat, lvl, seed)] = df
                    if mode == "curriculum":
                        if lvl <= 5:
                            track_a_steps += int(df["steps"].sum())
                        else:
                            track_b_steps += int(df["steps"].sum())
    return store


class TestEvaluationPlots(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_eval_plots_"))
        self.tab_store = _build_synthetic_store()
        self.hyb_store = _build_synthetic_store()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_track_pretraining_and_levels(self):
        self.assertEqual(gce.LEVEL_IDS, [1, 2, 3, 4, 5, 6, 7])
        # Track A: L1 has 0 pre-training, L2 has L1 steps (12,000)
        self.assertEqual(gce.get_track_pretraining_steps(self.tab_store, "curriculum", "balanced", 1, "seed42"), 0.0)
        self.assertEqual(gce.get_track_pretraining_steps(self.tab_store, "curriculum", "balanced", 2, "seed42"), 12000.0)
        # Track B: L6 resets to 0 pre-training, L7 has L6 steps (12,000)
        self.assertEqual(gce.get_track_pretraining_steps(self.tab_store, "curriculum", "balanced", 6, "seed42"), 0.0)
        self.assertEqual(gce.get_track_pretraining_steps(self.tab_store, "curriculum", "balanced", 7, "seed42"), 12000.0)
        # Separate mode always has 0 pre-training
        self.assertEqual(gce.get_track_pretraining_steps(self.tab_store, "separate", "balanced", 3, "seed42"), 0.0)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_track_pretraining_and_levels`
- Expected result: `FAIL` (`AssertionError: [1, 2, 3, 4, 5, 6, 7, 8] != [1, 2, 3, 4, 5, 6, 7]` and `AttributeError: module ... has no attribute 'get_track_pretraining_steps'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`:
  1. Set `LEVEL_IDS = [1, 2, 3, 4, 5, 6, 7]` and `TRACKS = {"A": [1, 2, 3, 4, 5], "B": [6, 7]}`.
  2. Add `get_track_pretraining_steps(data_store, mode, strategy, level_id, seed) -> float`:
     ```python
     def get_track_pretraining_steps(
         data_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
         mode: str,
         strategy: str,
         level_id: int,
         seed: str,
     ) -> float:
         """Return cumulative steps from earlier levels within the same curriculum track."""
         if mode != "curriculum":
             return 0.0
         track_levels = TRACKS["A"] if level_id <= 5 else TRACKS["B"]
         total = 0.0
         for prev_lvl in track_levels:
             if prev_lvl >= level_id:
                 break
             df_prev = data_store.get((mode, strategy, prev_lvl, seed))
             if df_prev is not None and len(df_prev) > 0:
                 total += float(df_prev["steps"].sum())
         return total
     ```
  3. Add shared helpers `_draw_stage_dividers(ax, n_eps_per_level: int, y_label_pos: float, show_labels: bool = True)` (draws dashed lines at `k * n_eps_per_level`, a bold Track A | Track B separator at `5 * n_eps_per_level`, and centered `L1`–`L7` tags) and `_save_figure(fig, save_pdf: Path, save_png: Path, label: str)`.

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_track_pretraining_and_levels`
- Expected result: `OK` (1 test passes).

**Step 4: Commit Checkpoint**
- `refactor(eval): restrict LEVEL_IDS to 1..7 and add track-aware pretraining helper`

---

### Task 2: Implement Figure 1 (`plot_fig1_reward_trajectories`) — Section 7.2

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Add `test_plot_fig1_reward_trajectories` to `TestEvaluationPlots`:

```python
    def test_plot_fig1_reward_trajectories(self):
        pdf = self.tmp_dir / "fig1_reward_trajectories.pdf"
        png = self.tmp_dir / "fig1_reward_trajectories.png"
        gce.plot_fig1_reward_trajectories(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig1_reward_trajectories`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'plot_fig1_reward_trajectories'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`, replace `plot_figure_2_reward_trajectories` with `plot_fig1_reward_trajectories(tabular_store, hybrid_store, save_pdf, save_png)`:
  - `fig, axes = plt.subplots(3, 2, figsize=(14.5, 9.8), sharex=True)`
  - Rows: `["sparse", "balanced", "dense"]`; Columns: `[("curriculum", "Curriculum Transfer"), ("separate", "Separate Training")]`.
  - In every cell `(r, c)`, call `get_multi_seed_trajectories(store, mode, strat)` for both `tabular` (`ARCH_COLORS["tabular"]`) and `hybrid_expected` (`ARCH_COLORS["hybrid_expected"]`) across `LEVEL_IDS = [1..7]`, plotting `reward_mean` with `± 1 std` `fill_between`.
  - Compute shared native `y_min, y_max` across both columns of each row (8% span padding) so Sparse collapse (~0 after L1) and Dense shaping plateaus remain undistorted.
  - Draw `_draw_stage_dividers` (`L1`–`L7` + bold `Track A | Track B` vertical dashed divider at `5 * n_eps_per_level`) on each subplot; add stage labels on top row (`r == 0`).
  - Save via `_save_figure(fig, save_pdf, save_png, "Fig. 1 (Reward Trajectories)")`.

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig1_reward_trajectories`
- Expected result: `OK`.

**Step 4: Commit Checkpoint**
- `feat(eval): implement Section 7.2 Fig. 1 cross-architecture reward trajectories`

---

### Task 3: Implement Figure 2 (`plot_fig2_balanced_overview`) — Section 7.3

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Add `test_plot_fig2_balanced_overview` to `TestEvaluationPlots`:

```python
    def test_plot_fig2_balanced_overview(self):
        pdf = self.tmp_dir / "fig2_balanced_overview.pdf"
        png = self.tmp_dir / "fig2_balanced_overview.png"
        gce.plot_fig2_balanced_overview(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig2_balanced_overview`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'plot_fig2_balanced_overview'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`, replace `plot_figure_1_win_rate` with `plot_fig2_balanced_overview(tabular_store, hybrid_store, save_pdf, save_png)`:
  - `fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True, sharey=True)`
  - Hold reward fixed to `"balanced"`.
  - For `(mode, ax)` in `[("curriculum", ax_c), ("separate", ax_s)]`:
    - Create `ax_twin = ax.twinx()` for $\epsilon$ exploration schedule (`-0.02` to `1.08`).
    - Plot `Tabular Q-Learning` and `Hybrid Expected SARSA` rolling win rate (`%`, `W=100`, `± 1 std` ribbons) using `get_multi_seed_trajectories(store, mode, "balanced")`.
    - Plot horizontal dashed `80%` milestone line (`color="#444444", linestyle=":", linewidth=1.1`).
    - Draw `_draw_stage_dividers` (`L1`–`L7` stage boundaries + bold `Track A | Track B` divider).
  - Save via `_save_figure(fig, save_pdf, save_png, "Fig. 2 (Balanced Overview L1-L7)")`.

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig2_balanced_overview`
- Expected result: `OK`.

**Step 4: Commit Checkpoint**
- `feat(eval): implement Section 7.3 Fig. 2 Balanced L1-L7 overview curve`

---

### Task 4: Implement Figure 3 (`plot_fig3_spotlight_l3_l4_l6`) — Section 7.3

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Add `test_plot_fig3_spotlight_l3_l4_l6` to `TestEvaluationPlots`:

```python
    def test_plot_fig3_spotlight_l3_l4_l6(self):
        pdf = self.tmp_dir / "fig3_spotlight_l3_l4_l6.pdf"
        png = self.tmp_dir / "fig3_spotlight_l3_l4_l6.png"
        gce.plot_fig3_spotlight_l3_l4_l6(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig3_spotlight_l3_l4_l6`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'plot_fig3_spotlight_l3_l4_l6'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`, replace `plot_per_level_comparisons` with `plot_fig3_spotlight_l3_l4_l6(tabular_store, hybrid_store, save_pdf, save_png)`:
  - `spotlight_levels = [(3, "Level 3: Whispering Graves"), (4, "Level 4: Nightmare Catacombs"), (6, "Level 6: Awakening of the Corpse Demon")]`
  - `fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6), sharex="col", sharey=True)`
  - Rows: `[("curriculum", "Curriculum Transfer"), ("separate", "Separate Training")]`; Columns: `L3, L4, L6`.
  - Hold reward fixed to `"balanced"`. For each `(row_idx, (mode, mode_label))` and `(col_idx, (lvl, lvl_title))`:
    - Call `get_level_multi_seed_trajectory(store, mode, "balanced", lvl, wr_window=100, rew_window=100)` for `tabular` and `hybrid_expected`.
    - Plot rolling win rate (`± 1 std`), `80%` threshold dashed line, and $\epsilon$ on `ax.twinx()` (display right Y-axis label only on `col_idx == 2` to avoid clutter; hide tick labels on `col_idx < 2`).
    - Set Y-limit `(-3, 105)`, add concise row label on `col_idx == 0`, column title on `row_idx == 0`, X-label `"Within-Level Episode"` on `row_idx == 1`.
  - Save via `_save_figure(fig, save_pdf, save_png, "Fig. 3 (Spotlight L3, L4, L6)")`.

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig3_spotlight_l3_l4_l6`
- Expected result: `OK`.

**Step 4: Commit Checkpoint**
- `feat(eval): implement Section 7.3 Fig. 3 spotlight grid for Levels 3, 4, and 6`

---

### Task 5: Implement Figure 4 (`plot_fig4_sample_efficiency`) — Section 7.4

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Add `test_plot_fig4_sample_efficiency` to `TestEvaluationPlots`:

```python
    def test_plot_fig4_sample_efficiency(self):
        pdf = self.tmp_dir / "fig4_sample_efficiency.pdf"
        png = self.tmp_dir / "fig4_sample_efficiency.png"
        gce.plot_fig4_sample_efficiency(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig4_sample_efficiency`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'plot_fig4_sample_efficiency'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`, replace `plot_figure_3_sample_efficiency` and `plot_cross_arch_sample_efficiency` with `plot_fig4_sample_efficiency(tabular_store, hybrid_store, save_pdf, save_png)`:
  - `fig, (ax_adapt, ax_glob) = plt.subplots(2, 1, figsize=(14.5, 8.6), sharex=True)`
  - Hold reward fixed to `"balanced"` across `LEVEL_IDS = [1..7]`.
  - 4 grouped bars per level (`bar_width = 0.19`, `offsets = [-0.285, -0.095, 0.095, 0.285]`):
    1. `("tabular", tabular_store, "curriculum", False)` — Tabular Curriculum (Blue, solid)
    2. `("tabular", tabular_store, "separate", True)` — Tabular Separate (Blue, hatched `//`)
    3. `("hybrid_expected", hybrid_store, "curriculum", False)` — Hybrid Curriculum (Orange, solid)
    4. `("hybrid_expected", hybrid_store, "separate", True)` — Hybrid Separate (Orange, hatched `//`)
  - For each `(lvl, seed)` with `compute_sample_efficiency_milestone(df_lvl, threshold_wr=80.0)` returning `(ep_h, adapt_steps)`:
    - Row 0 (`ax_adapt`): `adapt_steps` (`level_cumulative_steps`) mean ± std.
    - Row 1 (`ax_glob`): `get_track_pretraining_steps(store, mode, "balanced", lvl, seed) + adapt_steps` mean ± std.
  - Render `k/N` badges above partial-convergence bars and rotated `DNC` labels where `k == 0`; format Y-axis with thousands separators or `M`/`k` formatter for clean readability; mark Track A (`L1–L5`) vs Track B (`L6–L7`) vertical separator at `x = 4.5`.
  - Save via `_save_figure(fig, save_pdf, save_png, "Fig. 4 (Sample Efficiency L1-L7)")`.

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_plot_fig4_sample_efficiency`
- Expected result: `OK`.

**Step 4: Commit Checkpoint**
- `feat(eval): implement Section 7.4 Fig. 4 Balanced sample efficiency and global compute plot`

---

### Task 6: Implement Section 7 Markdown Template Tables Generator (`generate_paper_evaluation_tables`)

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`
- Output: `evaluation/paper_evaluation_tables.md`

**Step 1: Write Failing Test (RED)**
- Add `test_generate_paper_evaluation_tables` to `TestEvaluationPlots`:

```python
    def test_generate_paper_evaluation_tables(self):
        md_path = self.tmp_dir / "paper_evaluation_tables.md"
        gce.generate_paper_evaluation_tables(md_path)
        self.assertTrue(md_path.exists())
        text = md_path.read_text(encoding="utf-8")
        self.assertIn("PLACEHOLDER", text)
        for marker in ("## 7.1", "## 7.2", "Table 1", "## 7.4", "Table 2", "## 7.5", "Table 3", "Level 7"):
            self.assertIn(marker, text)
        self.assertNotIn("Level 8", text)
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_generate_paper_evaluation_tables`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'generate_paper_evaluation_tables'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`, replace `generate_summary_report` and `generate_cross_architecture_comparison` with `generate_paper_evaluation_tables(save_md: Path) -> None`:
  - Writes a prominent `> [!WARNING]` banner stating **ALL numeric cells are `[PLACEHOLDER]` mock template values** awaiting completion of the updated training and 100-seed deterministic checkpoint evaluation (where each saved checkpoint plays each level across `100` procedural seeds).
  - Includes 4 cleanly formatted Markdown tables:
    1. **7.1 Setup & Evaluation Protocol Table**: Fixed hyperparameters (`episodes_per_level`, `seeds`, $\alpha_0 \to \alpha_{\min}$, $\gamma$, $\epsilon$ schedule, curriculum re-warmup/replay, Hybrid $\beta$, Track A `L1–L5` & Track B `L6–L7`) and the `100`-seed blind checkpoint evaluation rule.
    2. **7.2 Table 1 — Overall Final Win Rates (`100`-Seed Checkpoint Eval)**: All `7` levels (`L1–L7`) × `3` rewards (`Balanced`, `Dense`, `Sparse`) × `2` agents (`Tabular Q`, `Hybrid Expected SARSA`) × `2` modes (`Curriculum`, `Separate`) + $\Delta$ columns (`[PLACEHOLDER]` mock values).
    3. **7.4 Table 2 — Sample Efficiency & Total Compute Cost (`Balanced` Only, `L1–L7`)**: `Conv. (k/8)`, `Ep to 80% WR`, `Adaptation Steps to 80% WR`, `Pre-Training Steps`, `Global Steps to 80% WR`, and `Total Compute Cost` for all 4 `(agent, mode)` combinations (`[PLACEHOLDER]` mock values).
    4. **7.5 Table 3 — Failure Modes & Discovered State Counts (`L4`, `L5`, `L7`, `Balanced` Only)**: Mini-table with `Timeout %`, `Pit Death %`, `Ghost/Monster Death %`, `Win %`, and `Discovered States` (`[PLACEHOLDER]` mock values).

**Step 3: Verify (REFACTOR & CHECK)**
- Command: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_generate_paper_evaluation_tables`
- Expected result: `OK`.

**Step 4: Commit Checkpoint**
- `feat(eval): add Section 7.1-7.5 placeholder Markdown evaluation tables generator`

---

### Task 7: Wire `main()`, Add Stale Figure Cleanup, Execute End-to-End & Visually Inspect

**Files:**
- Test: `tests/test_evaluation_plots.py`
- Implementation: `evaluation/generate_curriculum_evaluation.py`

**Step 1: Write Failing Test (RED)**
- Add `test_cleanup_stale_figures_preserves_data_and_new_figs` to `TestEvaluationPlots`:

```python
    def test_cleanup_stale_figures_preserves_data_and_new_figs(self):
        eval_dir = self.tmp_dir / "evaluation"
        lncs_dir = self.tmp_dir / "springer_lncs" / "figures"
        (eval_dir / "tabular" / "data").mkdir(parents=True)
        (eval_dir / "hybrid_expected" / "data").mkdir(parents=True)
        (eval_dir / "comparison").mkdir(parents=True)
        lncs_dir.mkdir(parents=True)

        keep_csv = eval_dir / "tabular" / "data" / "curriculum_balanced_level_1_seed42.csv"
        keep_csv.write_text("a,b\n1,2\n", encoding="utf-8")
        keep_new_fig = eval_dir / "fig1_reward_trajectories.pdf"
        keep_new_fig.write_text("new", encoding="utf-8")
        keep_lncs_fig = lncs_dir / "fig1_reward_trajectories.pdf"
        keep_lncs_fig.write_text("new", encoding="utf-8")

        stale_files = [
            eval_dir / "figure_1_curriculum_win_rate.pdf",
            eval_dir / "tabular" / "figure_1_curriculum_win_rate.png",
            eval_dir / "hybrid_expected" / "figure_2_reward_trajectories.pdf",
            eval_dir / "comparison" / "level_1_balanced.png",
            lncs_dir / "tabular_figure_1_win_rate.pdf",
            lncs_dir / "figure_1_curriculum_win_rate.pdf",
        ]
        for sf in stale_files:
            sf.write_text("stale", encoding="utf-8")

        gce.cleanup_stale_figures(eval_dir, lncs_dir)

        self.assertTrue(keep_csv.exists())
        self.assertTrue(keep_new_fig.exists())
        self.assertTrue(keep_lncs_fig.exists())
        for sf in stale_files:
            self.assertFalse(sf.exists(), f"Stale file not removed: {sf}")
```

- Command to run: `python -m unittest tests.test_evaluation_plots.TestEvaluationPlots.test_cleanup_stale_figures_preserves_data_and_new_figs`
- Expected result: `FAIL` (`AttributeError: module ... has no attribute 'cleanup_stale_figures'`).

**Step 2: Implement Code (GREEN)**
- In `evaluation/generate_curriculum_evaluation.py`:
  1. Add `PAPER_FIG_STEMS = ("fig1_reward_trajectories", "fig2_balanced_overview", "fig3_spotlight_l3_l4_l6", "fig4_sample_efficiency")`.
  2. Implement `cleanup_stale_figures(eval_dir: Path, lncs_fig_dir: Optional[Path] = None) -> int` removing any `.pdf`/`.png` files in `eval_dir`, `eval_dir/{tabular,hybrid_expected,comparison}`, and `lncs_fig_dir` whose stem is not in `PAPER_FIG_STEMS` (and removing empty `eval_dir / "comparison"` if empty). Never touch `data/*.csv` or `.md` files.
  3. Update `main()` to:
     - Load `tabular_store` and `hybrid_store` via `load_all_split_data`.
     - Generate `fig1`–`fig4` (`.pdf` & `.png`) into `EVAL_DIR`.
     - Generate `EVAL_DIR / "paper_evaluation_tables.md"`.
     - Sync the 4 `.pdf` figures to `PROJECT_ROOT / "springer_lncs" / "figures"`.
     - Call `cleanup_stale_figures(EVAL_DIR, lncs_fig_dir)`.

**Step 3: Verify (REFACTOR & CHECK)**
1. Run full unit test suite:
   ```powershell
   python -m unittest discover -s tests -p "test_*.py"
   ```
2. Run the evaluation generator on the real workspace dataset:
   ```powershell
   python evaluation/generate_curriculum_evaluation.py
   ```
3. Verify generated figures and absence of stale files:
   ```powershell
   Get-ChildItem evaluation, evaluation\tabular, evaluation\hybrid_expected, springer_lncs\figures -File -Name
   ```
4. Visually inspect `evaluation/fig1_reward_trajectories.png`, `evaluation/fig2_balanced_overview.png`, `evaluation/fig3_spotlight_l3_l4_l6.png`, and `evaluation/fig4_sample_efficiency.png` using `read_file` to confirm layout, legibility, and zero overlapping labels, and inspect `evaluation/paper_evaluation_tables.md`.

**Step 4: Commit Checkpoint**
- `feat(eval): wire Section 7 figure suite, cleanup stale plots, and generate placeholder tables`
