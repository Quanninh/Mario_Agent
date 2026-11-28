import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import evaluation.generate_curriculum_evaluation as gce
import evaluation.split_level_data as sld


def _make_level_df(
    level_id: int,
    n_eps: int = 120,
    win_after: int = 40,
    steps_per_ep: int = 100,
    start_global_steps: int = 0,
) -> pd.DataFrame:
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
                cum_steps = 0
                for lvl in range(1, 8):
                    win_after = 20 if (strat == "balanced" and lvl in (1, 2, 3, 6)) else 9999
                    base = cum_steps if mode == "curriculum" else 0
                    df = _make_level_df(
                        lvl, n_eps=120, win_after=win_after, steps_per_ep=100, start_global_steps=base
                    )
                    store[(mode, strat, lvl, seed)] = df
                    if mode == "curriculum":
                        cum_steps += int(df["steps"].sum())
    return store


class TestEvaluationPlots(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_eval_plots_"))
        self.tab_store = _build_synthetic_store()
        self.hyb_store = _build_synthetic_store()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_splitter_excludes_level_8_and_parses_lvl1_8_names(self):
        out_dir = self.tmp_dir / "outputs"
        eval_dir = self.tmp_dir / "evaluation"
        out_dir.mkdir(parents=True)

        dfs = [_make_level_df(lvl, n_eps=10) for lvl in range(1, 9)]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all["global_episode"] = range(1, len(df_all) + 1)
        csv_path = out_dir / "training_metrics_tabular_curriculum_balanced_lvl1-8_seed42.csv"
        df_all.to_csv(csv_path, index=False)
        # Non-benchmark file should be skipped safely
        (out_dir / "training_metrics_hybrid_linear_separate_balanced.csv").write_text("x\n1\n", encoding="utf-8")

        res = sld.split_all_master_files(outputs_dir=out_dir, eval_dir=eval_dir)
        tab_files = [p.name for p in res["tabular"]]
        self.assertEqual(len(tab_files), 7)
        for lvl in range(1, 8):
            self.assertIn(f"curriculum_balanced_level_{lvl}_seed42.csv", tab_files)
        self.assertNotIn("curriculum_balanced_level_8_seed42.csv", tab_files)

    def test_levels_and_pretraining_steps(self):
        self.assertEqual(gce.LEVEL_IDS, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(gce.get_pretraining_steps(self.tab_store, "curriculum", "balanced", 1, "seed42"), 0.0)
        self.assertEqual(gce.get_pretraining_steps(self.tab_store, "curriculum", "balanced", 2, "seed42"), 12000.0)
        self.assertEqual(gce.get_pretraining_steps(self.tab_store, "curriculum", "balanced", 6, "seed42"), 60000.0)
        self.assertEqual(gce.get_pretraining_steps(self.tab_store, "separate", "balanced", 3, "seed42"), 0.0)

    def test_plot_fig1_reward_trajectories(self):
        pdf = self.tmp_dir / "fig1_reward_trajectories.pdf"
        png = self.tmp_dir / "fig1_reward_trajectories.png"
        gce.plot_fig1_reward_trajectories(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)

    def test_plot_fig2_balanced_overview(self):
        pdf = self.tmp_dir / "fig2_balanced_overview.pdf"
        png = self.tmp_dir / "fig2_balanced_overview.png"
        gce.plot_fig2_balanced_overview(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)

    def test_plot_fig3_spotlight_l3_l4_l6(self):
        pdf = self.tmp_dir / "fig3_spotlight_l3_l4_l6.pdf"
        png = self.tmp_dir / "fig3_spotlight_l3_l4_l6.png"
        gce.plot_fig3_spotlight_l3_l4_l6(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)

    def test_plot_fig4_sample_efficiency(self):
        pdf = self.tmp_dir / "fig4_sample_efficiency.pdf"
        png = self.tmp_dir / "fig4_sample_efficiency.png"
        gce.plot_fig4_sample_efficiency(self.tab_store, self.hyb_store, pdf, png)
        self.assertTrue(pdf.exists() and pdf.stat().st_size > 1000)
        self.assertTrue(png.exists() and png.stat().st_size > 1000)

    def test_generate_paper_evaluation_tables(self):
        md_path = self.tmp_dir / "paper_evaluation_tables.md"
        gce.generate_paper_evaluation_tables(self.tab_store, self.hyb_store, md_path, outputs_dir=self.tmp_dir)
        self.assertTrue(md_path.exists())
        text = md_path.read_text(encoding="utf-8")
        for marker in ("## 7.1", "## 7.2", "Table 1", "## 7.4", "Table 2", "## 7.5", "Table 3", "Level 7"):
            self.assertIn(marker, text)
        self.assertNotIn("Level 8", text)

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


if __name__ == "__main__":
    unittest.main()
