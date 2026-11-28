"""
Mario RL Agent — Section 7 Evaluation Figure & Table Suite (Levels 1-7)
=======================================================================
Processes split level CSVs from evaluation/<arch>/data/ (and optional metadata /
greedy evaluation summaries from outputs/training_board_*.json) across Levels 1-7
(strictly excluding Level 8) and produces:

1. fig1_reward_trajectories.{pdf,png}:
   6-panel cross-architecture reward return grid (Sparse, Balanced, Dense x
   Curriculum vs. Separate) across Levels 1-7.
2. fig2_balanced_overview.{pdf,png}:
   2-panel Balanced reward overview comparing Tabular Q-Learning vs. Hybrid
   Expected SARSA rolling win rate (W=100) and epsilon schedule across Levels 1-7.
3. fig3_spotlight_l3_l4_l6.{pdf,png}:
   2x3 spotlight grid focusing on Levels 3, 4, and 6 (Balanced) under Curriculum
   and Separate training.
4. fig4_sample_efficiency.{pdf,png}:
   2-panel Balanced sample efficiency comparison across Levels 1-7:
   within-level adaptation steps vs. global compute steps to 80% win rate.
5. paper_evaluation_tables.md:
   Quantitative Markdown tables for Section 7 (Levels 1-7 only).

Also synchronizes the 4 PDFs to springer_lncs/figures/ and cleans stale legacy
figures without touching evaluation/*/data/*.csv or *.md files.
"""

import argparse
import glob
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Academic typography & aesthetic configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.92,
    "figure.titlesize": 14.0,
    "figure.titleweight": "bold",
})

ARCH_COLORS = {
    "tabular": "#1f77b4",
    "hybrid_expected": "#ff7f0e",
}

ARCH_LABELS = {
    "tabular": "Tabular Q-Learning",
    "hybrid_expected": "Hybrid Expected SARSA",
}

STRATEGY_LABELS = {
    "balanced": "Balanced",
    "dense": "Dense",
    "sparse": "Sparse",
}

LEVEL_IDS = [1, 2, 3, 4, 5, 6, 7]
TRACKS = {"A": [1, 2, 3, 4, 5], "B": [6, 7]}
PAPER_FIG_STEMS = (
    "fig1_reward_trajectories",
    "fig2_balanced_overview",
    "fig3_spotlight_l3_l4_l6",
    "fig4_sample_efficiency",
)


def _format_compact_number(val: float, _pos: Optional[int] = None) -> str:
    """Format large integer counts with M / k suffixes for clean axis ticks."""
    abs_v = abs(val)
    if abs_v >= 1_000_000:
        scaled = val / 1_000_000.0
        return f"{scaled:.1f}M" if abs(scaled - round(scaled)) > 1e-6 else f"{int(round(scaled))}M"
    if abs_v >= 1_000:
        scaled = val / 1_000.0
        return f"{scaled:.1f}k" if abs(scaled - round(scaled)) > 1e-6 else f"{int(round(scaled))}k"
    return f"{int(round(val))}"


def _save_figure(fig: plt.Figure, save_pdf: Path, save_png: Path, label: str) -> None:
    """Save figure to PDF and PNG and close it."""
    save_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {label}: {save_pdf.name} & {save_png.name}")


def _draw_stage_dividers(
    ax: plt.Axes,
    n_eps_per_level: int,
    show_labels: bool = True,
    label_y: Optional[float] = None,
) -> None:
    """Draw dashed vertical stage boundaries for Levels 1-7 and centered L1..L7 labels."""
    for k in range(1, len(LEVEL_IDS)):
        boundary = k * n_eps_per_level
        ax.axvline(boundary, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)

    if show_labels:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_pos = label_y if label_y is not None else (y_max + 0.025 * y_span)
        for idx, lvl in enumerate(LEVEL_IDS):
            center = (idx + 0.5) * n_eps_per_level
            ax.text(
                center,
                y_pos,
                f"L{lvl}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#333333",
            )


def _configure_stage_xaxis(ax: plt.Axes, n_eps_per_level: int) -> None:
    """Configure x-axis bounds and ticks aligned with stage boundaries."""
    total_eps = len(LEVEL_IDS) * n_eps_per_level
    ax.set_xlim(0, total_eps)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(n_eps_per_level))
    if n_eps_per_level >= 1000:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_compact_number))


def load_all_split_data(data_dir: Path) -> Dict[Tuple[str, str, int, str], pd.DataFrame]:
    """Load split CSV files for Levels 1-7 from data_dir."""
    pattern = str(data_dir / "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No split CSV files found in {data_dir}. Run split_level_data.py first.")

    filename_regex = re.compile(
        r"^(curriculum|separate)_(balanced|dense|sparse)_level_(\d+)_(seed\w+|\d+)\.csv$"
    )

    data_store: Dict[Tuple[str, str, int, str], pd.DataFrame] = {}
    for fpath_str in sorted(files):
        fpath = Path(fpath_str)
        match = filename_regex.match(fpath.name)
        if not match:
            continue
        mode, strategy, lvl_str, seed = match.groups()
        lvl = int(lvl_str)
        if lvl not in LEVEL_IDS:
            continue
        data_store[(mode, strategy, lvl, seed)] = pd.read_csv(fpath)

    return data_store


def get_pretraining_steps(
    data_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    mode: str,
    strategy: str,
    level_id: int,
    seed: str,
) -> float:
    """
    Return cumulative pre-training steps prior to `level_id`.
    Reads `global_cumulative_steps` directly when present so both Track A/B
    splits and continuous Track C runs (L1-L7) are accounted for accurately.
    """
    if mode != "curriculum" or level_id <= 1:
        return 0.0

    df_lvl = data_store.get((mode, strategy, level_id, seed))
    if df_lvl is not None and len(df_lvl) > 0 and "global_cumulative_steps" in df_lvl.columns:
        return max(0.0, float(df_lvl["global_cumulative_steps"].iloc[0] - df_lvl["steps"].iloc[0]))

    total = 0.0
    for prev_lvl in LEVEL_IDS:
        if prev_lvl >= level_id:
            break
        df_prev = data_store.get((mode, strategy, prev_lvl, seed))
        if df_prev is not None and len(df_prev) > 0:
            total += float(df_prev["steps"].sum())
    return total


def assemble_continuous_run(
    data_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    mode: str,
    strategy: str,
    seed: str,
    wr_window: int = 100,
    rew_window: int = 100,
) -> Optional[Tuple[pd.DataFrame, int]]:
    """Concatenate Levels 1-7 for (mode, strategy, seed) and compute rolling metrics."""
    dfs = []
    lengths = []
    for lvl in LEVEL_IDS:
        key = (mode, strategy, lvl, seed)
        if key not in data_store:
            return None
        df_lvl = data_store[key]
        if len(df_lvl) == 0:
            return None
        dfs.append(df_lvl)
        lengths.append(len(df_lvl))

    if len(set(lengths)) != 1:
        return None
    n_eps_per_level = lengths[0]

    df_full = pd.concat(dfs, ignore_index=True)
    df_full["global_ep"] = np.arange(1, len(df_full) + 1)
    df_full["rolling_wr"] = (
        df_full["won"].astype(float).rolling(wr_window, min_periods=1).mean() * 100.0
    )
    df_full["rolling_reward"] = (
        df_full["total_reward"].astype(float).rolling(rew_window, min_periods=1).mean()
    )
    return df_full, n_eps_per_level


def get_multi_seed_trajectories(
    data_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    mode: str,
    strategy: str,
    wr_window: int = 100,
    rew_window: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Aggregate continuous Levels 1-7 trajectories across all seeds."""
    seeds = sorted({k[3] for k in data_store.keys() if k[0] == mode and k[1] == strategy})

    wr_runs: List[np.ndarray] = []
    reward_runs: List[np.ndarray] = []
    eps_runs: List[np.ndarray] = []
    expected_n_eps: Optional[int] = None

    for s in seeds:
        assembled = assemble_continuous_run(
            data_store, mode, strategy, s, wr_window=wr_window, rew_window=rew_window
        )
        if assembled is None:
            continue
        df_run, n_eps_per_level = assembled
        if expected_n_eps is None:
            expected_n_eps = n_eps_per_level
        if n_eps_per_level != expected_n_eps:
            continue
        wr_runs.append(df_run["rolling_wr"].values)
        reward_runs.append(df_run["rolling_reward"].values)
        if "epsilon" in df_run.columns:
            eps_runs.append(df_run["epsilon"].values)

    n_seeds = len(wr_runs)
    if n_seeds == 0 or expected_n_eps is None:
        fallback_n = 100
        total = fallback_n * len(LEVEL_IDS)
        episodes = np.arange(1, total + 1)
        zeros = np.zeros(total)
        return episodes, zeros, zeros, zeros, zeros, zeros, 0, fallback_n

    total_eps = len(LEVEL_IDS) * expected_n_eps
    episodes = np.arange(1, total_eps + 1)
    wr_mat = np.vstack(wr_runs)
    reward_mat = np.vstack(reward_runs)

    wr_mean = np.mean(wr_mat, axis=0)
    wr_std = np.std(wr_mat, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(wr_mean)
    reward_mean = np.mean(reward_mat, axis=0)
    reward_std = np.std(reward_mat, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(reward_mean)
    eps_curve = np.mean(np.vstack(eps_runs), axis=0) if eps_runs else np.zeros_like(wr_mean)

    return episodes, wr_mean, wr_std, reward_mean, reward_std, eps_curve, n_seeds, expected_n_eps


def get_level_multi_seed_trajectory(
    data_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    mode: str,
    strategy: str,
    level_id: int,
    wr_window: int = 100,
    rew_window: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate within-level trajectories across all seeds for a single level."""
    seeds = sorted({
        k[3] for k in data_store.keys()
        if k[0] == mode and k[1] == strategy and k[2] == level_id
    })

    wr_runs: List[np.ndarray] = []
    reward_runs: List[np.ndarray] = []
    eps_runs: List[np.ndarray] = []
    expected_len: Optional[int] = None

    for s in seeds:
        df_lvl = data_store[(mode, strategy, level_id, s)]
        if len(df_lvl) == 0:
            continue
        if expected_len is None:
            expected_len = len(df_lvl)
        if len(df_lvl) != expected_len:
            continue
        wr_series = df_lvl["won"].astype(float).rolling(wr_window, min_periods=1).mean() * 100.0
        rew_series = df_lvl["total_reward"].astype(float).rolling(rew_window, min_periods=1).mean()
        wr_runs.append(wr_series.values)
        reward_runs.append(rew_series.values)
        if "epsilon" in df_lvl.columns:
            eps_runs.append(df_lvl["epsilon"].values)

    n_seeds = len(wr_runs)
    if n_seeds == 0 or expected_len is None:
        episodes = np.arange(1, 101)
        zeros = np.zeros(100)
        return episodes, zeros, zeros, zeros, zeros, zeros, 0

    episodes = np.arange(1, expected_len + 1)
    wr_mat = np.vstack(wr_runs)
    rew_mat = np.vstack(reward_runs)

    wr_mean = np.mean(wr_mat, axis=0)
    wr_std = np.std(wr_mat, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(wr_mean)
    rew_mean = np.mean(rew_mat, axis=0)
    rew_std = np.std(rew_mat, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(rew_mean)
    eps_curve = np.mean(np.vstack(eps_runs), axis=0) if eps_runs else np.zeros_like(wr_mean)

    return episodes, wr_mean, wr_std, rew_mean, rew_std, eps_curve, n_seeds


def compute_sample_efficiency_milestone(
    df_level: pd.DataFrame,
    threshold_wr: float = 80.0,
    window: int = 100,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Compute the first within-level episode and level_cumulative_steps where
    the rolling win rate (over `window` episodes) reaches `threshold_wr`.
    """
    rolling_wr = df_level["won"].astype(float).rolling(window, min_periods=window).mean() * 100.0
    hit_idx = np.where(rolling_wr >= threshold_wr)[0]
    if len(hit_idx) == 0:
        return None, None
    first_idx = int(hit_idx[0])
    ep_to_hit = int(df_level["level_episode"].iloc[first_idx])
    steps_to_hit = float(df_level["level_cumulative_steps"].iloc[first_idx])
    return ep_to_hit, steps_to_hit


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Cross-Architecture Reward Trajectories (Levels 1-7)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_reward_trajectories(
    tabular_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    hybrid_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    save_pdf: Path,
    save_png: Path,
) -> None:
    """
    Fig. 1: 3x2 grid (Sparse, Balanced, Dense x Curriculum, Separate) comparing
    Tabular Q-Learning vs. Hybrid Expected SARSA smoothed returns across Levels 1-7.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 9.8), sharex=True)
    strategies = ["sparse", "balanced", "dense"]
    modes = [("curriculum", "Curriculum Transfer"), ("separate", "Separate Training")]
    arch_stores = [("tabular", tabular_store), ("hybrid_expected", hybrid_store)]

    n_eps_ref = 100
    for r, strat in enumerate(strategies):
        row_mins: List[float] = []
        row_maxs: List[float] = []

        for c, (mode, _mode_title) in enumerate(modes):
            ax = axes[r, c]
            for arch, store in arch_stores:
                ep, _, _, rew_m, rew_s, _, n_seeds, n_eps = get_multi_seed_trajectories(
                    store, mode, strat, wr_window=100, rew_window=100
                )
                n_eps_ref = n_eps
                color = ARCH_COLORS[arch]
                label = f"{ARCH_LABELS[arch]} (N={n_seeds})" if n_seeds > 1 else ARCH_LABELS[arch]
                ax.plot(ep, rew_m, color=color, linewidth=1.7, label=label)
                if n_seeds > 1:
                    ax.fill_between(ep, rew_m - rew_s, rew_m + rew_s, color=color, alpha=0.18)
                row_mins.append(float(np.min(rew_m - rew_s)))
                row_maxs.append(float(np.max(rew_m + rew_s)))

        y_lo = min(row_mins)
        y_hi = max(max(row_maxs), 1.0)
        span = y_hi - y_lo
        pad = 0.08 * (span if span > 0 else 1.0)
        y_min, y_max = y_lo - pad, y_hi + pad

        for c in range(2):
            ax = axes[r, c]
            ax.set_ylim(y_min, y_max)
            _configure_stage_xaxis(ax, n_eps_ref)
            _draw_stage_dividers(ax, n_eps_ref, show_labels=(r == 0))
            if c == 0:
                ax.set_ylabel(f"{STRATEGY_LABELS[strat]} Return\n(100-Ep Mean)", fontweight="bold")
            if c == 0:
                ax.legend(loc="upper right", framealpha=0.92)

    for c, (_mode, mode_title) in enumerate(modes):
        axes[0, c].set_title(mode_title, pad=22, fontweight="bold")
        axes[2, c].set_xlabel("Cumulative Training Episodes (Levels 1–7)", fontweight="bold")

    fig.suptitle(
        "Cross-Architecture Reward Trajectories Across Levels 1–7 (Mean ± 1 SD)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.08, right=0.98, hspace=0.20, wspace=0.14)
    _save_figure(fig, save_pdf, save_png, "Fig. 1 (Reward Trajectories L1-L7)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Balanced Overview Win Rate Curves (Levels 1-7)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_balanced_overview(
    tabular_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    hybrid_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    save_pdf: Path,
    save_png: Path,
) -> None:
    """
    Fig. 2: 2x1 vertically stacked Balanced win-rate overview (Curriculum vs. Separate)
    comparing Tabular Q-Learning and Hybrid Expected SARSA across Levels 1-7.
    """
    fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True, sharey=True)
    arch_stores = [("tabular", tabular_store), ("hybrid_expected", hybrid_store)]
    n_eps_ref = 100

    for mode, ax, panel_title in [
        ("curriculum", ax_c, "Curriculum Transfer (Continuous Policy Transfer L1–L7)"),
        ("separate", ax_s, "Separate Training (Fresh Agent Per Stage L1–L7)"),
    ]:
        ax_twin = ax.twinx()
        eps_plotted = False

        for arch, store in arch_stores:
            ep, wr_m, wr_s, _, _, eps_curve, n_seeds, n_eps = get_multi_seed_trajectories(
                store, mode, "balanced", wr_window=100, rew_window=100
            )
            n_eps_ref = n_eps
            color = ARCH_COLORS[arch]
            label = f"{ARCH_LABELS[arch]} (N={n_seeds})" if n_seeds > 1 else ARCH_LABELS[arch]
            ax.plot(ep, wr_m, color=color, linewidth=1.9, label=label)
            if n_seeds > 1:
                ax.fill_between(
                    ep,
                    np.clip(wr_m - wr_s, 0.0, 100.0),
                    np.clip(wr_m + wr_s, 0.0, 100.0),
                    color=color,
                    alpha=0.18,
                )
            if not eps_plotted and eps_curve is not None:
                ax_twin.plot(
                    ep,
                    eps_curve,
                    color="#666666",
                    linestyle="-." if mode == "curriculum" else ":",
                    linewidth=1.2,
                    alpha=0.75,
                    label=r"Exploration $\epsilon$",
                )
                eps_plotted = True

        ax.axhline(80.0, color="#444444", linestyle=":", linewidth=1.1, alpha=0.85, label="80% Milestone")
        ax.set_ylim(-3.0, 105.0)
        _configure_stage_xaxis(ax, n_eps_ref)
        _draw_stage_dividers(ax, n_eps_ref, show_labels=True, label_y=106.5)
        ax.set_ylabel("Rolling Win Rate (%)\n[W = 100]", fontweight="bold")
        ax.set_title(panel_title, pad=22, fontweight="bold")

        ax_twin.set_ylim(-0.02, 1.08)
        ax_twin.set_ylabel(r"Exploration Rate ($\epsilon$)", color="#555555", fontweight="bold")
        ax_twin.tick_params(axis="y", labelcolor="#555555")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.92)

    ax_s.set_xlabel("Cumulative Training Episodes (Levels 1–7, Balanced Reward)", fontweight="bold")
    fig.suptitle(
        "Balanced Reward Learning Curves Across Levels 1–7 (Tabular Q vs. Hybrid Expected SARSA)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.92, hspace=0.28)
    _save_figure(fig, save_pdf, save_png, "Fig. 2 (Balanced Overview L1-L7)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Spotlight Grid for Levels 3, 4, and 6 (Balanced)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_spotlight_l3_l4_l6(
    tabular_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    hybrid_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    save_pdf: Path,
    save_png: Path,
) -> None:
    """
    Fig. 3: 2x3 spotlight grid (Curriculum, Separate x Levels 3, 4, 6) under
    Balanced reward, highlighting transition, hazard gauntlet, and monster ambush stages.
    """
    spotlight_levels = [
        (3, "Level 3: Whispering Graves"),
        (4, "Level 4: Nightmare Catacombs"),
        (6, "Level 6: Awakening of Corpse Demon"),
    ]
    modes = [("curriculum", "Curriculum Transfer"), ("separate", "Separate Training")]
    arch_stores = [("tabular", tabular_store), ("hybrid_expected", hybrid_store)]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.8), sharex="col", sharey=True)

    for r, (mode, mode_label) in enumerate(modes):
        for c, (lvl, lvl_title) in enumerate(spotlight_levels):
            ax = axes[r, c]
            ax_twin = ax.twinx()
            eps_plotted = False
            ep_len = 100

            for arch, store in arch_stores:
                ep, wr_m, wr_s, _, _, eps_curve, n_seeds = get_level_multi_seed_trajectory(
                    store, mode, "balanced", lvl, wr_window=100, rew_window=100
                )
                ep_len = len(ep)
                color = ARCH_COLORS[arch]
                label = f"{ARCH_LABELS[arch]} (N={n_seeds})" if n_seeds > 1 else ARCH_LABELS[arch]
                ax.plot(ep, wr_m, color=color, linewidth=1.8, label=label)
                if n_seeds > 1:
                    ax.fill_between(
                        ep,
                        np.clip(wr_m - wr_s, 0.0, 100.0),
                        np.clip(wr_m + wr_s, 0.0, 100.0),
                        color=color,
                        alpha=0.18,
                    )
                if not eps_plotted and eps_curve is not None:
                    ax_twin.plot(
                        ep,
                        eps_curve,
                        color="#666666",
                        linestyle="-." if mode == "curriculum" else ":",
                        linewidth=1.15,
                        alpha=0.72,
                        label=r"$\epsilon$",
                    )
                    eps_plotted = True

            ax.axhline(80.0, color="#444444", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_ylim(-3.0, 105.0)
            ax.set_xlim(0, ep_len)
            if ep_len >= 1000:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_compact_number))

            ax_twin.set_ylim(-0.02, 1.08)
            if c == 2:
                ax_twin.set_ylabel(r"Exploration ($\epsilon$)", color="#555555", fontweight="bold")
                ax_twin.tick_params(axis="y", labelcolor="#555555")
            else:
                ax_twin.set_yticklabels([])
                ax_twin.tick_params(axis="y", length=0)

            if r == 0:
                ax.set_title(lvl_title, pad=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{mode_label}\nWin Rate (%) [W=100]", fontweight="bold")
            if r == 1:
                ax.set_xlabel("Within-Level Episode", fontweight="bold")
            if r == 0 and c == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax_twin.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.92, fontsize=8.5)

    fig.suptitle(
        "Stage Spotlight: Rolling Win Rate on Levels 3, 4, and 6 (Balanced Reward)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.89, bottom=0.09, left=0.07, right=0.94, hspace=0.18, wspace=0.12)
    _save_figure(fig, save_pdf, save_png, "Fig. 3 (Spotlight L3, L4, L6)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Balanced Sample Efficiency & Global Compute Cost (Levels 1-7)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_sample_efficiency(
    tabular_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    hybrid_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    save_pdf: Path,
    save_png: Path,
) -> None:
    """
    Fig. 4: 2-row grouped bar chart (Balanced reward, Levels 1-7) comparing:
    - Row 0: Within-Level Adaptation Steps (level_cumulative_steps) to 80% WR.
    - Row 1: Global Compute Steps (pre-training + adaptation steps) to 80% WR.
    """
    fig, (ax_adapt, ax_glob) = plt.subplots(2, 1, figsize=(14.5, 8.6), sharex=True)

    combos = [
        ("tabular", tabular_store, "curriculum", False, "Tabular Curriculum"),
        ("tabular", tabular_store, "separate", True, "Tabular Separate"),
        ("hybrid_expected", hybrid_store, "curriculum", False, "Hybrid Expected Curriculum"),
        ("hybrid_expected", hybrid_store, "separate", True, "Hybrid Expected Separate"),
    ]
    bar_width = 0.19
    offsets = [-0.285, -0.095, 0.095, 0.285]
    x_indices = np.arange(len(LEVEL_IDS))

    # Store bar objects + badge metadata so text offsets adapt cleanly to final ylim
    pending_annotations: List[Tuple[plt.Axes, object, float, str]] = []

    for idx, (arch, store, mode, is_hatched, legend_label) in enumerate(combos):
        color = ARCH_COLORS[arch]
        hatch = "//" if is_hatched else None
        edgecolor = "#222222" if is_hatched else "none"
        alpha = 0.78 if is_hatched else 0.92

        adapt_means, adapt_errs, adapt_badges = [], [], []
        glob_means, glob_errs, glob_badges = [], [], []

        seeds = sorted({k[3] for k in store.keys() if k[0] == mode and k[1] == "balanced"})

        for lvl in LEVEL_IDS:
            adapt_hits: List[float] = []
            glob_hits: List[float] = []
            n_cond_seeds = 0

            for s in seeds:
                key = (mode, "balanced", lvl, s)
                if key not in store:
                    continue
                n_cond_seeds += 1
                df_lvl = store[key]
                ep_h, adapt_st = compute_sample_efficiency_milestone(df_lvl, threshold_wr=80.0, window=100)
                if ep_h is not None and adapt_st is not None:
                    pre_st = get_pretraining_steps(store, mode, "balanced", lvl, s)
                    adapt_hits.append(adapt_st)
                    glob_hits.append(pre_st + adapt_st)

            k_conv = len(adapt_hits)
            badge = "DNC" if k_conv == 0 else (f"{k_conv}/{n_cond_seeds}" if k_conv < n_cond_seeds else "")

            for hits, means, errs, badges in [
                (adapt_hits, adapt_means, adapt_errs, adapt_badges),
                (glob_hits, glob_means, glob_errs, glob_badges),
            ]:
                if k_conv == 0:
                    means.append(0.0)
                    errs.append((0.0, 0.0))
                else:
                    mu = float(np.mean(hits))
                    sig = float(np.std(hits, ddof=1)) if k_conv > 1 else 0.0
                    means.append(mu)
                    errs.append((min(sig, mu), sig))
                badges.append(badge)

        x_pos = x_indices + offsets[idx]

        for ax, means, errs, badges in [
            (ax_adapt, adapt_means, adapt_errs, adapt_badges),
            (ax_glob, glob_means, glob_errs, glob_badges),
        ]:
            yerr = np.array(errs).T
            bars = ax.bar(
                x_pos,
                means,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor=edgecolor,
                linewidth=0.8,
                alpha=alpha,
                yerr=yerr,
                capsize=3,
                label=legend_label,
            )
            for rect, (_low, up), badge in zip(bars, errs, badges):
                if badge:
                    pending_annotations.append((ax, rect, up, badge))

    for ax in (ax_adapt, ax_glob):
        y_min, y_max = ax.get_ylim()
        y_max = max(y_max, 1.0)
        ax.set_ylim(0.0, y_max * 1.18)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_compact_number))

    for ax, rect, err_up, badge in pending_annotations:
        y_top = ax.get_ylim()[1]
        x_center = rect.get_x() + rect.get_width() / 2.0
        if badge == "DNC":
            ax.text(
                x_center,
                0.02 * y_top,
                "DNC",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#d62728",
                rotation=90,
            )
        else:
            ax.text(
                x_center,
                rect.get_height() + err_up + 0.02 * y_top,
                badge,
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color="#222222",
            )

    ax_adapt.set_ylabel("Adaptation Steps to 80% WR\n(Within-Level)", fontweight="bold")
    ax_adapt.set_title("Within-Level Adaptation Cost vs. Total Global Compute Cost (Balanced Reward, Levels 1–7)", pad=12)
    ax_adapt.legend(loc="upper left", ncol=2, framealpha=0.92)

    ax_glob.set_ylabel("Global Steps to 80% WR\n(Pre-Training + Adaptation)", fontweight="bold")
    ax_glob.set_xticks(x_indices)
    ax_glob.set_xticklabels([f"Level {lvl}" for lvl in LEVEL_IDS], fontweight="bold", fontsize=11)
    ax_glob.set_xlabel("Curriculum Stage (Levels 1–7)", fontweight="bold")

    fig.suptitle(
        "Sample Efficiency & Cumulative Compute Cost Across Levels 1–7",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.09, right=0.98, hspace=0.18)
    _save_figure(fig, save_pdf, save_png, "Fig. 4 (Sample Efficiency L1-L7)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 EVALUATION TABLES GENERATOR (Levels 1-7)
# ─────────────────────────────────────────────────────────────────────────────
def _load_training_boards(outputs_dir: Path) -> Dict[Tuple[str, str, str, str], dict]:
    """Load training_board_*.json metadata & evaluation summaries quickly."""
    boards: Dict[Tuple[str, str, str, str], dict] = {}
    if not outputs_dir.exists():
        return boards

    pat = re.compile(
        r"^training_board_(tabular|hybrid_expected)_(curriculum|separate)_(balanced|dense|sparse)(?:_lvl[\d-]+)?_(seed\w+)\.json$"
    )
    for fpath in sorted(outputs_dir.glob("training_board_*.json")):
        m = pat.match(fpath.name)
        if not m:
            continue
        arch, mode, strat, seed = m.groups()
        try:
            # Read only the header before "episode_history" to avoid parsing ~25MB/file
            prefix_lines: List[str] = []
            found_history = False
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if '"episode_history"' in line:
                        found_history = True
                        break
                    prefix_lines.append(line)
            if found_history and prefix_lines:
                raw_prefix = "".join(prefix_lines).rstrip().rstrip(",") + "\n}"
                boards[(arch, mode, strat, seed)] = json.loads(raw_prefix)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    boards[(arch, mode, strat, seed)] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return boards


def _fmt_mean_std(vals: List[float], decimals: int = 1, suffix: str = "", dnc_label: str = "DNC") -> str:
    if not vals:
        return dnc_label
    m = float(np.mean(vals))
    if len(vals) > 1:
        sd = float(np.std(vals, ddof=1))
        return f"{m:,.{decimals}f} ± {sd:,.{decimals}f}{suffix}"
    return f"{m:,.{decimals}f}{suffix}"


def generate_paper_evaluation_tables(
    tabular_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    hybrid_store: Dict[Tuple[str, str, int, str], pd.DataFrame],
    save_md: Path,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Generate evaluation/paper_evaluation_tables.md for Sections 7.1-7.5 across
    Levels 1-7 only (strictly excluding Level 8). Computes all values directly
    from the split training CSVs and training_board_*.json summaries.
    """
    boards = _load_training_boards(outputs_dir)
    all_seeds = sorted({
        k[3] for store in (tabular_store, hybrid_store) for k in store.keys()
    })
    sample_df = next(iter(tabular_store.values()), None)
    eps_per_lvl = len(sample_df) if sample_df is not None else 0

    seed_range_str = f"{all_seeds[0]}–{all_seeds[-1]}" if len(all_seeds) > 1 else (all_seeds[0] if all_seeds else "N/A")
    lines: List[str] = [
        "# Section 7 Quantitative Evaluation Tables (Levels 1–7)",
        "",
        f"**Scope**: Levels 1–7 | **Seeds**: `{len(all_seeds)}` (`{seed_range_str}`) | **Episodes per Level**: `{eps_per_lvl:,}`",
        "",
        "---",
        "",
        "## 7.1 Experimental Setup & Evaluation Protocol",
        "",
        "| Parameter | Value | Notes |",
        "| :--- | :--- | :--- |",
        "| **Curriculum Scope** | Levels 1–7 | Restricted to the 7 core evaluation stages |",
        f"| **Episodes per Level** | `{eps_per_lvl:,}` | `{eps_per_lvl * len(LEVEL_IDS):,}` episodes per 7-level run |",
        f"| **Random Seeds** | `{len(all_seeds)}` ({', '.join(all_seeds)}) | Independent agent + environment training seeds |",
        "| **Architectures** | Tabular Q-Learning, Hybrid Expected SARSA | Both evaluated in `curriculum` and `separate` modes |",
        "| **Reward Profiles** | `balanced`, `dense`, `sparse` | Canonical benchmark (`balanced`) + shaping & sparse ablations |",
        "| **Training Hyperparameters** | $\\alpha: 0.10 \\to 0.005$, $\\gamma = 0.99$, $\\epsilon$-decay $= 0.999$ | $\\epsilon \\in [0.01, 1.0]$; Curriculum re-warmup $\\epsilon_0=0.30$ (L2–3), $0.35$ (L4–7) |",
        "| **Training-Time Win Rate** | Final-100 episodes ($W=100$) | Arithmetic mean of `won` over last 100 episodes per stage |",
        "| **Greedy Eval Win Rate** | 50 greedy episodes (`evaluation_summary`) | Evaluated at stage completion from `training_board_*.json` |",
        "",
        "---",
        "",
        "## 7.2 Table 1 — Overall Win Rates Across Levels 1–7",
        "",
        "| Level | Reward | Tabular Curr (Train100 / Greedy50) | Hybrid Curr (Train100 / Greedy50) | Δ Curr (Train100) | Tabular Sep (Train100 / Greedy50) | Hybrid Sep (Train100 / Greedy50) | Δ Sep (Train100) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def _get_win_pair(arch: str, store: dict, mode: str, strat: str, lvl: int) -> Tuple[List[float], List[float]]:
        train_wrs = [
            float(df.tail(100)["won"].mean() * 100.0)
            for k, df in store.items()
            if k[0] == mode and k[1] == strat and k[2] == lvl
        ]
        greedy_wrs = []
        for s in all_seeds:
            b = boards.get((arch, mode, strat, s))
            if b:
                stats = b.get("evaluation_summary", {}).get(f"level_{lvl}")
                if stats and "trained_win_rate" in stats:
                    greedy_wrs.append(float(stats["trained_win_rate"]))
        return train_wrs, greedy_wrs

    for lvl in LEVEL_IDS:
        for strat in ("balanced", "dense", "sparse"):
            tc_tr, tc_gr = _get_win_pair("tabular", tabular_store, "curriculum", strat, lvl)
            hc_tr, hc_gr = _get_win_pair("hybrid_expected", hybrid_store, "curriculum", strat, lvl)
            ts_tr, ts_gr = _get_win_pair("tabular", tabular_store, "separate", strat, lvl)
            hs_tr, hs_gr = _get_win_pair("hybrid_expected", hybrid_store, "separate", strat, lvl)

            dc = (float(np.mean(hc_tr)) - float(np.mean(tc_tr))) if (tc_tr and hc_tr) else 0.0
            ds = (float(np.mean(hs_tr)) - float(np.mean(ts_tr))) if (ts_tr and hs_tr) else 0.0
            dc_str = f"{dc:+.1f}%"
            ds_str = f"{ds:+.1f}%"

            def _fmt_pair(tr: List[float], gr: List[float]) -> str:
                tr_s = _fmt_mean_std(tr, 1, "%", "N/A")
                gr_s = _fmt_mean_std(gr, 1, "%", "N/A")
                return f"{tr_s} / {gr_s}" if gr else tr_s

            lines.append(
                f"| Level {lvl} | {STRATEGY_LABELS[strat]} | {_fmt_pair(tc_tr, tc_gr)} | "
                f"{_fmt_pair(hc_tr, hc_gr)} | {dc_str} | {_fmt_pair(ts_tr, ts_gr)} | "
                f"{_fmt_pair(hs_tr, hs_gr)} | {ds_str} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 7.4 Table 2 — Sample Efficiency & Total Compute Cost (Balanced Only, Levels 1–7)",
        "",
        "| Level | Agent & Mode | Conv. (k/N) | Ep to 80% WR | Adaptation Steps | Pre-Training Steps | Global Steps to 80% WR | Total Stage Steps |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    combos = [
        ("Tabular Curriculum", tabular_store, "curriculum"),
        ("Tabular Separate", tabular_store, "separate"),
        ("Hybrid Expected Curriculum", hybrid_store, "curriculum"),
        ("Hybrid Expected Separate", hybrid_store, "separate"),
    ]
    for lvl in LEVEL_IDS:
        for label, store, mode in combos:
            ep_hits, adapt_hits, pre_hits, glob_hits, total_stage_steps = [], [], [], [], []
            n_total = 0
            for s in all_seeds:
                df_lvl = store.get((mode, "balanced", lvl, s))
                if df_lvl is None:
                    continue
                n_total += 1
                total_stage_steps.append(float(df_lvl["steps"].sum()))
                ep_h, st_h = compute_sample_efficiency_milestone(df_lvl, threshold_wr=80.0, window=100)
                if ep_h is not None and st_h is not None:
                    pre_st = get_pretraining_steps(store, mode, "balanced", lvl, s)
                    ep_hits.append(float(ep_h))
                    adapt_hits.append(st_h)
                    pre_hits.append(pre_st)
                    glob_hits.append(pre_st + st_h)

            conv_str = f"{len(ep_hits)}/{n_total}"
            ep_str = _fmt_mean_std(ep_hits, 1)
            ad_str = _fmt_mean_std(adapt_hits, 0)
            pr_str = _fmt_mean_std(pre_hits, 0) if mode == "curriculum" else "0"
            gl_str = _fmt_mean_std(glob_hits, 0)
            tot_str = _fmt_mean_std(total_stage_steps, 0, dnc_label="N/A")
            lines.append(
                f"| Level {lvl} | {label} | {conv_str} | {ep_str} | {ad_str} | {pr_str} | {gl_str} | {tot_str} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 7.5 Table 3 — Failure Modes & Discovered State Counts (Levels 4–7, Balanced Only)",
        "",
        "| Level | Agent & Mode | Win % | Timeout % | Pit Death % | Hazard/Entity Death % | Discovered States |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for lvl in (4, 5, 6, 7):
        for label, arch_key, store, mode in [
            ("Tabular Curriculum", "tabular", tabular_store, "curriculum"),
            ("Tabular Separate", "tabular", tabular_store, "separate"),
            ("Hybrid Expected Curriculum", "hybrid_expected", hybrid_store, "curriculum"),
            ("Hybrid Expected Separate", "hybrid_expected", hybrid_store, "separate"),
        ]:
            win_pcts, to_pcts, pit_pcts, haz_pcts, state_counts = [], [], [], [], []
            for s in all_seeds:
                df_lvl = store.get((mode, "balanced", lvl, s))
                if df_lvl is None or len(df_lvl) == 0:
                    continue
                outcomes = df_lvl["outcome"].astype(str).str.lower()
                n_ep = len(outcomes)
                win_pcts.append(float((outcomes == "win").sum()) / n_ep * 100.0)
                to_pcts.append(float((outcomes == "timeout").sum()) / n_ep * 100.0)
                pit_mask = outcomes.isin(["pit", "pit_death"])
                pit_pcts.append(float(pit_mask.sum()) / n_ep * 100.0)
                haz_pcts.append(float((~outcomes.isin(["win", "timeout", "pit", "pit_death"])).sum()) / n_ep * 100.0)

                b = boards.get((arch_key, mode, "balanced", s))
                if b:
                    st_map = b.get("metadata", {}).get("states_discovered_per_level", {})
                    if f"level_{lvl}" in st_map:
                        state_counts.append(float(st_map[f"level_{lvl}"]))

            st_str = _fmt_mean_std(state_counts, 0, dnc_label="N/A")
            lines.append(
                f"| Level {lvl} | {label} | {_fmt_mean_std(win_pcts, 1, '%')} | "
                f"{_fmt_mean_std(to_pcts, 1, '%')} | {_fmt_mean_std(pit_pcts, 1, '%')} | "
                f"{_fmt_mean_std(haz_pcts, 1, '%')} | {st_str} |"
            )

    save_md.parent.mkdir(parents=True, exist_ok=True)
    save_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [OK] Saved Evaluation Tables: {save_md.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP & SYNC HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_stale_figures(eval_dir: Path, lncs_fig_dir: Optional[Path] = None) -> int:
    """
    Remove legacy figure files (.pdf/.png) whose stem is not in PAPER_FIG_STEMS
    across eval_dir, eval_dir/{tabular,hybrid_expected,comparison}, and lncs_fig_dir.
    Never touches data/*.csv or *.md files.
    """
    removed = 0
    candidate_dirs = [
        eval_dir,
        eval_dir / "tabular",
        eval_dir / "hybrid_expected",
        eval_dir / "comparison",
    ]
    if lncs_fig_dir is not None:
        candidate_dirs.append(lncs_fig_dir)

    for folder in candidate_dirs:
        if not folder.is_dir():
            continue
        for fpath in folder.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in {".pdf", ".png"}:
                if fpath.stem not in PAPER_FIG_STEMS:
                    fpath.unlink()
                    removed += 1

    comp_dir = eval_dir / "comparison"
    if comp_dir.is_dir() and not any(comp_dir.iterdir()):
        comp_dir.rmdir()

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Section 7 Mario RL evaluation figures and tables (Levels 1-7).")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVAL_DIR,
        help="Evaluation root directory (default: evaluation/)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Training outputs directory (default: outputs/)",
    )
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    tabular_store = load_all_split_data(eval_dir / "tabular" / "data")
    hybrid_store = load_all_split_data(eval_dir / "hybrid_expected" / "data")
    print(f"Loaded {len(tabular_store)} tabular slices and {len(hybrid_store)} hybrid_expected slices (Levels 1-7).")

    fig1_pdf = eval_dir / "fig1_reward_trajectories.pdf"
    fig1_png = eval_dir / "fig1_reward_trajectories.png"
    fig2_pdf = eval_dir / "fig2_balanced_overview.pdf"
    fig2_png = eval_dir / "fig2_balanced_overview.png"
    fig3_pdf = eval_dir / "fig3_spotlight_l3_l4_l6.pdf"
    fig3_png = eval_dir / "fig3_spotlight_l3_l4_l6.png"
    fig4_pdf = eval_dir / "fig4_sample_efficiency.pdf"
    fig4_png = eval_dir / "fig4_sample_efficiency.png"
    tables_md = eval_dir / "paper_evaluation_tables.md"

    plot_fig1_reward_trajectories(tabular_store, hybrid_store, fig1_pdf, fig1_png)
    plot_fig2_balanced_overview(tabular_store, hybrid_store, fig2_pdf, fig2_png)
    plot_fig3_spotlight_l3_l4_l6(tabular_store, hybrid_store, fig3_pdf, fig3_png)
    plot_fig4_sample_efficiency(tabular_store, hybrid_store, fig4_pdf, fig4_png)
    generate_paper_evaluation_tables(tabular_store, hybrid_store, tables_md, outputs_dir=args.outputs_dir)

    lncs_fig_dir = PROJECT_ROOT / "springer_lncs" / "figures"
    if lncs_fig_dir.exists():
        for stem in PAPER_FIG_STEMS:
            src_pdf = eval_dir / f"{stem}.pdf"
            if src_pdf.exists():
                shutil.copy(src_pdf, lncs_fig_dir / f"{stem}.pdf")
        print(f"  [OK] Synchronized {len(PAPER_FIG_STEMS)} PDF figures to {lncs_fig_dir}")

    removed = cleanup_stale_figures(eval_dir, lncs_fig_dir if lncs_fig_dir.exists() else None)
    print(f"  [OK] Cleaned {removed} stale figure file(s).")


if __name__ == "__main__":
    main()
