"""
Mario RL Agent — Evaluation Graph Generator
=============================================
Generates a mock training log CSV (100 episodes) and produces 3
academic-grade evaluation figures for the technical report.

Dependencies: numpy, pandas, matplotlib, seaborn
Usage:        python evaluation/generate_evaluation_graphs.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ──────────────────────────────────────────────
# 0.  Paths
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "mario_training_log.csv")
GRAPH_1_PATH = os.path.join(SCRIPT_DIR, "graph_1_curriculum_learning_curve.pdf")
GRAPH_2_PATH = os.path.join(SCRIPT_DIR, "graph_2_reward_profiling_benchmark.pdf")
GRAPH_3_PATH = os.path.join(SCRIPT_DIR, "graph_3_playstyle_matrix.pdf")

# ──────────────────────────────────────────────
# 1.  Academic Plot Configuration
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────
# 2.  Mock Data Generation
# ──────────────────────────────────────────────
def generate_mock_data(n_episodes: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible mock training log with realistic trends."""
    np.random.seed(seed)

    episodes = np.arange(1, n_episodes + 1)
    progress = (episodes - 1) / (n_episodes - 1)          # 0 → 1

    # --- training_mode ---
    # First 50: 70 % train, last 50: 85 % eval
    mode_first = np.random.choice(
        ["train", "eval"], size=50, p=[0.70, 0.30]
    )
    mode_last = np.random.choice(
        ["train", "eval"], size=50, p=[0.15, 0.85]
    )
    training_mode = np.concatenate([mode_first, mode_last])

    # --- reward_profile ---
    reward_profile = np.random.choice(
        ["Sparse", "Dense", "Balanced"], size=n_episodes
    )

    # --- level (cyclic then shuffled within thirds) ---
    levels_base = np.tile(["Level_1", "Level_2", "Level_3"],
                          n_episodes // 3 + 1)[:n_episodes]
    # Shuffle within each third to mix order while keeping balanced counts
    for start in range(0, n_episodes, n_episodes // 3):
        end = min(start + n_episodes // 3, n_episodes)
        np.random.shuffle(levels_base[start:end])
    level = levels_base

    # --- epsilon (exponential decay 1.0 → 0.01) ---
    epsilon = 1.0 * (0.01 / 1.0) ** progress

    # --- rolling_win_rate_50 (sigmoid-shaped with per-level offsets) ---
    # Base sigmoid: steep rise centered around episode 50
    base_win_rate = 95.0 / (1.0 + np.exp(-0.10 * (episodes - 45)))
    level_offset = np.where(
        level == "Level_1",  8.0,
        np.where(level == "Level_2", 0.0, -10.0)
    )
    noise_wr = np.random.normal(0, 3, size=n_episodes)
    rolling_win_rate_50 = np.clip(base_win_rate + level_offset + noise_wr,
                                  0, 100)

    # --- total_reward (linear trend with per-profile multiplier) ---
    base_reward = 50 + 450 * progress
    profile_mult = np.where(
        reward_profile == "Dense",    1.3,
        np.where(reward_profile == "Balanced", 1.0, 0.7)
    )
    noise_rw = np.random.normal(0, 15, size=n_episodes)
    total_reward = np.round(base_reward * profile_mult + noise_rw, 2)

    # --- steps (high → low) ---
    base_steps = 800 - 600 * progress
    noise_st = np.random.normal(0, 30, size=n_episodes)
    steps = np.clip(base_steps + noise_st, 80, 1000).astype(int)

    # --- stomps ---
    stomps = np.random.randint(0, 6, size=n_episodes)

    # --- coins ---
    coins = np.array([np.random.randint(0, 3 + ep // 10) for ep in episodes])

    # --- won (probability scales with episode + level adjustment) ---
    p_win = 0.10 + 0.80 * progress
    level_p_adj = np.where(
        level == "Level_1",  0.08,
        np.where(level == "Level_2", 0.0, -0.10)
    )
    p_win_adj = np.clip(p_win + level_p_adj, 0, 1)
    won = (np.random.rand(n_episodes) < p_win_adj).astype(int)

    df = pd.DataFrame({
        "episode": episodes,
        "training_mode": training_mode,
        "reward_profile": reward_profile,
        "level": level,
        "epsilon": np.round(epsilon, 6),
        "rolling_win_rate_50": np.round(rolling_win_rate_50, 2),
        "total_reward": total_reward,
        "steps": steps,
        "stomps": stomps,
        "coins": coins,
        "won": won,
    })
    return df


# ──────────────────────────────────────────────
# 3.  Graph 1 — Curriculum Learning Curve
# ──────────────────────────────────────────────
def plot_curriculum_learning_curve(df: pd.DataFrame, save_path: str) -> None:
    """Win-rate per level with epsilon decay on a secondary axis."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Primary axis — win rate by level
    palette = sns.color_palette("viridis", n_colors=3)
    sns.lineplot(
        data=df, x="episode", y="rolling_win_rate_50",
        hue="level", hue_order=["Level_1", "Level_2", "Level_3"],
        palette=palette, linewidth=1.8, ax=ax1,
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Rolling Win Rate (%)")
    ax1.set_title("Curriculum Learning Curve (Generalization Benchmark)")
    ax1.set_ylim(-5, 110)

    # Secondary axis — epsilon
    ax2 = ax1.twinx()
    ax2.plot(
        df["episode"], df["epsilon"],
        color="grey", linestyle="--", linewidth=1.5, alpha=0.7,
        label="ε (Exploration Rate)",
    )
    ax2.set_ylabel("Epsilon (ε)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(-0.05, 1.10)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax1.legend(
        handles1 + handles2, labels1 + labels2,
        loc="center right", framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 4.  Graph 2 — Reward Profiling Benchmark
# ──────────────────────────────────────────────
def plot_reward_profiling(df: pd.DataFrame, save_path: str) -> None:
    """Total reward by reward profile (eval runs only)."""
    df_eval = df[df["training_mode"] == "eval"].copy()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        data=df_eval, x="episode", y="total_reward",
        hue="reward_profile",
        hue_order=["Dense", "Balanced", "Sparse"],
        palette="deep", linewidth=1.8, ax=ax,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Profiling Benchmark (Eval Runs Only)")
    ax.legend(title="Reward Profile", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 5.  Graph 3 — Playstyle Matrix
# ──────────────────────────────────────────────
def plot_playstyle_matrix(df: pd.DataFrame, save_path: str) -> None:
    """Scatter of steps vs stomps for the last 50 episodes."""
    df_last50 = df[df["episode"] > 50].copy()
    df_last50["outcome"] = df_last50["won"].map(
        {1: "Victory", 0: "Death/Failure"}
    )

    custom_palette = {"Victory": "#2196F3", "Death/Failure": "#F44336"}

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df_last50, x="steps", y="stomps",
        hue="outcome", style="outcome",
        hue_order=["Victory", "Death/Failure"],
        palette=custom_palette,
        markers={"Victory": "o", "Death/Failure": "X"},
        s=100, alpha=0.8, edgecolor="black", linewidth=0.6,
        ax=ax,
    )
    ax.set_xlabel("Steps to Finish (Lower = Faster)")
    ax.set_ylabel("Enemies Defeated (Stomps)")
    ax.set_title("Playstyle Matrix (Converged Agent Identity, Last 50 Episodes)")
    ax.legend(title="Outcome", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  Mario RL — Evaluation Graph Generator")
    print("=" * 55)

    # Phase 1: Generate CSV
    print("\n[Phase 1] Generating mock training log …")
    df = generate_mock_data()
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✓ Saved: {CSV_PATH}")
    print(f"    Shape: {df.shape}  |  Columns: {list(df.columns)}")

    # Phase 2: Generate Graphs
    print("\n[Phase 2] Generating evaluation graphs …")
    plot_curriculum_learning_curve(df, GRAPH_1_PATH)
    plot_reward_profiling(df, GRAPH_2_PATH)
    plot_playstyle_matrix(df, GRAPH_3_PATH)

    print("\n" + "=" * 55)
    print("  Done! All files saved to:", SCRIPT_DIR)
    print("=" * 55)


if __name__ == "__main__":
    main()
