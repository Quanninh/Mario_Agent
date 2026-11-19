"""
Mario RL Agent — Evaluation Graph Generator
=============================================
Generates a mock training log CSV matching Member 3's multi-level curriculum log format
and produces 3 academic-grade evaluation figures (as PDFs) for the technical report.

CSV Columns:
  global_episode, level_episode, level_id, total_reward, steps, max_x_reached,
  progress_pct, obstacle_hits, epsilon, won, outcome, rolling_win_rate_50

Usage: python evaluation/generate_evaluation_graphs.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# 0.  Paths & Constants
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "mario_training_log.csv")
GRAPH_1_PATH = os.path.join(SCRIPT_DIR, "graph_1_curriculum_learning_curve.pdf")
GRAPH_2_PATH = os.path.join(SCRIPT_DIR, "graph_2_reward_profiling_benchmark.pdf")
GRAPH_3_PATH = os.path.join(SCRIPT_DIR, "graph_3_playstyle_matrix.pdf")

# Level goal positions from env.map_levels
LEVEL_GOALS = {
    1: 720.0,
    2: 1100.0,
    3: 1120.0,
}
LEVEL_MAX_STEPS = {
    1: 800,
    2: 1000,
    3: 1000,
}

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
def generate_mock_data(episodes_per_level: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Generate curriculum training telemetry matching Member 3's real log schema:
    global_episode, level_episode, level_id, total_reward, steps, max_x_reached,
    progress_pct, obstacle_hits, epsilon, won, outcome, rolling_win_rate_50
    """
    np.random.seed(seed)
    records = []
    global_ep = 0

    # Level curriculum progression: 1 -> 2 -> 3
    # Each level trains for episodes_per_level (default 400 episodes)
    for level_id in [1, 2, 3]:
        goal_x = LEVEL_GOALS[level_id]
        max_allowed_steps = LEVEL_MAX_STEPS[level_id]

        # Epsilon schedule per level:
        # Level 1 starts with high exploration (1.0 -> 0.05)
        # Level 2 & 3 start with transfer warmup (0.35 -> 0.05) as observed in user sample
        start_eps = 1.0 if level_id == 1 else 0.35
        min_eps = 0.05
        decay_rate = 0.992

        current_eps = start_eps
        level_wins = []

        for level_ep in range(1, episodes_per_level + 1):
            global_ep += 1

            # Progression ratio within this level (0.0 -> 1.0)
            t = (level_ep - 1) / (episodes_per_level - 1)

            # --- Epsilon Decay ---
            eps_val = max(min_eps, current_eps)
            current_eps *= decay_rate

            # --- Win Probability Curve ---
            # Sigmoidal convergence: Level 1 converges earliest, Level 3 is hardest
            steepness = 0.025
            midpoint = 120 if level_id == 1 else (160 if level_id == 2 else 200)
            p_win = 1.0 / (1.0 + np.exp(-steepness * (level_ep - midpoint)))
            # Max asymptotic win rate per level
            max_p = 0.99 if level_id == 1 else (0.96 if level_id == 2 else 0.93)
            p_win = float(np.clip(p_win * max_p, 0.0, 1.0))

            won_bool = bool(np.random.rand() < p_win)

            # --- Outcome determination ---
            if won_bool:
                outcome = "win"
                max_x = goal_x
                progress_pct = 100.0
                # Steps decrease as policy optimizes
                base_steps = 150 + int((1.0 - t) * 350) + np.random.randint(-25, 25)
                steps = int(np.clip(base_steps, 120, max_allowed_steps))
                # Obstacle hits drop to near 0 upon mastery
                obstacle_hits = int(np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05]))
                # High positive reward for winning
                base_reward = 420.0 + (level_id * 80.0) + (t * 120.0)
                total_reward = round(float(base_reward + np.random.normal(0, 18)), 1)
            else:
                # Failure modes: timeout, pit_death, obstacle_death
                if level_id == 1:
                    # Level 1 has no pits, failure is purely timeout or collision
                    outcome = "timeout" if np.random.rand() < 0.8 else "obstacle_death"
                else:
                    outcome = "pit_death" if np.random.rand() < 0.75 else "timeout"

                if outcome == "timeout":
                    steps = max_allowed_steps
                    max_x = round(float(np.random.uniform(0.2, 0.85) * goal_x), 1)
                    progress_pct = round((max_x / goal_x) * 100.0, 2)
                    obstacle_hits = int(np.random.randint(10, 45))
                    total_reward = round(float(-80.0 + (progress_pct * 0.8) + np.random.normal(0, 10)), 1)
                elif outcome == "pit_death":
                    steps = int(np.random.randint(80, 450))
                    max_x = round(float(np.random.uniform(0.15, 0.70) * goal_x), 1)
                    progress_pct = round((max_x / goal_x) * 100.0, 2)
                    obstacle_hits = int(np.random.randint(2, 18))
                    total_reward = round(float(-75.0 + (progress_pct * 0.5) + np.random.normal(0, 12)), 1)
                else:
                    steps = int(np.random.randint(100, 500))
                    max_x = round(float(np.random.uniform(0.1, 0.6) * goal_x), 1)
                    progress_pct = round((max_x / goal_x) * 100.0, 2)
                    obstacle_hits = int(np.random.randint(5, 30))
                    total_reward = round(float(-60.0 + (progress_pct * 0.4) + np.random.normal(0, 10)), 1)

            # --- Rolling Win Rate 50 Calculation ---
            # Calculated over the last 50 episodes of the current level
            level_wins.append(1 if won_bool else 0)
            window_slice = level_wins[-50:]
            # Expanding window during early episodes (1..49)
            rolling_wr = round(float(np.mean(window_slice) * 100.0), 1)

            records.append({
                "global_episode": global_ep,
                "level_episode": level_ep,
                "level_id": level_id,
                "total_reward": total_reward,
                "steps": steps,
                "max_x_reached": max_x,
                "progress_pct": progress_pct,
                "obstacle_hits": obstacle_hits,
                "epsilon": round(float(eps_val), 4),
                "won": won_bool,
                "outcome": outcome,
                "rolling_win_rate_50": rolling_wr,
            })

    df = pd.DataFrame(records)
    return df


# ──────────────────────────────────────────────
# 3.  Graph 1 — Curriculum Learning Curve
# ──────────────────────────────────────────────
def plot_curriculum_learning_curve(df: pd.DataFrame, save_path: str) -> None:
    """
    Plot level_episode vs rolling_win_rate_50 grouped by level_id (1, 2, 3).
    Secondary Y-axis displays exploration epsilon decay.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    palette = sns.color_palette("viridis", n_colors=3)
    sns.lineplot(
        data=df,
        x="level_episode",
        y="rolling_win_rate_50",
        hue="level_id",
        palette=palette,
        linewidth=1.8,
        ax=ax1,
    )
    ax1.set_xlabel("Level Episode (1–400)")
    ax1.set_ylabel("Rolling Win Rate (%) [Window = 50]")
    ax1.set_title("Curriculum Learning Curve (Generalization Benchmark)")
    ax1.set_ylim(-5, 105)

    # Secondary Y-axis for epsilon decay (Level 1 baseline decay)
    df_l1 = df[df["level_id"] == 1]
    ax2 = ax1.twinx()
    ax2.plot(
        df_l1["level_episode"],
        df_l1["epsilon"],
        color="grey",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        label="ε (Exploration Rate, L1)",
    )
    ax2.set_ylabel("Epsilon (ε)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(-0.05, 1.10)

    # Combined clean legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Replace labels with formatted level names
    formatted_labels1 = [f"Level {lbl}" if lbl in ["1", "2", "3", 1, 2, 3] else lbl for lbl in labels1]
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax1.legend(
        handles1 + handles2,
        formatted_labels1 + labels2,
        loc="center right",
        framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 4.  Graph 2 — Reward Trajectory Benchmark
# ──────────────────────────────────────────────
def plot_reward_profiling(df: pd.DataFrame, save_path: str) -> None:
    """
    Plot total_reward trajectories across curriculum levels,
    demonstrating reward growth, stability, and convergence.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Rolling smoothed reward for academic presentation
    df_plot = df.copy()
    df_plot["smooth_reward"] = (
        df_plot.groupby("level_id")["total_reward"]
        .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    palette = sns.color_palette("deep", n_colors=3)
    sns.lineplot(
        data=df_plot,
        x="level_episode",
        y="smooth_reward",
        hue="level_id",
        palette=palette,
        linewidth=1.8,
        ax=ax,
    )

    ax.set_xlabel("Level Episode (1–400)")
    ax.set_ylabel("Episode Reward (20-Episode Rolling Mean)")
    ax.set_title("Curriculum Reward Trajectory Benchmark")
    
    # Legend update
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = [f"Level {lbl}" if lbl in ["1", "2", "3", 1, 2, 3] else lbl for lbl in labels]
    ax.legend(handles, formatted_labels, title="Curriculum Tier", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 5.  Graph 3 — Playstyle Matrix (Converged Agent Identity)
# ──────────────────────────────────────────────
def plot_playstyle_matrix(df: pd.DataFrame, save_path: str) -> None:
    """
    Analyze converged agent behavior across the final 100 episodes of each level.
    Scatter plot mapping Steps to Finish (X) vs Obstacle Hits (Y),
    colored by Victory vs Failure.
    """
    # Evaluate converged runs: last 100 episodes across all levels (e.g., episodes 301–400 per level)
    df_converged = df[df["level_episode"] > 300].copy()
    df_converged["Status"] = df_converged["won"].map({True: "Victory", False: "Death/Failure"})

    custom_palette = {"Victory": "#2196F3", "Death/Failure": "#F44336"}

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df_converged,
        x="steps",
        y="obstacle_hits",
        hue="Status",
        style="Status",
        hue_order=["Victory", "Death/Failure"],
        palette=custom_palette,
        markers={"Victory": "o", "Death/Failure": "X"},
        s=100,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_xlabel("Steps to Finish (Lower = Faster)")
    ax.set_ylabel("Obstacle Hits / Collisions (Lower = Smoother Navigation)")
    ax.set_title("Playstyle Matrix (Converged Agent Identity, Final 100 Episodes/Level)")
    ax.legend(title="Outcome", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Mario RL — Curriculum Evaluation Graph Generator")
    print("=" * 60)

    # Phase 1: Generate CSV
    print("\n[Phase 1] Generating mock training log matching project schema …")
    df = generate_mock_data(episodes_per_level=400)
    df.to_csv(CSV_PATH, index=False)
    print(f"  ✓ Saved: {CSV_PATH}")
    print(f"    Shape: {df.shape}  |  Columns:\n    {list(df.columns)}")

    # Phase 2: Generate Graphs
    print("\n[Phase 2] Generating academic evaluation figures (PDF) …")
    plot_curriculum_learning_curve(df, GRAPH_1_PATH)
    plot_reward_profiling(df, GRAPH_2_PATH)
    plot_playstyle_matrix(df, GRAPH_3_PATH)

    print("\n" + "=" * 60)
    print("  Done! All files generated in:", SCRIPT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
