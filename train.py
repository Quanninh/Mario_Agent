import argparse
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from agent import HybridQSARSAAgent, TabularQAgent
from discretizer import MarioStateDiscretizer
from env import create_env
from rewards import create_reward_strategy
from training import Trainer


class ProgressTracker:
    """Zero-dependency ASCII progress tracker with inline or newline output."""

    def __init__(
        self,
        agent_tag: str,
        mode: str,
        reward: str,
        total_episodes: int,
        use_inline: bool,
    ) -> None:
        self.agent_tag = agent_tag
        self.mode = mode
        self.reward = reward
        self.total_episodes = total_episodes
        self.use_inline = use_inline
        self.start_time = time.time()
        self.last_update_time = 0.0

    def callback(self, data: Dict[str, Any]) -> None:
        current_ep = data["global_episode"]
        total_ep = data["total_episodes"]
        lvl_ep = data["level_episode"]
        ep_per_lvl = data["episodes_per_level"]
        elapsed = time.time() - self.start_time

        # Calculate ETA
        if current_ep > 0:
            rate = elapsed / current_ep
            rem_seconds = int(rate * (total_ep - current_ep))
            eta_str = f"{rem_seconds}s" if rem_seconds < 60 else f"{rem_seconds // 60}m {rem_seconds % 60}s"
        else:
            eta_str = "--"

        is_level_end = lvl_ep == ep_per_lvl

        if self.use_inline:
            # Update inline every 10 episodes or at level completion
            if current_ep % 10 == 0 or is_level_end or current_ep == 1:
                pct = current_ep / max(1, total_ep)
                bar_len = 24
                filled = int(round(bar_len * pct))
                filled = min(bar_len, max(0, filled))
                bar = "#" * filled + "-" * (bar_len - filled)
                pct_str = f"{pct * 100.0:5.1f}%"

                line = (
                    f"\r[{bar}] {pct_str} | {self.agent_tag} | {self.mode.capitalize()} | {self.reward.upper()} | "
                    f"Lvl {data['level_id']} ({lvl_ep}/{ep_per_lvl}) | "
                    f"Win50: {data['rolling_win_rate_50']:5.1f}% | "
                    f"R: {data['total_reward']:+7.1f} | eps: {data['epsilon']:.3f} | "
                    f"States: {data['states_discovered']:4d}/1152 | ETA: {eta_str}"
                )
                sys.stdout.write(line)
                if is_level_end:
                    sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            # Clean newline logs every 100 episodes, episode 1, or level completion
            if current_ep == 1 or current_ep % 100 == 0 or is_level_end:
                print(
                    f"[{self.agent_tag.upper()} | {self.mode.capitalize()} | {self.reward.upper()}] "
                    f"Global Ep {current_ep:4d}/{total_ep:4d} | "
                    f"Lvl {data['level_id']} ({lvl_ep:4d}/{ep_per_lvl:4d}) | "
                    f"Win50: {data['rolling_win_rate_50']:5.1f}% | "
                    f"R: {data['total_reward']:+7.1f} | eps: {data['epsilon']:.3f} | "
                    f"States: {data['states_discovered']:4d}/1152 | ETA: {eta_str}"
                )
                sys.stdout.flush()


def print_comparison_table(all_results: Dict[Tuple[str, str, str], Dict], levels: List[int]) -> None:
    """Prints a clear comparison table across all completed (agent, mode, reward) experiments."""
    print("\n" + "=" * 135)
    print(" " * 48 + "EXPERIMENT EVALUATION COMPARISON SUMMARY")
    print("=" * 135)
    header = (
        f"{'Agent':<16} {'Mode':<12} {'Reward':<10} {'Level':<14} {'Untrained Win%':<16} "
        f"{'Trained Win%':<14} {'Untrained R':<14} {'Trained R':<14} {'Clear Steps':<14} {'Items'}"
    )
    print(header)
    print("-" * 135)

    for (agent_tag, mode, reward_name), res in all_results.items():
        summary = res.get("evaluation_summary", {})
        for lvl in levels:
            lvl_key = f"level_{lvl}"
            stats = summary.get(lvl_key, {})

            untrained_win = f"{stats.get('untrained_win_rate', 0.0):.1f}%"
            trained_win = f"{stats.get('trained_win_rate', 0.0):.1f}%"
            untrained_r = f"{stats.get('untrained_avg_reward', 0.0):+.1f}"
            trained_r = f"{stats.get('trained_avg_reward', 0.0):+.1f}"

            clear_steps_val = stats.get("avg_completion_steps", -1.0)
            clear_steps = f"{clear_steps_val:.1f}" if clear_steps_val > 0 else "N/A"
            items = f"{stats.get('avg_items_collected', 0.0):.1f}"

            print(
                f"{agent_tag:<16} {mode:<12} {reward_name:<10} {f'Level {lvl}':<14} {untrained_win:<16} "
                f"{trained_win:<14} {untrained_r:<14} {trained_r:<14} {clear_steps:<14} {items}"
            )

            # Print zero-shot notes for curriculum
            if mode == "curriculum":
                if lvl == 2 and "level_3_zero_shot" in summary:
                    z3 = summary["level_3_zero_shot"]
                    z3_win = f"{z3.get('win_rate', 0.0):.1f}%"
                    z3_r = f"{z3.get('avg_reward', 0.0):+.1f}"
                    z3_items = f"{z3.get('avg_items_collected', 0.0):.1f}"
                    print(
                        f"{'':<16} {'':<12} {'':<10} {'  -> L3 ZeroShot':<14} {'--':<16} "
                        f"{z3_win:<14} {'--':<14} {z3_r:<14} {'--':<14} {z3_items}"
                    )
                elif lvl == 4 and "level_5_zero_shot" in summary:
                    z5 = summary["level_5_zero_shot"]
                    z5_win = f"{z5.get('win_rate', 0.0):.1f}%"
                    z5_r = f"{z5.get('avg_reward', 0.0):+.1f}"
                    z5_items = f"{z5.get('avg_items_collected', 0.0):.1f}"
                    print(
                        f"{'':<16} {'':<12} {'':<10} {'  -> L5 ZeroShot':<14} {'--':<16} "
                        f"{z5_win:<14} {'--':<14} {z5_r:<14} {'--':<14} {z5_items}"
                    )
        print("-" * 135)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless RL Training & Evaluation Harness for Hoá's Midnight Escape"
    )
    parser.add_argument(
        "--agent",
        choices=["tabular", "hybrid", "all"],
        default="all",
        help="Agent algorithm family: 'tabular', 'hybrid', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--hybrid-mode",
        choices=["expected", "linear", "hazard_gated"],
        default="expected",
        help="Hybrid SARSA-Q target formulation: 'expected', 'linear', or 'hazard_gated' (default: 'expected')",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="SARSA blending factor beta for Hybrid agent (default: 0.25)",
    )
    parser.add_argument(
        "--mode",
        choices=["curriculum", "separate", "all"],
        default="all",
        help="Training mode: 'curriculum', 'separate', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--reward",
        choices=["balanced", "sparse", "dense", "all"],
        default="all",
        help="Reward strategy profile: 'balanced', 'sparse', 'dense', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Levels to train/evaluate on (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes per level (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across runs (default: 42)",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Action repeat ticks per decision step (default: 1)",
    )
    parser.add_argument(
        "--rewarm-epsilon",
        type=float,
        default=0.30,
        help="Exploration re-warming epsilon when entering new level in curriculum (default: 0.30)",
    )
    parser.add_argument(
        "--rewarm-epsilon-ghost",
        type=float,
        default=0.35,
        help="Exploration re-warming epsilon when entering ghost levels >=4 in curriculum (default: 0.35)",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.35,
        help="Probability of interleaving background replay from earlier levels in curriculum (default: 0.35)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.12,
        help="Starting Q-learning learning rate alpha (default: 0.12)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.02,
        help="Minimum Q-learning learning rate alpha after decay (default: 0.02)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon exploration rate (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Minimum epsilon exploration rate (default: 0.01)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.997,
        help="Per-episode epsilon decay multiplier (default: 0.997)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Interval of episodes between greedy evaluations for best model checkpointing (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=400,
        help="Interval of episodes between periodic checkpoints (default: 400)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory to save Q-table checkpoints (default: 'checkpoints')",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory to save JSON/CSV Member 4 data boards (default: 'outputs')",
    )
    parser.add_argument(
        "--map-seed",
        type=int,
        default=None,
        help=(
            "Starting procedural map seed for Levels 1-7 "
            "(default: None = fixed handcrafted maps; defaults to --seed if --map-seed-interval > 0)"
        ),
    )
    parser.add_argument(
        "--map-seed-interval",
        type=int,
        default=0,
        help=(
            "Rotate to a new random procedural map seed every N episodes per level for Levels 1-7 "
            "(e.g. 3000 or 5000; 0 disables rotation)"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast verification flag (5 eps/lvl, eval_interval=2, completes in seconds)",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable in-place carriage return progress updates and use plain newline logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Smoke-test parameter overrides
    if args.smoke_test:
        print("[train.py] Smoke-test mode active: overriding parameters for fast verification.")
        episodes_per_level = 5
        eval_interval = 2
        eval_episodes_periodic = 2
        final_eval_episodes = 3
        untrained_eval_episodes = 2
        checkpoint_interval = 5
    else:
        episodes_per_level = args.episodes
        eval_interval = args.eval_interval
        eval_episodes_periodic = 10
        final_eval_episodes = 50
        untrained_eval_episodes = 30
        checkpoint_interval = args.checkpoint_interval

    # Determine experiment grid
    agents = ["tabular", "hybrid"] if args.agent == "all" else [args.agent]
    modes = ["curriculum", "separate"] if args.mode == "all" else [args.mode]
    rewards = ["balanced", "sparse", "dense"] if args.reward == "all" else [args.reward]

    total_experiments = len(agents) * len(modes) * len(rewards)
    print(
        f"[train.py] Executing {total_experiments} experiment(s): "
        f"Agents={agents}, Modes={modes}, Rewards={rewards}, Levels={args.levels}, "
        f"MapSeed={args.map_seed if args.map_seed is not None else (f'random(seed={args.seed})' if args.map_seed_interval > 0 else None)}, "
        f"MapRotateInterval={args.map_seed_interval}"
    )
    if (args.map_seed is not None or args.map_seed_interval > 0) and any(lvl > 7 for lvl in args.levels):
        print("[train.py] Note: Procedural map seeds apply to Levels 1-7; Level 8 remains a fixed handcrafted layout.")

    use_inline = sys.stdout.isatty() and not args.no_progress_bar

    all_results: Dict[Tuple[str, str, str], Dict] = {}
    exp_idx = 0

    for agent_type in agents:
        agent_tag = "tabular" if agent_type == "tabular" else f"hybrid_{args.hybrid_mode}"
        for mode in modes:
            for reward_name in rewards:
                exp_idx += 1
                print(f"\n{'='*75}")
                print(
                    f"[{exp_idx}/{total_experiments}] Starting: Agent={agent_tag.upper()} | "
                    f"Mode={mode.upper()} | Reward={reward_name.upper()}"
                )
                print(f"{'='*75}")

                # Deterministic per-experiment seed
                random.seed(args.seed)
                np.random.seed(args.seed)

                # Factory for agents with exact seed and hyperparameters
                if agent_type == "tabular":
                    def make_agent() -> TabularQAgent:
                        return TabularQAgent(
                            learning_rate=args.alpha,
                            discount_factor=args.gamma,
                            epsilon=args.epsilon_start,
                            epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay,
                            seed=args.seed,
                        )
                else:
                    def make_agent() -> HybridQSARSAAgent:
                        return HybridQSARSAAgent(
                            learning_rate=args.alpha,
                            discount_factor=args.gamma,
                            epsilon=args.epsilon_start,
                            epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay,
                            mode=args.hybrid_mode,
                            beta=args.beta,
                            seed=args.seed,
                        )

                reward_strategy = create_reward_strategy(reward_name)
                discretizer = MarioStateDiscretizer()

                trainer = Trainer(
                    env_factory=create_env,
                    discretizer=discretizer,
                    reward_strategy=reward_strategy,
                    reward_name=reward_name,
                    agent_type=agent_type,
                    hybrid_mode=args.hybrid_mode,
                    beta=args.beta,
                    action_repeat=args.action_repeat,
                    rewarm_epsilon=args.rewarm_epsilon,
                    rewarm_epsilon_ghost=args.rewarm_epsilon_ghost,
                    alpha_min=args.alpha_min,
                    replay_ratio=args.replay_ratio,
                    eval_interval=eval_interval,
                    eval_episodes_periodic=eval_episodes_periodic,
                    final_eval_episodes=final_eval_episodes,
                    untrained_eval_episodes=untrained_eval_episodes,
                    checkpoint_interval=checkpoint_interval,
                    checkpoints_dir=args.checkpoints_dir,
                    outputs_dir=args.outputs_dir,
                    map_seed=args.map_seed,
                    map_seed_interval=args.map_seed_interval,
                    seed=args.seed,
                )

                total_eps = len(args.levels) * episodes_per_level
                tracker = ProgressTracker(
                    agent_tag=agent_tag,
                    mode=mode,
                    reward=reward_name,
                    total_episodes=total_eps,
                    use_inline=use_inline,
                )

                is_canonical = (
                    mode == "curriculum"
                    and reward_name == "balanced"
                    and agent_tag == "tabular"
                )

                result = trainer.run_experiment(
                    agent_factory=make_agent,
                    mode=mode,
                    levels=args.levels,
                    episodes_per_level=episodes_per_level,
                    is_canonical_production=is_canonical,
                    progress_callback=tracker.callback,
                )
                all_results[(agent_tag, mode, reward_name)] = result

    # Print summary comparison table
    print_comparison_table(all_results, args.levels)
    print("\n[train.py] All experiments finished successfully!")


if __name__ == "__main__":
    main()
