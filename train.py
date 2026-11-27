import argparse
from pathlib import Path
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
from training.trainer import DEFAULT_EPISODE_SCHEDULE


class ProgressTracker:
    """Zero-dependency ASCII progress tracker with inline or newline output."""

    def __init__(
        self,
        agent_tag: str,
        mode: str,
        reward: str,
        total_episodes: int,
        use_inline: bool,
        state_space_max: int = 110592,
    ) -> None:
        self.agent_tag = agent_tag
        self.mode = mode
        self.reward = reward
        self.total_episodes = total_episodes
        self.use_inline = use_inline
        self.state_space_max = state_space_max
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
                    f"States: {data['states_discovered']:5d}/{self.state_space_max} | ETA: {eta_str}"
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
                    f"Global Ep {current_ep:5d}/{total_ep:5d} | "
                    f"Lvl {data['level_id']} ({lvl_ep:4d}/{ep_per_lvl:4d}) | "
                    f"Win50: {data['rolling_win_rate_50']:5.1f}% | "
                    f"R: {data['total_reward']:+7.1f} | eps: {data['epsilon']:.3f} | "
                    f"States: {data['states_discovered']:5d}/{self.state_space_max} | ETA: {eta_str}"
                )
                sys.stdout.flush()


def print_comparison_table(all_results: Dict[Tuple[str, str, str], Dict], levels: List[int]) -> None:
    """Prints a clear comparison table across all completed (agent, mode, reward) experiments."""
    print("\n" + "=" * 150)
    print(" " * 55 + "EXPERIMENT EVALUATION COMPARISON SUMMARY")
    print("=" * 150)
    header = (
        f"{'Agent':<16} {'Mode':<11} {'Reward':<9} {'Level':<14} {'Untr%':<7} "
        f"{'Final%':<8} {'BestCP%':<8} {'Untr R':<9} {'Final R':<10} {'Best R':<10} {'Steps':<8} {'BestSt':<8} {'Items'}"
    )
    print(header)
    print("-" * 150)

    for (agent_tag, mode, reward_name), res in all_results.items():
        summary = res.get("evaluation_summary", {})
        for lvl in levels:
            lvl_key = f"level_{lvl}"
            stats = summary.get(lvl_key, {})

            untrained_win = f"{stats.get('untrained_win_rate', 0.0):.1f}%"
            trained_win = f"{stats.get('trained_win_rate', 0.0):.1f}%"
            best_win = f"{stats.get('best_checkpoint_win_rate', stats.get('trained_win_rate', 0.0)):.1f}%"
            untrained_r = f"{stats.get('untrained_avg_reward', 0.0):+.1f}"
            trained_r = f"{stats.get('trained_avg_reward', 0.0):+.1f}"
            best_r = f"{stats.get('best_checkpoint_avg_reward', stats.get('trained_avg_reward', 0.0)):+.1f}"

            clear_steps_val = stats.get("avg_completion_steps", -1.0)
            clear_steps = f"{clear_steps_val:.1f}" if clear_steps_val > 0 else "N/A"
            best_steps_val = stats.get("best_checkpoint_completion_steps", clear_steps_val)
            best_steps = f"{best_steps_val:.1f}" if best_steps_val > 0 else "N/A"
            items = f"{stats.get('avg_items_collected', 0.0):.1f}"

            # Print zero-shot notes dynamically before each target level in curriculum order
            zs_key = f"level_{lvl}_zero_shot"
            if mode == "curriculum" and zs_key in summary:
                zs = summary[zs_key]
                zs_win = f"{zs.get('win_rate', 0.0):.1f}%"
                zs_r = f"{zs.get('avg_reward', 0.0):+.1f}"
                zs_items = f"{zs.get('avg_items_collected', 0.0):.1f}"
                print(
                    f"{'':<16} {'':<11} {'':<9} {f'  -> L{lvl} ZeroShot':<14} {'--':<7} "
                    f"{zs_win:<8} {'--':<8} {'--':<9} {zs_r:<10} {'--':<10} {'--':<8} {'--':<8} {zs_items}"
                )

            print(
                f"{agent_tag:<16} {mode:<11} {reward_name:<9} {f'Level {lvl}':<14} {untrained_win:<7} "
                f"{trained_win:<8} {best_win:<8} {untrained_r:<9} {trained_r:<10} {best_r:<10} "
                f"{clear_steps:<8} {best_steps:<8} {items}"
            )
        print("-" * 150)


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
        default=[1, 2, 3, 4, 6, 5],
        help="Levels to train/evaluate on in execution order (default: 1 2 3 4 6 5)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Flat training episodes per level; omit to use stage-scaled schedule (800/1200/1800/7000/3500/30000)",
    )
    parser.add_argument(
        "--l5-episodes",
        type=int,
        default=None,
        help="Explicit episode budget override for Level 5 (e.g., 30000 or 50000)",
    )
    parser.add_argument(
        "--l6-episodes",
        type=int,
        default=None,
        help="Explicit episode budget override for Level 6 (default: 3500)",
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
        "--max-steps-cap",
        type=int,
        default=2400,
        help="Trainer-side episode step cap; pass 0 for uncapped (default: 2400)",
    )
    parser.add_argument(
        "--rewarm-epsilon",
        type=float,
        default=0.30,
        help="Exploration re-warming epsilon when entering Level 2 in curriculum (default: 0.30)",
    )
    parser.add_argument(
        "--rewarm-epsilon-ghost",
        type=float,
        default=0.35,
        help="Exploration re-warming epsilon when entering Level 3 in curriculum (default: 0.35)",
    )
    parser.add_argument(
        "--rewarm-epsilon-hard",
        type=float,
        default=0.45,
        help="Exploration re-warming epsilon when entering Level 4 in curriculum (default: 0.45)",
    )
    parser.add_argument(
        "--rewarm-epsilon-l5",
        type=float,
        default=0.40,
        help="Exploration re-warming epsilon when entering Level 5 in curriculum (default: 0.40)",
    )
    parser.add_argument(
        "--rewarm-epsilon-l6",
        type=float,
        default=0.40,
        help="Exploration re-warming epsilon when entering Level 6 in curriculum (default: 0.40)",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.35,
        help="Probability of interleaving background replay from earlier levels in curriculum (default: 0.35)",
    )
    parser.add_argument(
        "--enable-wisp-sensor",
        action="store_true",
        help="Enable discrete wisp proximity sensor (default: False to suppress Level 5 overhead noise)",
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
        default=0.003,
        help="Minimum Q-learning learning rate alpha after decay (default: 0.003)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99)",
    )
    parser.add_argument(
        "--lambda",
        dest="trace_decay",
        type=float,
        default=0.0,
        help="Replacing eligibility trace decay lambda in [0.0, 1.0] (default: 0.0 for 1-step TD)",
    )
    parser.add_argument(
        "--lambda-min",
        dest="trace_decay_min",
        type=float,
        default=0.35,
        help="Minimum eligibility trace decay floor reached at 75%% of each stage (default: 0.35)",
    )
    parser.add_argument(
        "--lambda-polish-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable two-phase trace schedule with polish tail ramping lambda->0.0, eps->0.005, alpha->alpha_min (default: True)",
    )
    parser.add_argument(
        "--polish-epsilon-min",
        type=float,
        default=0.005,
        help="Target epsilon floor during the final 85%-100% polish tail when --lambda-polish-zero is enabled (default: 0.005)",
    )
    parser.add_argument(
        "--polish-anchor-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload stage-best Q-weights at Polish Tail onset and roll back mid-polish regressions (default: True)",
    )
    parser.add_argument(
        "--main-collapse-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable main-phase anti-collapse recovery guard on long stages >= 5000 eps (default: True)",
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
        default=None,
        help="Explicit per-episode epsilon decay multiplier (default: auto-computed per stage)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Interval of episodes between greedy evaluations for best model checkpointing (default: 25)",
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

    max_steps_cap = None if args.max_steps_cap == 0 else args.max_steps_cap

    # Episode schedule & smoke-test overrides
    if args.smoke_test:
        print("[train.py] Smoke-test mode active: overriding parameters for fast verification.")
        episodes_per_level = 5
        episodes_schedule = {lvl: 5 for lvl in args.levels}
        eval_interval = 2
        eval_episodes_periodic = 2
        final_eval_episodes = 3
        untrained_eval_episodes = 2
        checkpoint_interval = 5
    elif args.episodes is not None:
        episodes_per_level = args.episodes
        episodes_schedule = {lvl: args.episodes for lvl in args.levels}
        eval_interval = args.eval_interval
        eval_episodes_periodic = 10
        final_eval_episodes = 50
        untrained_eval_episodes = 30
        checkpoint_interval = args.checkpoint_interval
    else:
        episodes_schedule = {
            lvl: DEFAULT_EPISODE_SCHEDULE.get(lvl, 10000) for lvl in args.levels
        }
        if args.l5_episodes is not None and 5 in episodes_schedule:
            episodes_schedule[5] = int(args.l5_episodes)
        if args.l6_episodes is not None and 6 in episodes_schedule:
            episodes_schedule[6] = int(args.l6_episodes)
        episodes_per_level = max(episodes_schedule.values()) if episodes_schedule else 2000
        eval_interval = args.eval_interval
        eval_episodes_periodic = 10
        final_eval_episodes = 50
        untrained_eval_episodes = 30
        checkpoint_interval = args.checkpoint_interval

    if not args.smoke_test and args.episodes is not None:
        if args.l5_episodes is not None and 5 in episodes_schedule:
            episodes_schedule[5] = int(args.l5_episodes)
        if args.l6_episodes is not None and 6 in episodes_schedule:
            episodes_schedule[6] = int(args.l6_episodes)
        episodes_per_level = max(episodes_schedule.values()) if episodes_schedule else episodes_per_level

    auto_epsilon_decay = args.epsilon_decay is None
    init_epsilon_decay = 0.997 if args.epsilon_decay is None else args.epsilon_decay

    # Determine experiment grid
    agents = ["tabular", "hybrid"] if args.agent == "all" else [args.agent]
    modes = ["curriculum", "separate"] if args.mode == "all" else [args.mode]
    rewards = ["balanced", "sparse", "dense"] if args.reward == "all" else [args.reward]

    total_experiments = len(agents) * len(modes) * len(rewards)
    print(
        f"[train.py] Executing {total_experiments} experiment(s): "
        f"Agents={agents}, Modes={modes}, Rewards={rewards}, Levels={args.levels}, "
        f"Schedule={episodes_schedule}, Lambda={args.trace_decay}->{args.trace_decay_min:.2f}"
        f"{'->0(polish)' if args.lambda_polish_zero else ''}, "
        f"PolishAnchor={args.polish_anchor_best}, AlphaMin={args.alpha_min}, "
        f"WispSensor={args.enable_wisp_sensor}, MaxStepsCap={max_steps_cap}"
    )

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
                            epsilon_decay=init_epsilon_decay,
                            trace_decay=args.trace_decay,
                            seed=args.seed,
                        )
                else:
                    def make_agent() -> HybridQSARSAAgent:
                        return HybridQSARSAAgent(
                            learning_rate=args.alpha,
                            discount_factor=args.gamma,
                            epsilon=args.epsilon_start,
                            epsilon_min=args.epsilon_min,
                            epsilon_decay=init_epsilon_decay,
                            mode=args.hybrid_mode,
                            beta=args.beta,
                            trace_decay=args.trace_decay,
                            seed=args.seed,
                        )

                reward_kwargs = (
                    {"wisp_threat_distance": 65.0 if args.enable_wisp_sensor else -1.0}
                    if reward_name == "balanced"
                    else {}
                )
                reward_strategy = create_reward_strategy(reward_name, **reward_kwargs)
                discretizer = MarioStateDiscretizer(enable_wisp_sensor=args.enable_wisp_sensor)

                trainer = Trainer(
                    env_factory=create_env,
                    discretizer=discretizer,
                    reward_strategy=reward_strategy,
                    reward_name=reward_name,
                    agent_type=agent_type,
                    hybrid_mode=args.hybrid_mode,
                    beta=args.beta,
                    trace_decay=args.trace_decay,
                    trace_decay_min=args.trace_decay_min,
                    lambda_polish_zero=args.lambda_polish_zero,
                    polish_epsilon_min=args.polish_epsilon_min,
                    polish_anchor_best=args.polish_anchor_best,
                    main_collapse_guard=args.main_collapse_guard,
                    max_steps_cap=max_steps_cap,
                    action_repeat=args.action_repeat,
                    rewarm_epsilon=args.rewarm_epsilon,
                    rewarm_epsilon_ghost=args.rewarm_epsilon_ghost,
                    rewarm_epsilon_hard=args.rewarm_epsilon_hard,
                    rewarm_epsilon_l5=args.rewarm_epsilon_l5,
                    rewarm_epsilon_l6=args.rewarm_epsilon_l6,
                    alpha_min=args.alpha_min,
                    replay_ratio=args.replay_ratio,
                    eval_interval=eval_interval,
                    eval_episodes_periodic=eval_episodes_periodic,
                    final_eval_episodes=final_eval_episodes,
                    untrained_eval_episodes=untrained_eval_episodes,
                    checkpoint_interval=checkpoint_interval,
                    checkpoints_dir=args.checkpoints_dir,
                    outputs_dir=args.outputs_dir,
                    seed=args.seed,
                )

                total_eps = sum(episodes_schedule.values())
                tracker = ProgressTracker(
                    agent_tag=agent_tag,
                    mode=mode,
                    reward=reward_name,
                    total_episodes=total_eps,
                    use_inline=use_inline,
                    state_space_max=discretizer.state_space_size,
                )

                using_default_dirs = (
                    Path(args.checkpoints_dir) == Path("checkpoints")
                    and Path(args.outputs_dir) == Path("outputs")
                )
                is_canonical = (
                    using_default_dirs
                    and mode == "curriculum"
                    and reward_name == "balanced"
                    and agent_tag == "tabular"
                )

                result = trainer.run_experiment(
                    agent_factory=make_agent,
                    mode=mode,
                    levels=args.levels,
                    episodes_per_level=episodes_per_level,
                    episodes_schedule=episodes_schedule,
                    auto_epsilon_decay=auto_epsilon_decay,
                    is_canonical_production=is_canonical,
                    progress_callback=tracker.callback,
                )
                all_results[(agent_tag, mode, reward_name)] = result

    # Print summary comparison table
    print_comparison_table(all_results, args.levels)
    print("\n[train.py] All experiments finished successfully!")


if __name__ == "__main__":
    main()
