import argparse
import copy
from dataclasses import dataclass
import multiprocessing as mp
import os
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


class ProgressTracker:
    """Zero-dependency ASCII progress tracker with inline or newline output."""

    def __init__(
        self,
        agent_tag: str,
        mode: str,
        reward: str,
        total_episodes: int,
        use_inline: bool,
        seed: Optional[int] = None,
    ) -> None:
        self.agent_tag = agent_tag
        self.mode = mode
        self.reward = reward
        self.total_episodes = total_episodes
        self.use_inline = use_inline
        self.seed = seed
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
        seed_str = f" | Seed {self.seed}" if self.seed is not None else ""

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
                    f"\r[{bar}] {pct_str} | {self.agent_tag} | {self.mode.capitalize()} | {self.reward.upper()}{seed_str} | "
                    f"Lvl {data['level_id']} ({lvl_ep}/{ep_per_lvl}) | "
                    f"Win100: {data.get('rolling_win_rate_100', data.get('rolling_win_rate_50', 0.0)):5.1f}% | "
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
                    f"[{self.agent_tag.upper()} | {self.mode.capitalize()} | {self.reward.upper()}{seed_str}] "
                    f"Global Ep {current_ep:4d}/{total_ep:4d} | "
                    f"Lvl {data['level_id']} ({lvl_ep:4d}/{ep_per_lvl:4d}) | "
                    f"Win100: {data.get('rolling_win_rate_100', data.get('rolling_win_rate_50', 0.0)):5.1f}% | "
                    f"R: {data['total_reward']:+7.1f} | eps: {data['epsilon']:.3f} | "
                    f"States: {data['states_discovered']:4d}/1152 | ETA: {eta_str}"
                )
                sys.stdout.flush()


def print_comparison_table(
    all_results: Dict[Tuple[str, str, str, Optional[int]], Dict],
    levels: List[int],
    is_multi_seed: bool = False,
) -> None:
    """Prints a clear comparison table across all completed (agent, mode, reward, seed) experiments."""
    print("\n" + "=" * 140)
    print(" " * 50 + "EXPERIMENT EVALUATION COMPARISON SUMMARY")
    print("=" * 140)
    
    if not is_multi_seed:
        header = (
            f"{'Agent':<16} {'Mode':<12} {'Reward':<10} {'Level':<14} {'Untrained Win%':<16} "
            f"{'Trained Win%':<14} {'Untrained R':<14} {'Trained R':<14} {'Clear Steps':<14} {'Items'}"
        )
        print(header)
        print("-" * 140)

        for (agent_tag, mode, reward_name, seed), res in all_results.items():
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
            print("-" * 140)
    else:
        # Group results by (agent_tag, mode, reward_name) across seeds
        grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
        for (agent_tag, mode, reward_name, seed), res in all_results.items():
            key = (agent_tag, mode, reward_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(res)

        header = (
            f"{'Agent':<16} {'Mode':<12} {'Reward':<10} {'Seeds':<7} {'Level':<12} "
            f"{'Trained Win% (Mean±Std)':<26} {'Trained Reward (Mean±Std)':<28} {'Clear Steps (Mean)'}"
        )
        print(header)
        print("-" * 140)

        for (agent_tag, mode, reward_name), res_list in grouped.items():
            n_seeds = len(res_list)
            for lvl in levels:
                lvl_key = f"level_{lvl}"
                win_rates = []
                rewards = []
                steps = []

                for r in res_list:
                    stats = r.get("evaluation_summary", {}).get(lvl_key, {})
                    win_rates.append(stats.get("trained_win_rate", 0.0))
                    rewards.append(stats.get("trained_avg_reward", 0.0))
                    s_val = stats.get("avg_completion_steps", -1.0)
                    if s_val > 0:
                        steps.append(s_val)

                win_mean, win_std = np.mean(win_rates), np.std(win_rates)
                r_mean, r_std = np.mean(rewards), np.std(rewards)
                steps_str = f"{np.mean(steps):.1f}" if steps else "N/A"

                win_str = f"{win_mean:.1f}% ± {win_std:.1f}%"
                rew_str = f"{r_mean:+.1f} ± {r_std:.1f}"

                print(
                    f"{agent_tag:<16} {mode:<12} {reward_name:<10} {n_seeds:<7} {f'Level {lvl}':<12} "
                    f"{win_str:<26} {rew_str:<28} {steps_str}"
                )
            print("-" * 140)


@dataclass
class JobConfig:
    agent_type: str
    agent_tag: str
    mode: str
    reward_name: str
    seed: int
    levels: List[int]
    episodes_per_level: int
    eval_interval: int
    eval_episodes_periodic: int
    final_eval_episodes: int
    untrained_eval_episodes: int
    checkpoint_interval: int
    alpha: float
    alpha_min: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    hybrid_mode: str
    beta: float
    action_repeat: int
    rewarm_epsilon: float
    rewarm_epsilon_ghost: float
    replay_ratio: float
    checkpoints_dir: str
    outputs_dir: str
    map_seed: Optional[int]
    map_seed_interval: int
    use_inline: bool
    is_multi_job: bool


def _run_single_job(job: JobConfig) -> Tuple[Tuple[str, str, str, Optional[int]], Dict]:
    """Worker function executed inside multiprocessing pool or single process."""
    # Deterministic per-experiment seed
    random.seed(job.seed)
    np.random.seed(job.seed)

    # Factory for agents with exact seed and hyperparameters
    if job.agent_type == "tabular":
        def make_agent() -> TabularQAgent:
            return TabularQAgent(
                learning_rate=job.alpha,
                discount_factor=job.gamma,
                epsilon=job.epsilon_start,
                epsilon_min=job.epsilon_min,
                epsilon_decay=job.epsilon_decay,
                seed=job.seed,
            )
    else:
        def make_agent() -> HybridQSARSAAgent:
            return HybridQSARSAAgent(
                learning_rate=job.alpha,
                discount_factor=job.gamma,
                epsilon=job.epsilon_start,
                epsilon_min=job.epsilon_min,
                epsilon_decay=job.epsilon_decay,
                mode=job.hybrid_mode,
                beta=job.beta,
                seed=job.seed,
            )

    reward_strategy = create_reward_strategy(job.reward_name)
    discretizer = MarioStateDiscretizer()

    trainer = Trainer(
        env_factory=create_env,
        discretizer=discretizer,
        reward_strategy=reward_strategy,
        reward_name=job.reward_name,
        agent_type=job.agent_type,
        hybrid_mode=job.hybrid_mode,
        beta=job.beta,
        action_repeat=job.action_repeat,
        rewarm_epsilon=job.rewarm_epsilon,
        rewarm_epsilon_ghost=job.rewarm_epsilon_ghost,
        alpha_min=job.alpha_min,
        replay_ratio=job.replay_ratio,
        eval_interval=job.eval_interval,
        eval_episodes_periodic=job.eval_episodes_periodic,
        final_eval_episodes=job.final_eval_episodes,
        untrained_eval_episodes=job.untrained_eval_episodes,
        checkpoint_interval=job.checkpoint_interval,
        checkpoints_dir=job.checkpoints_dir,
        outputs_dir=job.outputs_dir,
        map_seed=job.map_seed,
        map_seed_interval=job.map_seed_interval,
        seed=job.seed,
    )

    total_eps = len(job.levels) * job.episodes_per_level
    # In multi-job parallel execution, suppress in-place carriage returns to avoid garbled output
    tracker = ProgressTracker(
        agent_tag=job.agent_tag,
        mode=job.mode,
        reward=job.reward_name,
        total_episodes=total_eps,
        use_inline=job.use_inline and not job.is_multi_job,
        seed=job.seed,
    )

    is_canonical = (
        job.mode == "curriculum"
        and job.reward_name == "balanced"
        and job.agent_tag == "tabular"
        and (not job.is_multi_job or job.seed == 42)
    )

    seed_arg = job.seed if job.is_multi_job else None

    result = trainer.run_experiment(
        agent_factory=make_agent,
        mode=job.mode,
        levels=job.levels,
        episodes_per_level=job.episodes_per_level,
        is_canonical_production=is_canonical,
        progress_callback=tracker.callback,
        seed=seed_arg,
    )

    key = (job.agent_tag, job.mode, job.reward_name, job.seed if job.is_multi_job else None)
    return key, result


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
        help="Default random seed if --seeds is not given (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of random seeds to run (e.g. --seeds 1 2 3 4 ...)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds to generate starting from --seed (e.g. --num-seeds 16)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel worker processes to use across CPU cores (default: 1)",
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

    # Determine seeds list
    if args.seeds is not None:
        seeds = args.seeds
    elif args.num_seeds is not None:
        seeds = [args.seed + i for i in range(args.num_seeds)]
    else:
        seeds = [args.seed]

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

    use_inline = sys.stdout.isatty() and not args.no_progress_bar

    # Build job list
    jobs: List[JobConfig] = []
    for agent_type in agents:
        agent_tag = "tabular" if agent_type == "tabular" else f"hybrid_{args.hybrid_mode}"
        for mode in modes:
            for reward_name in rewards:
                for seed in seeds:
                    jobs.append(
                        JobConfig(
                            agent_type=agent_type,
                            agent_tag=agent_tag,
                            mode=mode,
                            reward_name=reward_name,
                            seed=seed,
                            levels=args.levels,
                            episodes_per_level=episodes_per_level,
                            eval_interval=eval_interval,
                            eval_episodes_periodic=eval_episodes_periodic,
                            final_eval_episodes=final_eval_episodes,
                            untrained_eval_episodes=untrained_eval_episodes,
                            checkpoint_interval=checkpoint_interval,
                            alpha=args.alpha,
                            alpha_min=args.alpha_min,
                            gamma=args.gamma,
                            epsilon_start=args.epsilon_start,
                            epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay,
                            hybrid_mode=args.hybrid_mode,
                            beta=args.beta,
                            action_repeat=args.action_repeat,
                            rewarm_epsilon=args.rewarm_epsilon,
                            rewarm_epsilon_ghost=args.rewarm_epsilon_ghost,
                            replay_ratio=args.replay_ratio,
                            checkpoints_dir=args.checkpoints_dir,
                            outputs_dir=args.outputs_dir,
                            map_seed=args.map_seed,
                            map_seed_interval=args.map_seed_interval,
                            use_inline=use_inline,
                            is_multi_job=(len(seeds) > 1 or len(agents) > 1 or len(modes) > 1 or len(rewards) > 1),
                        )
                    )

    total_jobs = len(jobs)
    workers = max(1, min(args.workers, total_jobs, os.cpu_count() or 1))

    if args.map_seed is not None:
        map_seed_desc = args.map_seed
    elif args.map_seed_interval > 0:
        seed_label = args.seed if len(seeds) == 1 else "per-job seed"
        map_seed_desc = f"random(seed={seed_label})"
    else:
        map_seed_desc = None

    print(
        f"[train.py] Executing {total_jobs} total job(s) across {workers} parallel worker(s): "
        f"Agents={agents}, Modes={modes}, Rewards={rewards}, Seeds={seeds}, Levels={args.levels}, "
        f"MapSeed={map_seed_desc}, MapRotateInterval={args.map_seed_interval}"
    )
    if (args.map_seed is not None or args.map_seed_interval > 0) and any(lvl > 7 for lvl in args.levels):
        print("[train.py] Note: Procedural map seeds apply to Levels 1-7; Level 8 remains a fixed handcrafted layout.")

    all_results: Dict[Tuple[str, str, str, Optional[int]], Dict] = {}
    start_time = time.time()

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, (key, res) in enumerate(pool.imap_unordered(_run_single_job, jobs), 1):
                all_results[key] = res
                agent_tag, mode, reward_name, seed_val = key
                seed_info = f" (Seed {seed_val})" if seed_val is not None else ""
                print(
                    f"[{idx}/{total_jobs}] Completed: {agent_tag.upper()} | "
                    f"{mode.upper()} | {reward_name.upper()}{seed_info} in {time.time() - start_time:.1f}s"
                )
    else:
        for idx, job in enumerate(jobs, 1):
            seed_info = f" (Seed {job.seed})" if len(seeds) > 1 else ""
            print(f"\n{'='*75}")
            print(
                f"[{idx}/{total_jobs}] Starting: Agent={job.agent_tag.upper()} | "
                f"Mode={job.mode.upper()} | Reward={job.reward_name.upper()}{seed_info}"
            )
            print(f"{'='*75}")
            key, res = _run_single_job(job)
            all_results[key] = res

    is_multi_seed = len(seeds) > 1
    print_comparison_table(all_results, args.levels, is_multi_seed=is_multi_seed)
    total_time = time.time() - start_time
    print(f"\n[train.py] All {total_jobs} job(s) finished successfully in {total_time:.2f}s!")


if __name__ == "__main__":
    main()
