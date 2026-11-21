from collections import deque
import csv
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.base_env import BaseEnvironment, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy


class Trainer:
    """
    Modular training and evaluation harness for Hoá's Midnight Escape RL agent.
    Decoupled strictly from concrete implementations via core interfaces.
    """

    def __init__(
        self,
        env_factory: Callable[[int], BaseEnvironment],
        discretizer: BaseDiscretizer,
        reward_strategy: BaseRewardStrategy,
        reward_name: str = "balanced",
        action_repeat: int = 1,
        rewarm_epsilon: float = 0.30,
        rewarm_epsilon_ghost: float = 0.40,
        alpha_min: float = 0.02,
        replay_ratio: float = 0.35,
        eval_interval: int = 50,
        eval_episodes_periodic: int = 10,
        final_eval_episodes: int = 50,
        untrained_eval_episodes: int = 30,
        checkpoint_interval: int = 200,
        checkpoints_dir: str = "checkpoints",
        outputs_dir: str = "outputs",
    ) -> None:
        self.env_factory = env_factory
        self.discretizer = discretizer
        self.reward_strategy = reward_strategy
        self.reward_name = reward_name
        self.action_repeat = max(1, action_repeat)
        self.rewarm_epsilon = rewarm_epsilon
        self.rewarm_epsilon_ghost = rewarm_epsilon_ghost
        self.alpha_min = alpha_min
        self.replay_ratio = replay_ratio
        self.eval_interval = eval_interval
        self.eval_episodes_periodic = eval_episodes_periodic
        self.final_eval_episodes = final_eval_episodes
        self.untrained_eval_episodes = untrained_eval_episodes
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints_dir = checkpoints_dir
        self.outputs_dir = outputs_dir

    def _run_episode(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        training: bool = True,
        decay_eps: bool = True,
    ) -> Dict:
        """
        Executes one full episode headlessly.
        Never calls env.render() to preserve thousands of FPS throughput.
        """
        start_x = getattr(env, "start_x", 50.0)
        goal = getattr(env, "goal", None)
        goal_x = getattr(goal, "x", 1000.0) if goal is not None else 1000.0
        items = getattr(env, "items", [])
        items_total = len(items)

        raw_state: RawState = env.reset()
        discrete_state = self.discretizer.discretize(raw_state)

        total_reward = 0.0
        steps = 0
        max_x = raw_state.player_x
        obstacle_hits = 0
        items_collected = 0
        done = False
        last_events: Optional[GameEvents] = None

        while not done:
            action = agent.select_action(discrete_state)

            macro_reward = 0.0
            # Execute action with action repeat
            for _ in range(self.action_repeat):
                next_raw_state, events, done = env.step(action)
                steps += 1
                step_reward = self.reward_strategy.calculate_reward(
                    raw_state, action, next_raw_state, events, done
                )
                macro_reward += step_reward
                max_x = max(max_x, next_raw_state.player_x)
                if events.hit_obstacle:
                    obstacle_hits += 1
                if events.collected_item:
                    items_collected += 1
                last_events = events
                raw_state = next_raw_state
                if done:
                    break

            next_discrete_state = self.discretizer.discretize(raw_state)

            if training:
                agent.update(
                    discrete_state, action, macro_reward, next_discrete_state, done
                )

            discrete_state = next_discrete_state
            total_reward += macro_reward

        # Exploration rate decay at episode completion
        if training and decay_eps:
            agent.decay_epsilon()

        # Outcome categorization
        if last_events is not None and last_events.reached_goal:
            outcome = "win"
        elif last_events is not None and last_events.hit_ghost:
            outcome = "ghost_death"
        elif last_events is not None and last_events.fell_in_pit:
            outcome = "pit_death"
        else:
            outcome = "timeout"

        won = outcome == "win"

        player_width = getattr(getattr(env, "player", None), "width", 24.0)
        effective_goal_x = max(start_x + 1.0, goal_x - player_width)
        if won:
            progress_pct = 100.0
        else:
            progress_pct = float(
                np.clip(
                    ((max_x - start_x) / (effective_goal_x - start_x)) * 100.0,
                    0.0,
                    100.0,
                )
            )

        return {
            "total_reward": round(float(total_reward), 2),
            "steps": int(steps),
            "max_x_reached": round(float(max_x), 1),
            "progress_pct": round(float(progress_pct), 2),
            "obstacle_hits": int(obstacle_hits),
            "items_collected": int(items_collected),
            "items_total": int(items_total),
            "won": bool(won),
            "outcome": outcome,
        }

    def evaluate_agent(
        self,
        agent: BaseAgent,
        level_id: int,
        episodes: int,
        random_baseline: bool = False,
    ) -> Dict:
        """
        Runs evaluation episodes.
        If random_baseline=True, sets exploration to pure random (epsilon=1.0).
        Otherwise sets agent to pure greedy exploitation (epsilon=0.0 / training=False).
        Restores agent state afterward.
        """
        if episodes <= 0:
            return {
                "win_rate": 0.0,
                "avg_reward": 0.0,
                "avg_completion_steps": -1.0,
                "avg_items_collected": 0.0,
            }

        prev_training = getattr(agent, "training", True)
        prev_epsilon = getattr(agent, "epsilon", 1.0)

        if random_baseline:
            agent.set_mode(True)
            setattr(agent, "epsilon", 1.0)
        else:
            agent.set_mode(False)

        wins = 0
        total_rewards: List[float] = []
        completion_steps: List[int] = []
        items_collected_list: List[int] = []

        env = self.env_factory(level_id)
        try:
            for _ in range(episodes):
                res = self._run_episode(env, agent, training=False)
                total_rewards.append(res["total_reward"])
                items_collected_list.append(res["items_collected"])
                if res["won"]:
                    wins += 1
                    completion_steps.append(res["steps"])
        finally:
            env.close()
            agent.set_mode(prev_training)
            setattr(agent, "epsilon", prev_epsilon)

        win_rate = round(float((wins / episodes) * 100.0), 2)
        avg_reward = round(float(np.mean(total_rewards)), 2) if total_rewards else 0.0
        avg_completion_steps = (
            round(float(np.mean(completion_steps)), 1) if completion_steps else -1.0
        )
        avg_items_collected = (
            round(float(np.mean(items_collected_list)), 2) if items_collected_list else 0.0
        )

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_completion_steps": avg_completion_steps,
            "avg_items_collected": avg_items_collected,
        }

    def run_experiment(
        self,
        agent_factory: Callable[[], BaseAgent],
        mode: str = "curriculum",
        levels: Optional[List[int]] = None,
        episodes_per_level: int = 1000,
        is_canonical_production: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """
        Executes a complete RL training experiment in either curriculum or separate mode.
        Exports checkpoints and Member 4 data boards (JSON & CSV).
        """
        if levels is None:
            levels = [1, 2, 3, 4, 5]

        # Canonical production run if curriculum + balanced
        if mode == "curriculum" and self.reward_name == "balanced":
            is_canonical_production = True

        run_cp_dir = Path(self.checkpoints_dir) / f"{mode}_{self.reward_name}"
        run_cp_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir_path = Path(self.outputs_dir)
        outputs_dir_path.mkdir(parents=True, exist_ok=True)

        episode_history: List[Dict] = []
        evaluation_summary: Dict[str, Dict] = {}
        states_discovered_per_level: Dict[str, int] = {}
        global_episode = 0

        # 1. Compute untrained baselines for each level
        print(f"[Trainer] Computing untrained baselines ({self.untrained_eval_episodes} eps each)...")
        untrained_baselines: Dict[int, Dict] = {}
        for lvl in levels:
            temp_agent = agent_factory()
            untrained_baselines[lvl] = self.evaluate_agent(
                temp_agent, lvl, self.untrained_eval_episodes, random_baseline=True
            )

        # ----------------------------------------------------
        # CURRICULUM MODE: One shared agent trained across levels
        # ----------------------------------------------------
        alpha_start = 0.12
        if mode == "curriculum":
            agent = agent_factory()
            alpha_start = float(getattr(agent, "alpha", 0.12))
            # 3-tuple ranking: (len(unlocked_levels), avg_win_rate, avg_eval_reward)
            best_global_score = (0, -1.0, float("-inf"))

            for lvl_idx, lvl in enumerate(levels):
                print(f"[Trainer] [Curriculum] Starting Level {lvl} ({episodes_per_level} episodes)...")
                recent_wins: deque = deque(maxlen=50)  # Reset per level
                best_stage_score = (-1.0, float("-inf"))
                unlocked_levels = levels[: lvl_idx + 1]

                # If transitioning to a new level
                if lvl_idx > 0:
                    prev_lvl = levels[lvl_idx - 1]
                    # Zero-shot evaluation checks
                    if lvl == 3 and prev_lvl == 2:
                        print(f"[Trainer] Running Level 3 zero-shot evaluation ({self.final_eval_episodes} eps)...")
                        z3_res = self.evaluate_agent(agent, 3, self.final_eval_episodes)
                        evaluation_summary["level_3_zero_shot"] = z3_res
                    elif lvl == 5 and prev_lvl == 4:
                        print(f"[Trainer] Running Level 5 zero-shot evaluation ({self.final_eval_episodes} eps)...")
                        z5_res = self.evaluate_agent(agent, 5, self.final_eval_episodes)
                        evaluation_summary["level_5_zero_shot"] = z5_res

                    # Stage-Aware exploration re-warming (0.40 for ghost levels >= 4, else 0.30)
                    rewarm_target = (
                        self.rewarm_epsilon_ghost if lvl >= 4 else self.rewarm_epsilon
                    )
                    current_eps = getattr(agent, "epsilon", 0.0)
                    new_eps = max(current_eps, rewarm_target)
                    setattr(agent, "epsilon", new_eps)
                    print(
                        f"[Trainer] Re-warmed exploration epsilon: {current_eps:.3f} -> {new_eps:.3f} (target: {rewarm_target:.3f})"
                    )

                env = self.env_factory(lvl)
                try:
                    for lvl_ep in range(1, episodes_per_level + 1):
                        global_episode += 1

                        # Linear learning-rate decay (alpha_start -> alpha_min) per stage
                        if hasattr(agent, "alpha"):
                            agent.alpha = self.alpha_min + (alpha_start - self.alpha_min) * (
                                1.0 - (lvl_ep / max(1, episodes_per_level))
                            )

                        ep_res = self._run_episode(env, agent, training=True, decay_eps=True)

                        # Option B Interleaved Background Replay (Curriculum Mode Only)
                        if lvl_idx > 1 and self.replay_ratio > 0.0:
                            if random.random() < self.replay_ratio:
                                replay_candidates = levels[: lvl_idx - 1]
                                if replay_candidates:
                                    replay_lvl = random.choice(replay_candidates)
                                    replay_env = self.env_factory(replay_lvl)
                                    try:
                                        self._run_episode(
                                            replay_env, agent, training=True, decay_eps=False
                                        )
                                    finally:
                                        replay_env.close()

                        recent_wins.append(1 if ep_res["won"] else 0)
                        rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                        rec = {
                            "global_episode": global_episode,
                            "level_episode": lvl_ep,
                            "level_id": lvl,
                            "total_reward": round(float(ep_res["total_reward"]), 2),
                            "steps": ep_res["steps"],
                            "max_x_reached": round(float(ep_res["max_x_reached"]), 1),
                            "progress_pct": round(float(ep_res["progress_pct"]), 2),
                            "obstacle_hits": ep_res["obstacle_hits"],
                            "items_collected": ep_res["items_collected"],
                            "items_total": ep_res["items_total"],
                            "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                            "won": ep_res["won"],
                            "outcome": ep_res["outcome"],
                            "rolling_win_rate_50": rolling_win_rate,
                        }
                        episode_history.append(rec)

                        if progress_callback is not None:
                            progress_callback({
                                "mode": mode,
                                "reward": self.reward_name,
                                "level_id": lvl,
                                "level_index": lvl_idx + 1,
                                "total_levels": len(levels),
                                "level_episode": lvl_ep,
                                "episodes_per_level": episodes_per_level,
                                "global_episode": global_episode,
                                "total_episodes": len(levels) * episodes_per_level,
                                "total_reward": round(float(ep_res["total_reward"]), 2),
                                "rolling_win_rate_50": rolling_win_rate,
                                "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                                "states_discovered": len(getattr(agent, "q_table", {})),
                            })

                        # Periodic greedy evaluation for checkpointing
                        if (lvl_ep % self.eval_interval == 0) or (lvl_ep == episodes_per_level):
                            # Evaluate across all unlocked levels so far
                            stage_evals = [
                                self.evaluate_agent(agent, u_lvl, self.eval_episodes_periodic)
                                for u_lvl in unlocked_levels
                            ]
                            stage_win_rate = round(float(np.mean([r["win_rate"] for r in stage_evals])), 2)
                            stage_avg_reward = round(float(np.mean([r["avg_reward"] for r in stage_evals])), 2)

                            # 1. Update per-stage best
                            stage_score = (stage_win_rate, stage_avg_reward)
                            if stage_score > best_stage_score:
                                best_stage_score = stage_score
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.pkl"))
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.json"))

                            # 2. Update global 3-tuple best: (len(unlocked_levels), avg_win_rate, avg_eval_reward)
                            candidate_global_score = (len(unlocked_levels), stage_win_rate, stage_avg_reward)
                            if candidate_global_score > best_global_score:
                                best_global_score = candidate_global_score
                                agent.save(str(run_cp_dir / "q_table_best.pkl"))
                                agent.save(str(run_cp_dir / "q_table_best.json"))
                                if is_canonical_production:
                                    root_cp = Path(self.checkpoints_dir)
                                    root_cp.mkdir(parents=True, exist_ok=True)
                                    agent.save(str(root_cp / "q_table_best.pkl"))
                                    agent.save(str(root_cp / "q_table_best.json"))

                        # Periodic checkpointing
                        if global_episode % self.checkpoint_interval == 0:
                            agent.save(str(run_cp_dir / f"q_table_ep{global_episode}.pkl"))

                finally:
                    env.close()

                # Record states discovered up to this level
                states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                # Level completion trained evaluation
                eval_trained = self.evaluate_agent(agent, lvl, self.final_eval_episodes)
                evaluation_summary[f"level_{lvl}"] = {
                    "untrained_win_rate": round(float(untrained_baselines[lvl]["win_rate"]), 2),
                    "trained_win_rate": round(float(eval_trained["win_rate"]), 2),
                    "untrained_avg_reward": round(float(untrained_baselines[lvl]["avg_reward"]), 2),
                    "trained_avg_reward": round(float(eval_trained["avg_reward"]), 2),
                    "avg_completion_steps": (
                        round(float(eval_trained["avg_completion_steps"]), 1)
                        if eval_trained["avg_completion_steps"] > 0
                        else -1.0
                    ),
                    "avg_items_collected": round(float(eval_trained["avg_items_collected"]), 2),
                }

            # Final checkpoint saves
            agent.save(str(run_cp_dir / "q_table_final.pkl"))
            agent.save(str(run_cp_dir / "q_table_final.json"))

            # Ensure best checkpoints exist even if somehow not saved yet
            if not (run_cp_dir / "q_table_best.pkl").exists():
                agent.save(str(run_cp_dir / "q_table_best.pkl"))
                agent.save(str(run_cp_dir / "q_table_best.json"))

            if is_canonical_production:
                root_cp = Path(self.checkpoints_dir)
                root_cp.mkdir(parents=True, exist_ok=True)
                agent.save(str(root_cp / "q_table_final.pkl"))
                agent.save(str(root_cp / "q_table_final.json"))
                if not (root_cp / "q_table_best.pkl").exists():
                    agent.save(str(root_cp / "q_table_best.pkl"))
                    agent.save(str(root_cp / "q_table_best.json"))

        # ----------------------------------------------------
        # SEPARATE MODE: Independent fresh agent per level
        # ----------------------------------------------------
        else:
            agent = None
            for lvl_idx, lvl in enumerate(levels):
                print(f"[Trainer] [Separate] Training Level {lvl} with fresh agent...")
                agent = agent_factory()
                alpha_start = float(getattr(agent, "alpha", 0.12))
                recent_wins: deque = deque(maxlen=50)  # Reset per level
                best_stage_score = (-1.0, float("-inf"))

                env = self.env_factory(lvl)
                try:
                    for lvl_ep in range(1, episodes_per_level + 1):
                        global_episode += 1

                        # Linear learning-rate decay (alpha_start -> alpha_min) per stage
                        if hasattr(agent, "alpha"):
                            agent.alpha = self.alpha_min + (alpha_start - self.alpha_min) * (
                                1.0 - (lvl_ep / max(1, episodes_per_level))
                            )

                        ep_res = self._run_episode(env, agent, training=True, decay_eps=True)

                        recent_wins.append(1 if ep_res["won"] else 0)
                        rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                        rec = {
                            "global_episode": global_episode,
                            "level_episode": lvl_ep,
                            "level_id": lvl,
                            "total_reward": round(float(ep_res["total_reward"]), 2),
                            "steps": ep_res["steps"],
                            "max_x_reached": round(float(ep_res["max_x_reached"]), 1),
                            "progress_pct": round(float(ep_res["progress_pct"]), 2),
                            "obstacle_hits": ep_res["obstacle_hits"],
                            "items_collected": ep_res["items_collected"],
                            "items_total": ep_res["items_total"],
                            "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                            "won": ep_res["won"],
                            "outcome": ep_res["outcome"],
                            "rolling_win_rate_50": rolling_win_rate,
                        }
                        episode_history.append(rec)

                        if progress_callback is not None:
                            progress_callback({
                                "mode": mode,
                                "reward": self.reward_name,
                                "level_id": lvl,
                                "level_index": lvl_idx + 1,
                                "total_levels": len(levels),
                                "level_episode": lvl_ep,
                                "episodes_per_level": episodes_per_level,
                                "global_episode": global_episode,
                                "total_episodes": len(levels) * episodes_per_level,
                                "total_reward": round(float(ep_res["total_reward"]), 2),
                                "rolling_win_rate_50": rolling_win_rate,
                                "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                                "states_discovered": len(getattr(agent, "q_table", {})),
                            })

                        # Periodic greedy-eval best checkpointing per level
                        if (lvl_ep % self.eval_interval == 0) or (lvl_ep == episodes_per_level):
                            eval_res = self.evaluate_agent(
                                agent, lvl, self.eval_episodes_periodic
                            )
                            score = (eval_res["win_rate"], eval_res["avg_reward"])
                            if score > best_stage_score:
                                best_stage_score = score
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.pkl"))
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.json"))

                        # Periodic checkpointing
                        if lvl_ep % self.checkpoint_interval == 0:
                            agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_ep{lvl_ep}.pkl"))

                finally:
                    env.close()

                # Record states discovered for this level
                states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                # Level completion trained evaluation
                eval_trained = self.evaluate_agent(agent, lvl, self.final_eval_episodes)
                evaluation_summary[f"level_{lvl}"] = {
                    "untrained_win_rate": round(float(untrained_baselines[lvl]["win_rate"]), 2),
                    "trained_win_rate": round(float(eval_trained["win_rate"]), 2),
                    "untrained_avg_reward": round(float(untrained_baselines[lvl]["avg_reward"]), 2),
                    "trained_avg_reward": round(float(eval_trained["avg_reward"]), 2),
                    "avg_completion_steps": (
                        round(float(eval_trained["avg_completion_steps"]), 1)
                        if eval_trained["avg_completion_steps"] > 0
                        else -1.0
                    ),
                    "avg_items_collected": round(float(eval_trained["avg_items_collected"]), 2),
                }

                # Save final per-level checkpoints
                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_final.pkl"))
                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_final.json"))
                if not (run_cp_dir / f"q_table_lvl{lvl}_best.pkl").exists():
                    agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.pkl"))
                    agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.json"))

        # ----------------------------------------------------
        # EXPORT MEMBER 4 DATA BOARDS (JSON & CSV)
        # ----------------------------------------------------
        if mode == "curriculum":
            total_states = len(getattr(agent, "q_table", {})) if agent is not None else 0
        else:
            total_states = max(states_discovered_per_level.values()) if states_discovered_per_level else 0

        board_data = {
            "metadata": {
                "training_mode": mode,
                "reward_profile": self.reward_name,
                "action_repeat": self.action_repeat,
                "episodes_per_level": episodes_per_level,
                "levels": levels,
                "rewarm_epsilon": round(float(self.rewarm_epsilon), 4),
                "rewarm_epsilon_ghost": round(float(self.rewarm_epsilon_ghost), 4),
                "replay_ratio": round(float(self.replay_ratio if mode == "curriculum" else 0.0), 4),
                "state_space_max": 1152,
                "num_states_discovered": total_states,
                "states_discovered_per_level": states_discovered_per_level,
                "alpha": round(float(alpha_start), 4),
                "alpha_start": round(float(alpha_start), 4),
                "alpha_min": round(float(self.alpha_min), 4),
                "gamma": round(float(getattr(agent, "gamma", 0.99)), 4) if agent else 0.99,
                "epsilon_min": round(float(getattr(agent, "epsilon_min", 0.01)), 6) if agent else 0.01,
                "epsilon_decay": round(float(getattr(agent, "epsilon_decay", 0.995)), 6) if agent else 0.995,
            },
            "evaluation_summary": evaluation_summary,
            "episode_history": episode_history,
        }

        json_out_path = outputs_dir_path / f"training_board_{mode}_{self.reward_name}.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(board_data, f, indent=2)

        csv_out_path = outputs_dir_path / f"training_metrics_{mode}_{self.reward_name}.csv"
        csv_headers = [
            "global_episode",
            "level_episode",
            "level_id",
            "total_reward",
            "steps",
            "max_x_reached",
            "progress_pct",
            "obstacle_hits",
            "items_collected",
            "items_total",
            "epsilon",
            "won",
            "outcome",
            "rolling_win_rate_50",
        ]
        with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for row in episode_history:
                writer.writerow(row)

        print(f"[Trainer] Exported Member 4 JSON board to: {json_out_path}")
        print(f"[Trainer] Exported Member 4 CSV metrics to: {csv_out_path}")

        return board_data
