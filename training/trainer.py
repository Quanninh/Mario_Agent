from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.base_env import BaseEnvironment, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy


def _derive_random_map_seed(
    base_seed: int,
    level_id: int,
    block_idx: int,
    attempt: int = 0,
) -> int:
    """Derives a deterministic non-sequential 31-bit seed for a rotation interval block."""
    payload = f"{base_seed}:{level_id}:{block_idx}:{attempt}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "big") % 2_147_483_646)


def resolve_map_seed(
    level_id: int,
    level_episode: int,
    map_seed: Optional[int] = None,
    map_seed_interval: int = 0,
    fallback_seed: int = 42,
) -> Optional[int]:
    """
    Computes the procedural map seed for Levels 1-7 given episode and rotation interval.
    When map_seed is None and rotation is enabled, mixes the canonical map (None)
    every 3rd block starting at block 2 (block_idx % 3 == 2) so the agent retains
    standard-level mastery while learning procedural generalization.
    Returns None for Level 8 or when disabled.
    """
    if level_id < 1 or level_id > 7:
        return None
    interval = int(map_seed_interval)
    if interval > 0:
        base_seed = int(fallback_seed) if map_seed is None else int(map_seed)
        block_idx = (max(1, int(level_episode)) - 1) // interval
        if block_idx == 0 and map_seed is not None:
            return int(map_seed)
        if map_seed is None and block_idx >= 2 and block_idx % 3 == 2:
            return None
        prev_seed: Optional[int] = None
        if block_idx == 1 and map_seed is not None:
            prev_seed = int(map_seed)
        elif block_idx >= 1 and not (map_seed is None and (block_idx - 1) >= 2 and (block_idx - 1) % 3 == 2):
            prev_seed = _derive_random_map_seed(base_seed, level_id, block_idx - 1)
        candidate = _derive_random_map_seed(base_seed, level_id, block_idx)
        attempt = 1
        while prev_seed is not None and candidate == prev_seed:
            candidate = _derive_random_map_seed(base_seed, level_id, block_idx, attempt=attempt)
            attempt += 1
        return candidate
    if map_seed is not None:
        return int(map_seed)
    return None


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
        agent_type: str = "tabular",
        hybrid_mode: str = "expected",
        beta: float = 0.25,
        action_repeat: int = 1,
        rewarm_epsilon: float = 0.30,
        rewarm_epsilon_ghost: float = 0.35,
        alpha_min: float = 0.02,
        replay_ratio: float = 0.35,
        eval_interval: int = 50,
        eval_episodes_periodic: int = 10,
        final_eval_episodes: int = 50,
        untrained_eval_episodes: int = 30,
        checkpoint_interval: int = 200,
        checkpoints_dir: str = "checkpoints",
        outputs_dir: str = "outputs",
        blind_eval_seeds: Optional[List[int]] = None,
        map_seed: Optional[int] = None,
        map_seed_interval: int = 0,
        seed: int = 42,
    ) -> None:
        self.env_factory = env_factory
        self.discretizer = discretizer
        self.reward_strategy = reward_strategy
        self.reward_name = reward_name
        self.agent_type = agent_type
        self.hybrid_mode = hybrid_mode
        self.beta = beta
        self.agent_tag = "tabular" if agent_type == "tabular" else f"hybrid_{hybrid_mode}"
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
        self.blind_eval_seeds = list(blind_eval_seeds) if blind_eval_seeds else []
        self.map_seed = map_seed
        self.map_seed_interval = max(0, int(map_seed_interval))
        self.seed = int(seed)

    def _create_env(self, level_id: int, map_seed: Optional[int] = None) -> BaseEnvironment:
        """Creates an environment instance, passing seed when supported by env_factory."""
        if map_seed is not None:
            try:
                return self.env_factory(level_id, seed=map_seed)
            except TypeError:
                return self.env_factory(level_id)
        return self.env_factory(level_id)

    @staticmethod
    def _compute_reverse_spawns(
        env: BaseEnvironment,
        lvl: int,
        episodes_per_level: int,
        curriculum: bool,
    ) -> List[Optional[float]]:
        """Computes safe reverse-curriculum spawn coordinates for the active environment."""
        if curriculum:
            if lvl < 4 or episodes_per_level < 50:
                return [None]
            fractions = (0.80, 0.60, 0.40, 0.20)
            tail: List[Optional[float]] = [None]
        else:
            if lvl < 4 or episodes_per_level < 20:
                return []
            fractions = (0.75, 0.50, 0.25)
            tail = []

        goal_obj = getattr(env, "goal", None)
        lvl_goal_x = getattr(goal_obj, "x", 1000.0) if goal_obj is not None else 1000.0
        spawns: List[Optional[float]] = [
            Trainer._find_safe_spawn_x(env, lvl_goal_x * frac) for frac in fractions
        ]
        spawns.extend(tail)
        return spawns

    @staticmethod
    def _find_safe_spawn_x(env: BaseEnvironment, target_x: float) -> float:
        """Find a safe ground x-coordinate near target_x for reverse curriculum."""
        default_start = getattr(env, "start_x", 50.0)
        goal = getattr(env, "goal", None)
        goal_x = getattr(goal, "x", 1000.0) if goal is not None else 1000.0
        obstacles = getattr(env, "obstacles", [])
        pits = getattr(env, "pits", [])
        ghosts = getattr(env, "ghosts", [])
        monsters = getattr(env, "monsters", [])

        def is_safe(x: float) -> bool:
            if x < default_start or x > goal_x - 80.0:
                return False
            for p in pits:
                if (p.x_start - 40.0) <= x <= (p.x_end + 15.0):
                    return False
            for o in obstacles:
                if (o.x - 45.0) <= x <= (o.x + o.width + 15.0):
                    return False
            for g in ghosts:
                if abs(x - g.x) < 35.0 and g.y >= 220.0:
                    return False
            for m in monsters:
                if abs(x - m.x) < 45.0:
                    return False
            return True

        # Search outward from target_x in 15px increments, preferring slightly earlier x
        for offset in range(0, int(goal_x), 15):
            if is_safe(target_x - offset):
                return float(target_x - offset)
            if is_safe(target_x + offset):
                return float(target_x + offset)
        return float(default_start)

    def _replay_trajectory(
        self,
        agent: BaseAgent,
        trajectory: List[Tuple[Any, int, float, Any, bool, Optional[int]]],
    ) -> None:
        """Backward TD sweep over a self-discovered winning trajectory."""
        for s, a, r, s_next, done, next_a in reversed(trajectory):
            if next_a is not None:
                agent.update(s, a, r, s_next, done, next_action=next_a)
            else:
                agent.update(s, a, r, s_next, done)

    def _run_episode(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        training: bool = True,
        decay_eps: bool = True,
        start_x_override: Optional[float] = None,
        record_trajectory: bool = False,
    ) -> Dict:
        """
        Executes one full episode headlessly.
        Never calls env.render() to preserve thousands of FPS throughput.
        """
        orig_start_x = getattr(env, "start_x", 50.0)
        goal = getattr(env, "goal", None)
        goal_x = getattr(goal, "x", 1000.0) if goal is not None else 1000.0
        items = getattr(env, "items", [])
        items_total = len(items)

        if hasattr(self.reward_strategy, "reset"):
            self.reward_strategy.reset()
        if start_x_override is not None and hasattr(env, "start_x"):
            env.start_x = start_x_override
        try:
            raw_state: RawState = env.reset()
        finally:
            if start_x_override is not None and hasattr(env, "start_x"):
                env.start_x = orig_start_x
        start_x = orig_start_x
        discrete_state = self.discretizer.discretize(raw_state)

        total_reward = 0.0
        steps = 0
        max_x = raw_state.player_x
        obstacle_hits = 0
        items_collected = 0
        done = False
        last_events: Optional[GameEvents] = None
        trajectory: List[Tuple[Any, int, float, Any, bool, Optional[int]]] = []

        use_on_policy_next = (
            training
            and (self.agent_type == "hybrid")
            and (self.hybrid_mode in ("linear", "hazard_gated"))
        )

        if use_on_policy_next:
            action = agent.select_action(discrete_state)
            while not done:
                macro_reward = 0.0
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
                next_action = None if done else agent.select_action(next_discrete_state)

                if training:
                    agent.update(
                        discrete_state,
                        action,
                        macro_reward,
                        next_discrete_state,
                        done,
                        next_action=next_action,
                    )
                    if record_trajectory:
                        trajectory.append(
                            (discrete_state, action, macro_reward, next_discrete_state, done, next_action)
                        )

                discrete_state = next_discrete_state
                action = next_action
                total_reward += macro_reward
        else:
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
                    if record_trajectory:
                        trajectory.append(
                            (discrete_state, action, macro_reward, next_discrete_state, done, None)
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

        res = {
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
        if record_trajectory:
            res["trajectory"] = trajectory
        return res

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

        env = self._create_env(level_id, map_seed=None)
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

    def evaluate_blind_seeds(
        self,
        agent: BaseAgent,
        level_id: int,
        seeds: Optional[List[int]] = None,
    ) -> Dict:
        """
        Evaluates the agent greedily across held-out procedural blind test seeds.
        Restores agent mode and epsilon afterward.
        """
        eval_seeds = list(seeds) if seeds is not None else list(self.blind_eval_seeds)
        if not eval_seeds:
            return {
                "wins": 0,
                "total_seeds": 0,
                "win_rate": 0.0,
                "winning_seeds": [],
                "avg_reward": 0.0,
                "avg_progress_pct": 0.0,
                "avg_completion_steps": -1.0,
                "seed_results": [],
            }

        prev_training = getattr(agent, "training", True)
        prev_epsilon = getattr(agent, "epsilon", 1.0)
        agent.set_mode(False)

        wins = 0
        winning_seeds: List[int] = []
        total_rewards: List[float] = []
        progress_list: List[float] = []
        completion_steps: List[int] = []
        seed_results: List[Dict[str, Any]] = []

        try:
            for s in eval_seeds:
                env = self._create_env(level_id, map_seed=s)
                try:
                    res = self._run_episode(env, agent, training=False)
                finally:
                    env.close()

                total_rewards.append(res["total_reward"])
                progress_list.append(res["progress_pct"])
                if res["won"]:
                    wins += 1
                    winning_seeds.append(s)
                    completion_steps.append(res["steps"])
                seed_results.append({
                    "level_id": level_id,
                    "seed": s,
                    "won": bool(res["won"]),
                    "total_reward": round(float(res["total_reward"]), 2),
                    "steps": int(res["steps"]),
                    "progress_pct": round(float(res["progress_pct"]), 2),
                    "items_collected": int(res["items_collected"]),
                    "outcome": res["outcome"],
                })
        finally:
            agent.set_mode(prev_training)
            setattr(agent, "epsilon", prev_epsilon)

        total = len(eval_seeds)
        return {
            "wins": wins,
            "total_seeds": total,
            "win_rate": round(float((wins / total) * 100.0), 2),
            "winning_seeds": winning_seeds,
            "avg_reward": round(float(np.mean(total_rewards)), 2) if total_rewards else 0.0,
            "avg_progress_pct": round(float(np.mean(progress_list)), 2) if progress_list else 0.0,
            "avg_completion_steps": (
                round(float(np.mean(completion_steps)), 1) if completion_steps else -1.0
            ),
            "seed_results": seed_results,
        }

    def _build_eval_summary_entry(
        self,
        agent: BaseAgent,
        level_id: int,
        untrained_baseline: Dict,
    ) -> Dict:
        """Builds evaluation_summary entry for a level, including blind test seeds if configured."""
        eval_trained = self.evaluate_agent(agent, level_id, self.final_eval_episodes)
        entry: Dict[str, Any] = {
            "untrained_win_rate": round(float(untrained_baseline["win_rate"]), 2),
            "trained_win_rate": round(float(eval_trained["win_rate"]), 2),
            "untrained_avg_reward": round(float(untrained_baseline["avg_reward"]), 2),
            "trained_avg_reward": round(float(eval_trained["avg_reward"]), 2),
            "avg_completion_steps": (
                round(float(eval_trained["avg_completion_steps"]), 1)
                if eval_trained["avg_completion_steps"] > 0
                else -1.0
            ),
            "avg_items_collected": round(float(eval_trained["avg_items_collected"]), 2),
        }
        if self.blind_eval_seeds and 1 <= level_id <= 7:
            blind_res = self.evaluate_blind_seeds(agent, level_id, self.blind_eval_seeds)
            entry["blind_eval"] = blind_res
            print(
                f"[Trainer] [Blind Eval] Level {level_id}: "
                f"{blind_res['wins']}/{blind_res['total_seeds']} wins ({blind_res['win_rate']:.1f}%) | "
                f"Avg Prog: {blind_res['avg_progress_pct']:.1f}% | Winning seeds: {blind_res['winning_seeds']}"
            )
        return entry

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

        # Canonical production run strictly if curriculum + balanced + tabular
        if mode == "curriculum" and self.reward_name == "balanced" and self.agent_tag == "tabular":
            is_canonical_production = True
        else:
            is_canonical_production = False

        run_cp_dir = Path(self.checkpoints_dir) / self.agent_tag / f"{mode}_{self.reward_name}"
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
            orig_eps_decay = float(getattr(agent, "epsilon_decay", 0.997))
            winning_trajectories: Dict[int, deque] = {l: deque(maxlen=5) for l in levels}
            # 4-tuple ranking: (solid_levels>=80%, won_levels>0%, avg_win_rate, avg_eval_reward)
            best_global_score = (0, 0, -1.0, float("-inf"))

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

                current_map_seed = resolve_map_seed(
                    level_id=lvl,
                    level_episode=1,
                    map_seed=self.map_seed,
                    map_seed_interval=self.map_seed_interval,
                    fallback_seed=self.seed,
                )
                env = self._create_env(lvl, map_seed=current_map_seed)
                try:
                    reverse_spawns = self._compute_reverse_spawns(
                        env, lvl, episodes_per_level, curriculum=True
                    )
                    seg_idx = 0
                    seg_ep = 0
                    seg_cap = max(1, episodes_per_level // 7)

                    for lvl_ep in range(1, episodes_per_level + 1):
                        global_episode += 1
                        seg_ep += 1

                        next_map_seed = resolve_map_seed(
                            level_id=lvl,
                            level_episode=lvl_ep,
                            map_seed=self.map_seed,
                            map_seed_interval=self.map_seed_interval,
                            fallback_seed=self.seed,
                        )
                        if next_map_seed != current_map_seed:
                            env.close()
                            current_map_seed = next_map_seed
                            env = self._create_env(lvl, map_seed=current_map_seed)
                            reverse_spawns = self._compute_reverse_spawns(
                                env, lvl, episodes_per_level, curriculum=True
                            )
                            winning_trajectories[lvl].clear()
                            if hasattr(agent, "epsilon"):
                                agent.epsilon = max(float(getattr(agent, "epsilon", 0.0)), 0.25)
                            print(
                                f"[Trainer] Level {lvl} ep {lvl_ep}: rotated map seed -> {current_map_seed}"
                            )

                        # Linear learning-rate decay (alpha_start -> alpha_min) per stage
                        if hasattr(agent, "alpha"):
                            agent.alpha = self.alpha_min + (alpha_start - self.alpha_min) * (
                                1.0 - (lvl_ep / max(1, episodes_per_level))
                            )

                        spawn_x = reverse_spawns[seg_idx]
                        if hasattr(agent, "epsilon_decay"):
                            agent.epsilon_decay = 0.985 if spawn_x is not None else orig_eps_decay

                        ep_res = self._run_episode(
                            env,
                            agent,
                            training=True,
                            decay_eps=True,
                            start_x_override=spawn_x,
                            record_trajectory=True,
                        )

                        # Self-imitation replay on discovered winning trajectories
                        if ep_res["won"] and ep_res.get("trajectory"):
                            if spawn_x is None:
                                winning_trajectories[lvl].append(ep_res["trajectory"])
                            self._replay_trajectory(agent, ep_res["trajectory"])
                        elif winning_trajectories[lvl]:
                            self._replay_trajectory(agent, random.choice(winning_trajectories[lvl]))

                        # Performance-gated reverse curriculum advancement
                        if spawn_x is not None and (seg_ep % 25 == 0 or seg_ep >= seg_cap):
                            solved_seg = False
                            if seg_ep % 25 == 0:
                                prev_t = getattr(agent, "training", True)
                                agent.set_mode(False)
                                ev_seg = self._run_episode(
                                    env, agent, training=False, decay_eps=False, start_x_override=spawn_x
                                )
                                agent.set_mode(prev_t)
                                solved_seg = ev_seg["won"]
                            if solved_seg or seg_ep >= seg_cap:
                                seg_idx = min(seg_idx + 1, len(reverse_spawns) - 1)
                                seg_ep = 0
                                if hasattr(agent, "epsilon"):
                                    agent.epsilon = 0.30 if reverse_spawns[seg_idx] is not None else 0.25

                        # Round-robin Background Replay across ALL prior levels (Curriculum Mode Only)
                        if lvl_idx > 0 and self.replay_ratio > 0.0:
                            if random.random() < max(self.replay_ratio, 0.5):
                                replay_lvl = levels[lvl_ep % lvl_idx]
                                replay_seed = resolve_map_seed(
                                    level_id=replay_lvl,
                                    level_episode=lvl_ep,
                                    map_seed=self.map_seed,
                                    map_seed_interval=self.map_seed_interval,
                                    fallback_seed=self.seed,
                                )
                                replay_env = self._create_env(replay_lvl, map_seed=replay_seed)
                                saved_eps = getattr(agent, "epsilon", 0.0)
                                setattr(agent, "epsilon", 0.0)
                                try:
                                    rep_res = self._run_episode(
                                        replay_env,
                                        agent,
                                        training=True,
                                        decay_eps=False,
                                        record_trajectory=True,
                                    )
                                    if rep_res["won"] and rep_res.get("trajectory"):
                                        winning_trajectories[replay_lvl].append(rep_res["trajectory"])
                                    elif winning_trajectories[replay_lvl]:
                                        self._replay_trajectory(
                                            agent, random.choice(winning_trajectories[replay_lvl])
                                        )
                                finally:
                                    setattr(agent, "epsilon", saved_eps)
                                    replay_env.close()

                        recent_wins.append(1 if ep_res["won"] else 0)
                        rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                        rec = {
                            "global_episode": global_episode,
                            "level_episode": lvl_ep,
                            "level_id": lvl,
                            "map_seed": current_map_seed,
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
                                "agent_tag": self.agent_tag,
                                "agent_type": self.agent_type,
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

                            # 2. Update global 4-tuple best: (solid_levels>=80%, won_levels>0%, avg_win_rate, avg_eval_reward)
                            # Prevents forgetting earlier levels from overwriting a stronger multi-level checkpoint
                            levels_solid_count = sum(1 for r in stage_evals if r["win_rate"] >= 80.0)
                            levels_won_count = sum(1 for r in stage_evals if r["win_rate"] > 0.0)
                            candidate_global_score = (
                                levels_solid_count,
                                levels_won_count,
                                stage_win_rate,
                                stage_avg_reward,
                            )
                            if candidate_global_score > best_global_score:
                                best_global_score = candidate_global_score
                                agent.save(str(run_cp_dir / "q_table_best.pkl"))
                                agent.save(str(run_cp_dir / "q_table_best.json"))
                                if is_canonical_production:
                                    root_cp = Path(self.checkpoints_dir)
                                    root_cp.mkdir(parents=True, exist_ok=True)
                                    agent.save(str(root_cp / "q_table_best.pkl"))
                                    agent.save(str(root_cp / "q_table_best.json"))



                finally:
                    env.close()

                # Record states discovered up to this level
                states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                # Level completion trained evaluation
                evaluation_summary[f"level_{lvl}"] = self._build_eval_summary_entry(
                    agent, lvl, untrained_baselines[lvl]
                )

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

                current_map_seed = resolve_map_seed(
                    level_id=lvl,
                    level_episode=1,
                    map_seed=self.map_seed,
                    map_seed_interval=self.map_seed_interval,
                    fallback_seed=self.seed,
                )
                env = self._create_env(lvl, map_seed=current_map_seed)
                try:
                    reverse_spawns = self._compute_reverse_spawns(
                        env, lvl, episodes_per_level, curriculum=False
                    )

                    for lvl_ep in range(1, episodes_per_level + 1):
                        global_episode += 1

                        next_map_seed = resolve_map_seed(
                            level_id=lvl,
                            level_episode=lvl_ep,
                            map_seed=self.map_seed,
                            map_seed_interval=self.map_seed_interval,
                            fallback_seed=self.seed,
                        )
                        if next_map_seed != current_map_seed:
                            env.close()
                            current_map_seed = next_map_seed
                            env = self._create_env(lvl, map_seed=current_map_seed)
                            reverse_spawns = self._compute_reverse_spawns(
                                env, lvl, episodes_per_level, curriculum=False
                            )
                            if hasattr(agent, "epsilon"):
                                agent.epsilon = max(float(getattr(agent, "epsilon", 0.0)), 0.25)
                            print(
                                f"[Trainer] Level {lvl} ep {lvl_ep}: rotated map seed -> {current_map_seed}"
                            )

                        # Linear learning-rate decay (alpha_start -> alpha_min) per stage
                        if hasattr(agent, "alpha"):
                            agent.alpha = self.alpha_min + (alpha_start - self.alpha_min) * (
                                1.0 - (lvl_ep / max(1, episodes_per_level))
                            )

                        spawn_x = None
                        if reverse_spawns:
                            frac = lvl_ep / max(1, episodes_per_level)
                            if frac <= 0.18:
                                spawn_x = reverse_spawns[0]
                            elif frac <= 0.36:
                                spawn_x = reverse_spawns[1]
                            elif frac <= 0.54:
                                spawn_x = reverse_spawns[2]

                        ep_res = self._run_episode(
                            env, agent, training=True, decay_eps=True, start_x_override=spawn_x
                        )

                        recent_wins.append(1 if ep_res["won"] else 0)
                        rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                        rec = {
                            "global_episode": global_episode,
                            "level_episode": lvl_ep,
                            "level_id": lvl,
                            "map_seed": current_map_seed,
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
                                "agent_tag": self.agent_tag,
                                "agent_type": self.agent_type,
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



                finally:
                    env.close()

                # Record states discovered for this level
                states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                # Level completion trained evaluation
                evaluation_summary[f"level_{lvl}"] = self._build_eval_summary_entry(
                    agent, lvl, untrained_baselines[lvl]
                )

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
                "agent_type": self.agent_type,
                "agent_tag": self.agent_tag,
                "hybrid_mode": self.hybrid_mode,
                "beta": round(float(self.beta), 4),
                "training_mode": mode,
                "reward_profile": self.reward_name,
                "action_repeat": self.action_repeat,
                "episodes_per_level": episodes_per_level,
                "levels": levels,
                "map_seed": self.map_seed,
                "map_seed_interval": self.map_seed_interval,
                "blind_eval_seeds": list(self.blind_eval_seeds),
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

        json_out_path = outputs_dir_path / f"training_board_{self.agent_tag}_{mode}_{self.reward_name}.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(board_data, f, indent=2)

        csv_out_path = outputs_dir_path / f"training_metrics_{self.agent_tag}_{mode}_{self.reward_name}.csv"
        csv_headers = [
            "global_episode",
            "level_episode",
            "level_id",
            "map_seed",
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

        blind_rows: List[Dict[str, Any]] = []
        for lvl in levels:
            be = evaluation_summary.get(f"level_{lvl}", {}).get("blind_eval")
            if be and be.get("seed_results"):
                blind_rows.extend(be["seed_results"])

        if blind_rows:
            blind_csv_path = outputs_dir_path / f"blind_eval_{self.agent_tag}_{mode}_{self.reward_name}.csv"
            blind_headers = [
                "level_id",
                "seed",
                "won",
                "total_reward",
                "steps",
                "progress_pct",
                "items_collected",
                "outcome",
            ]
            with open(blind_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=blind_headers)
                writer.writeheader()
                for row in blind_rows:
                    writer.writerow(row)
            print(f"[Trainer] Exported Blind Eval CSV to: {blind_csv_path}")

        return board_data
