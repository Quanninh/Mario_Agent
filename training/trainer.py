from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.base_env import BaseEnvironment, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy


DEFAULT_EPISODE_SCHEDULE: Dict[int, int] = {
    1: 800,
    2: 1200,
    3: 1800,
    4: 7000,
    5: 30000,
    6: 3500,
}
DEFAULT_LEVEL_EPISODES: Dict[int, int] = DEFAULT_EPISODE_SCHEDULE

OUTCOME_KEYS: Tuple[str, ...] = (
    "win",
    "ghost_death",
    "fire_death",
    "laser_death",
    "wisp_death",
    "monster_death",
    "pit_death",
    "timeout",
)


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
        trace_decay: float = 0.0,
        trace_decay_min: float = 0.35,
        lambda_polish_zero: bool = True,
        polish_epsilon_min: float = 0.005,
        polish_anchor_best: bool = True,
        main_collapse_guard: bool = True,
        max_steps_cap: Optional[int] = 2400,
        action_repeat: int = 1,
        rewarm_epsilon: float = 0.30,
        rewarm_epsilon_ghost: float = 0.35,
        rewarm_epsilon_hard: float = 0.45,
        rewarm_epsilon_l5: float = 0.40,
        rewarm_epsilon_l6: float = 0.40,
        alpha_min: float = 0.003,
        replay_ratio: float = 0.35,
        eval_interval: int = 25,
        eval_episodes_periodic: int = 10,
        final_eval_episodes: int = 50,
        untrained_eval_episodes: int = 30,
        checkpoint_interval: int = 200,
        checkpoints_dir: str = "checkpoints",
        outputs_dir: str = "outputs",
        seed: Optional[int] = 42,
    ) -> None:
        self.env_factory = env_factory
        self.discretizer = discretizer
        self.reward_strategy = reward_strategy
        self.reward_name = reward_name
        self.agent_type = agent_type
        self.hybrid_mode = hybrid_mode
        self.beta = beta
        self.trace_decay = float(trace_decay)
        self.trace_decay_min = float(trace_decay_min)
        self.lambda_polish_zero = bool(lambda_polish_zero)
        self.polish_epsilon_min = float(polish_epsilon_min)
        self.polish_anchor_best = bool(polish_anchor_best)
        self.main_collapse_guard = bool(main_collapse_guard)
        if max_steps_cap is not None and max_steps_cap <= 0:
            self.max_steps_cap: Optional[int] = None
        else:
            self.max_steps_cap = max_steps_cap
        self.agent_tag = "tabular" if agent_type == "tabular" else f"hybrid_{hybrid_mode}"
        self.action_repeat = max(1, action_repeat)
        self.rewarm_epsilon = rewarm_epsilon
        self.rewarm_epsilon_ghost = rewarm_epsilon_ghost
        self.rewarm_epsilon_hard = rewarm_epsilon_hard
        self.rewarm_epsilon_l5 = rewarm_epsilon_l5
        self.rewarm_epsilon_l6 = rewarm_epsilon_l6
        self.alpha_min = alpha_min
        self.replay_ratio = replay_ratio
        self.eval_interval = eval_interval
        self.eval_episodes_periodic = eval_episodes_periodic
        self.final_eval_episodes = final_eval_episodes
        self.untrained_eval_episodes = untrained_eval_episodes
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints_dir = checkpoints_dir
        self.outputs_dir = outputs_dir
        self.seed = 42 if seed is None else int(seed)

    @staticmethod
    def _classify_outcome(last_events: Optional[GameEvents]) -> str:
        """Maps terminal GameEvents flags into discrete outcome categories."""
        if last_events is None:
            return "timeout"
        if last_events.reached_goal:
            return "win"
        if last_events.hit_ghost:
            return "ghost_death"
        if getattr(last_events, "hit_fire", False):
            return "fire_death"
        if getattr(last_events, "hit_laser", False):
            return "laser_death"
        if getattr(last_events, "hit_wisp", False):
            return "wisp_death"
        if getattr(last_events, "hit_monster", False):
            return "monster_death"
        if last_events.fell_in_pit:
            return "pit_death"
        return "timeout"

    @staticmethod
    def _compute_stage_epsilon_decay(eps_start: float, stage_episodes: int) -> float:
        """Computes per-stage epsilon_decay reaching ~0.015 at 85% polish onset."""
        polish_start_ep = max(1, int(0.85 * stage_episodes))
        return float(
            np.clip(
                (0.015 / max(eps_start, 0.016)) ** (1.0 / polish_start_ep),
                0.90,
                0.99998,
            )
        )

    @staticmethod
    def _compute_stage_trace_decay(
        trace_decay_start: float,
        trace_decay_min: float,
        lvl_ep: int,
        stage_episodes: int,
        lambda_polish_zero: bool = True,
    ) -> float:
        """Computes per-stage eligibility trace decay across linear or two-phase polish schedule."""
        if trace_decay_start <= 0.0:
            return 0.0
        lam_min = min(trace_decay_start, max(0.0, float(trace_decay_min)))
        phase1_end = max(1, int(0.75 * stage_episodes))
        if not lambda_polish_zero:
            if lvl_ep <= phase1_end:
                val = trace_decay_start - (trace_decay_start - lam_min) * ((lvl_ep - 1) / phase1_end)
                return max(lam_min, float(val))
            return float(lam_min)

        phase2_end = max(phase1_end, int(0.85 * stage_episodes))
        if lvl_ep <= phase1_end:
            if stage_episodes <= 1:
                return 0.0
            if phase1_end <= 1:
                return float(lam_min)
            val = trace_decay_start - (trace_decay_start - lam_min) * ((lvl_ep - 1) / (phase1_end - 1))
            return max(lam_min, float(val))
        if lvl_ep <= phase2_end:
            return float(lam_min)
        tail_len = max(1, stage_episodes - phase2_end)
        frac = (lvl_ep - phase2_end) / tail_len
        return max(0.0, float(lam_min * (1.0 - frac)))

    def _anneal_stage_hyperparams(
        self,
        agent: BaseAgent,
        alpha_start: float,
        trace_decay_start: float,
        lam_min: float,
        lvl_ep: int,
        stage_episodes: int,
        stage_polish_eps_start: Optional[float] = None,
        polish_alpha_cap: Optional[float] = None,
    ) -> None:
        """Anneals agent.alpha, agent.trace_decay, and polish-tail epsilon per stage."""
        if hasattr(agent, "alpha"):
            scheduled_alpha = self.alpha_min + (alpha_start - self.alpha_min) * (
                1.0 - (lvl_ep / max(1, stage_episodes))
            )
            if polish_alpha_cap is not None:
                scheduled_alpha = min(scheduled_alpha, float(polish_alpha_cap))
            agent.alpha = scheduled_alpha
        if hasattr(agent, "trace_decay"):
            agent.trace_decay = self._compute_stage_trace_decay(
                trace_decay_start,
                lam_min,
                lvl_ep,
                stage_episodes,
                lambda_polish_zero=self.lambda_polish_zero,
            )
            if agent.trace_decay <= 0.0:
                agent.trace_decay = 0.0
                if hasattr(agent, "reset_traces"):
                    agent.reset_traces()

        if self.lambda_polish_zero:
            phase1_end = max(1, int(0.75 * stage_episodes))
            phase2_end = max(phase1_end, int(0.85 * stage_episodes))
            if lvl_ep > phase2_end:
                if hasattr(agent, "epsilon_min"):
                    agent.epsilon_min = min(float(agent.epsilon_min), self.polish_epsilon_min)
                tail_len = max(1, stage_episodes - phase2_end)
                frac = (lvl_ep - phase2_end) / tail_len
                start_eps = (
                    float(stage_polish_eps_start)
                    if stage_polish_eps_start is not None
                    else float(getattr(agent, "epsilon", self.polish_epsilon_min))
                )
                target_eps = max(
                    self.polish_epsilon_min,
                    start_eps * (1.0 - frac) + self.polish_epsilon_min * frac,
                )
                if hasattr(agent, "epsilon"):
                    agent.epsilon = min(float(agent.epsilon), target_eps)

    @staticmethod
    def _select_replay_level(
        replay_candidates: List[int],
        current_level: int = 1,
        prior_low_retention: Optional[Dict[int, bool]] = None,
    ) -> int:
        """Samples a prior curriculum level with linear weights, doubled if retention < 95% in Stage >=4."""
        if len(replay_candidates) == 1:
            return replay_candidates[0]
        weights: List[float] = []
        for idx, p_lvl in enumerate(replay_candidates):
            w = float(idx + 1)
            if (
                current_level >= 4
                and prior_low_retention is not None
                and prior_low_retention.get(p_lvl, False)
            ):
                w *= 2.0
            weights.append(w)
        return random.choices(replay_candidates, weights=weights, k=1)[0]

    @staticmethod
    def _compute_top_death_clusters(stage_records: List[Dict]) -> Dict[str, List[Dict[str, int]]]:
        """Summarizes top 3 10px-binned max_x death locations per non-win outcome over the final 500 eps."""
        clusters: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for rec in stage_records[-500:]:
            outcome = rec.get("outcome", "timeout")
            if outcome == "win":
                continue
            x_bin = int(float(rec.get("max_x_reached", 0.0)) // 10) * 10
            clusters[outcome][x_bin] += 1
        result: Dict[str, List[Dict[str, int]]] = {}
        for outcome, bin_counts in clusters.items():
            sorted_bins = sorted(bin_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            result[outcome] = [{"x_bin": xb, "count": cnt} for xb, cnt in sorted_bins]
        return result

    @staticmethod
    def _score_curriculum_retention(
        retention_dict: Dict[str, Dict], levels: List[int]
    ) -> Tuple[float, int, float]:
        """Computes (mean_win_rate, count_levels_ge_90, mean_avg_reward) across all levels."""
        if not levels:
            return (0.0, 0, 0.0)
        wrs = [float(retention_dict.get(f"level_{u}", {}).get("win_rate", 0.0)) for u in levels]
        rews = [float(retention_dict.get(f"level_{u}", {}).get("avg_reward", 0.0)) for u in levels]
        ge90 = sum(1 for w in wrs if w >= 90.0)
        return (round(float(np.mean(wrs)), 2), ge90, round(float(np.mean(rews)), 2))

    @staticmethod
    def _should_promote_curriculum_best(
        best_score: Tuple[float, int, float],
        best_ret: Optional[Dict[str, Dict]],
        cand_score: Tuple[float, int, float],
        cand_ret: Dict[str, Dict],
        levels: List[int],
    ) -> bool:
        """Applies non-regression guard before promoting a later curriculum stage checkpoint."""
        if best_ret is None:
            return True
        protected_lvls = [
            u for u in levels if float(best_ret.get(f"level_{u}", {}).get("win_rate", 0.0)) >= 90.0
        ]
        all_ge_90 = all(
            float(cand_ret.get(f"level_{u}", {}).get("win_rate", 0.0)) >= 90.0
            for u in protected_lvls
        )
        all_ge_80 = all(
            float(cand_ret.get(f"level_{u}", {}).get("win_rate", 0.0)) >= 80.0
            for u in protected_lvls
        )
        improves_by_5 = (cand_score[0] - best_score[0]) >= 5.0
        passes_guard = all_ge_90 or (improves_by_5 and all_ge_80)
        return bool(passes_guard and (cand_score >= best_score))

    def _evaluate_retention_deterministic(
        self, eval_agent: BaseAgent, levels: List[int], episodes: int
    ) -> Dict[str, Dict]:
        """Evaluates retention across all levels under a deterministic RNG state without perturbing training."""
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(12345)
            np.random.seed(12345)
            return {
                f"level_{r_lvl}": self.evaluate_agent(eval_agent, r_lvl, episodes)
                for r_lvl in levels
            }
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _evaluate_stage_completion(
        self,
        agent: BaseAgent,
        agent_factory: Callable[[], BaseAgent],
        lvl: int,
        untrained_baseline: Dict,
        best_cp_dir: Path,
    ) -> Tuple[Dict, Optional[BaseAgent]]:
        """Evaluates final end-of-stage agent and saved q_table_lvl{lvl}_best checkpoint."""
        eval_trained = self.evaluate_agent(agent, lvl, self.final_eval_episodes)

        best_cp_pkl = best_cp_dir / f"q_table_lvl{lvl}_best.pkl"
        best_cp_json = best_cp_dir / f"q_table_lvl{lvl}_best.json"
        best_cp_path = best_cp_pkl if best_cp_pkl.exists() else best_cp_json

        best_eval_agent: Optional[BaseAgent] = None
        if best_cp_path.exists():
            best_eval_agent = agent_factory()
            best_eval_agent.load(str(best_cp_path))
            eval_best = self.evaluate_agent(best_eval_agent, lvl, self.final_eval_episodes)
        else:
            eval_best = eval_trained

        summary_entry = {
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
            "outcome_counts": eval_trained["outcome_counts"],
            "best_checkpoint_win_rate": round(float(eval_best["win_rate"]), 2),
            "best_checkpoint_avg_reward": round(float(eval_best["avg_reward"]), 2),
            "best_checkpoint_completion_steps": (
                round(float(eval_best["avg_completion_steps"]), 1)
                if eval_best["avg_completion_steps"] > 0
                else -1.0
            ),
            "best_checkpoint_items_collected": round(float(eval_best["avg_items_collected"]), 2),
            "best_checkpoint_outcome_counts": eval_best["outcome_counts"],
        }
        return summary_entry, best_eval_agent

    def _get_stage_eval_interval(self, lvl_ep: int, stage_episodes: int) -> int:
        """Returns periodic eval interval for lvl_ep (coarser main phase for >10k stages)."""
        polish_iv = max(1, int(self.eval_interval))
        if stage_episodes <= 10000:
            return polish_iv
        phase2_end = max(1, int(0.85 * stage_episodes))
        if lvl_ep <= phase2_end:
            return max(polish_iv, 100)
        return polish_iv

    def _get_stage_eval_interval_summary(self, stage_episodes: int) -> Dict[str, int]:
        """Summarizes main and polish eval intervals for metadata reporting."""
        polish_iv = max(1, int(self.eval_interval))
        main_iv = max(polish_iv, 100) if stage_episodes > 10000 else polish_iv
        return {"main": main_iv, "polish": polish_iv}

    def _get_rewarm_epsilon(self, lvl: int) -> float:
        """Returns order-agnostic curriculum re-warm target strictly by level_id."""
        rewarm_map = {
            2: self.rewarm_epsilon,
            3: self.rewarm_epsilon_ghost,
            4: self.rewarm_epsilon_hard,
            5: self.rewarm_epsilon_l5,
            6: self.rewarm_epsilon_l6,
        }
        return float(rewarm_map.get(lvl, self.rewarm_epsilon_hard if lvl >= 4 else self.rewarm_epsilon))

    @staticmethod
    def _clone_q_table(agent: BaseAgent) -> Dict[Any, np.ndarray]:
        """Creates an independent deep copy of agent.q_table."""
        return {state_key: q_vals.copy() for state_key, q_vals in agent.q_table.items()}

    @staticmethod
    def _restore_q_table_dict(agent: BaseAgent, saved_q_table: Dict[Any, np.ndarray]) -> None:
        """Restores cloned Q-table weights into live agent and resets traces/training mode."""
        num_act = getattr(agent, "num_actions", 6)
        copied = defaultdict(lambda n=num_act: np.zeros(n, dtype=np.float32))
        for state_key, q_vals in saved_q_table.items():
            copied[state_key] = q_vals.copy()
        agent.q_table = copied
        if hasattr(agent, "reset_traces"):
            agent.reset_traces()
        if hasattr(agent, "set_mode"):
            agent.set_mode(True)

    @staticmethod
    def _copy_q_table_weights(agent: BaseAgent, source_agent: BaseAgent) -> None:
        """Copies source_agent Q-table weights into live agent and resets traces/training mode."""
        Trainer._restore_q_table_dict(agent, source_agent.q_table)

    @staticmethod
    def _compute_stage_anchor_score(
        lvl_idx: int,
        prev_avg_win_rate: float,
        entry_prev_win_rate: float,
        curr_win_rate: float,
        curr_avg_reward: float,
    ) -> Tuple[int, float, float]:
        """Ranks stage anchors by (retention_ok, curr_win_rate, curr_avg_reward)."""
        retention_ok = (
            1
            if (lvl_idx == 0 or prev_avg_win_rate >= max(0.0, 0.90 * entry_prev_win_rate - 1e-6))
            else 0
        )
        return (retention_ok, float(curr_win_rate), float(curr_avg_reward))

    def _reload_stage_best_weights(
        self,
        agent: BaseAgent,
        agent_factory: Callable[[], BaseAgent],
        cp_dir: Path,
        lvl: int,
        stage_best_q_table: Optional[Dict[Any, np.ndarray]] = None,
    ) -> bool:
        """Reloads stage_best_q_table or q_table_lvl{lvl}_best preserving live hyperparams."""
        if stage_best_q_table is not None:
            self._restore_q_table_dict(agent, stage_best_q_table)
            return True
        best_cp_pkl = cp_dir / f"q_table_lvl{lvl}_best.pkl"
        best_cp_json = cp_dir / f"q_table_lvl{lvl}_best.json"
        best_cp_path = best_cp_pkl if best_cp_pkl.exists() else best_cp_json
        if not best_cp_path.exists():
            return False
        ref_agent = agent_factory()
        ref_agent.load(str(best_cp_path))
        self._copy_q_table_weights(agent, ref_agent)
        return True

    def _handle_mid_polish_regression(
        self,
        agent: BaseAgent,
        agent_factory: Callable[[], BaseAgent],
        cp_dir: Path,
        lvl: int,
        cur_score: Tuple[float, float],
        best_score: Tuple[float, float],
        polish_alpha_cap: Optional[float],
        stage_best_q_table: Optional[Dict[Any, np.ndarray]] = None,
    ) -> Tuple[bool, Optional[float]]:
        """Rolls back Q-weights and halves alpha when a periodic polish eval regresses severely."""
        eval_win_rate, eval_avg_reward = cur_score
        best_win_rate, best_avg_reward = best_score
        win_regressed = eval_win_rate < (best_win_rate - 20.0)
        reward_regressed = (best_avg_reward > 200.0) and (eval_avg_reward < 0.75 * best_avg_reward)
        if not (win_regressed or reward_regressed):
            return False, polish_alpha_cap
        if not self._reload_stage_best_weights(
            agent, agent_factory, cp_dir, lvl, stage_best_q_table=stage_best_q_table
        ):
            return False, polish_alpha_cap
        alpha_floor = max(0.001, self.alpha_min * 0.5)
        cur_alpha = float(getattr(agent, "alpha", self.alpha_min))
        halved_alpha = max(alpha_floor, cur_alpha * 0.5)
        if hasattr(agent, "alpha"):
            agent.alpha = halved_alpha
        next_cap = halved_alpha if polish_alpha_cap is None else min(polish_alpha_cap, halved_alpha)
        return True, next_cap

    def _check_main_phase_collapse_recovery(
        self,
        agent: BaseAgent,
        stage_best_q_table: Optional[Dict[Any, np.ndarray]],
        stage_episodes: int,
        lvl_ep: int,
        last_main_recovery_ep: int,
        main_phase_collapse_recoveries: int,
        curr_win_rate: float,
        curr_avg_reward: float,
        best_stage_curr_win_rate: float,
        best_stage_curr_reward: float,
        stage_best_probe_max_x: float,
        recent_train_max_x: Any,
    ) -> bool:
        """Recovers Q-weights and applies bounded epsilon nudge on verified main-phase collapse."""
        if not self.main_collapse_guard or stage_episodes < 5000:
            return False
        main_start = int(0.25 * stage_episodes)
        main_end = int(0.85 * stage_episodes)
        if not (main_start <= lvl_ep < main_end):
            return False
        if (lvl_ep - last_main_recovery_ep) < 500 or main_phase_collapse_recoveries >= 4:
            return False
        if stage_best_q_table is None:
            return False
        if not (best_stage_curr_reward >= 400.0 or best_stage_curr_win_rate >= 20.0):
            return False

        eval_collapsed = (
            (curr_win_rate == 0.0 and best_stage_curr_win_rate >= 20.0)
            or (curr_avg_reward < 0.65 * best_stage_curr_reward)
        )
        if not eval_collapsed:
            return False
        train_median_x = float(np.median(list(recent_train_max_x))) if recent_train_max_x else 0.0
        if not (train_median_x < 0.65 * stage_best_probe_max_x):
            return False

        self._restore_q_table_dict(agent, stage_best_q_table)
        cur_eps = float(getattr(agent, "epsilon", 0.0))
        if hasattr(agent, "epsilon"):
            agent.epsilon = max(cur_eps, min(0.12, 2.0 * cur_eps))
        return True

    @staticmethod
    def _build_episode_record(
        global_episode: int,
        lvl_ep: int,
        lvl: int,
        ep_res: Dict,
        agent: BaseAgent,
        rolling_win_rate: float,
    ) -> Dict:
        return {
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
        env_max_steps = getattr(env, "max_steps", 1000)
        step_cap = (
            min(env_max_steps, self.max_steps_cap)
            if self.max_steps_cap is not None
            else env_max_steps
        )

        raw_state: RawState = env.reset()
        if hasattr(self.discretizer, "reset"):
            self.discretizer.reset()
        if hasattr(agent, "reset_traces"):
            agent.reset_traces()
        discrete_state = self.discretizer.discretize(raw_state)

        total_reward = 0.0
        steps = 0
        max_x = raw_state.player_x
        obstacle_hits = 0
        items_collected = 0
        ep_done = False
        last_events: Optional[GameEvents] = None

        use_on_policy_next = (
            training
            and (self.agent_type == "hybrid")
            and (self.hybrid_mode in ("linear", "hazard_gated"))
        )

        if use_on_policy_next:
            action = agent.select_action(discrete_state)
            while not ep_done:
                macro_reward = 0.0
                for _ in range(self.action_repeat):
                    next_raw_state, events, done = env.step(action)
                    steps += 1
                    ep_done = bool(done or steps >= step_cap)
                    step_reward = self.reward_strategy.calculate_reward(
                        raw_state, action, next_raw_state, events, ep_done
                    )
                    macro_reward += step_reward
                    max_x = max(max_x, next_raw_state.player_x)
                    if events.hit_obstacle:
                        obstacle_hits += 1
                    if events.collected_item:
                        items_collected += 1
                    last_events = events
                    raw_state = next_raw_state
                    if ep_done:
                        break

                next_discrete_state = self.discretizer.discretize(raw_state)
                next_action = None if ep_done else agent.select_action(next_discrete_state)

                if training:
                    agent.update(
                        discrete_state,
                        action,
                        macro_reward,
                        next_discrete_state,
                        ep_done,
                        next_action=next_action,
                    )

                discrete_state = next_discrete_state
                action = next_action
                total_reward += macro_reward
        else:
            while not ep_done:
                action = agent.select_action(discrete_state)

                macro_reward = 0.0
                # Execute action with action repeat
                for _ in range(self.action_repeat):
                    next_raw_state, events, done = env.step(action)
                    steps += 1
                    ep_done = bool(done or steps >= step_cap)
                    step_reward = self.reward_strategy.calculate_reward(
                        raw_state, action, next_raw_state, events, ep_done
                    )
                    macro_reward += step_reward
                    max_x = max(max_x, next_raw_state.player_x)
                    if events.hit_obstacle:
                        obstacle_hits += 1
                    if events.collected_item:
                        items_collected += 1
                    last_events = events
                    raw_state = next_raw_state
                    if ep_done:
                        break

                next_discrete_state = self.discretizer.discretize(raw_state)

                if training:
                    agent.update(
                        discrete_state, action, macro_reward, next_discrete_state, ep_done
                    )

                discrete_state = next_discrete_state
                total_reward += macro_reward

        # Exploration rate decay at episode completion
        if training and decay_eps:
            agent.decay_epsilon()

        outcome = self._classify_outcome(last_events)
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
        outcome_counts: Dict[str, int] = {k: 0 for k in OUTCOME_KEYS}
        if episodes <= 0:
            return {
                "win_rate": 0.0,
                "avg_reward": 0.0,
                "avg_completion_steps": -1.0,
                "avg_items_collected": 0.0,
                "avg_max_x": 0.0,
                "outcome_counts": outcome_counts,
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
        max_x_list: List[float] = []

        env = self.env_factory(level_id)
        try:
            for _ in range(episodes):
                res = self._run_episode(env, agent, training=False)
                total_rewards.append(res["total_reward"])
                items_collected_list.append(res["items_collected"])
                max_x_list.append(float(res.get("max_x_reached", 0.0)))
                out = res["outcome"]
                outcome_counts[out] = outcome_counts.get(out, 0) + 1
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
        avg_max_x = round(float(np.mean(max_x_list)), 1) if max_x_list else 0.0

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_completion_steps": avg_completion_steps,
            "avg_items_collected": avg_items_collected,
            "avg_max_x": avg_max_x,
            "outcome_counts": outcome_counts,
        }

    def run_experiment(
        self,
        agent_factory: Callable[[], BaseAgent],
        mode: str = "curriculum",
        levels: Optional[List[int]] = None,
        episodes_per_level: int = 1000,
        episodes_schedule: Optional[Dict[int, int]] = None,
        auto_epsilon_decay: bool = True,
        is_canonical_production: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict:
        """
        Executes a complete RL training experiment in either curriculum or separate mode.
        Exports checkpoints and Member 4 data boards (JSON & CSV).
        """
        if levels is None:
            levels = [1, 2, 3, 4, 6, 5]
        else:
            levels = list(levels)

        if episodes_schedule is not None:
            stage_budget = {
                lvl: int(episodes_schedule.get(lvl, episodes_per_level)) for lvl in levels
            }
        else:
            stage_budget = {lvl: int(episodes_per_level) for lvl in levels}
        total_schedule_episodes = sum(stage_budget.values())

        is_default_dirs = (
            Path(self.checkpoints_dir) == Path("checkpoints")
            and Path(self.outputs_dir) == Path("outputs")
        )
        # Canonical production run strictly when writing curriculum+balanced+tabular to default dirs
        if (
            is_default_dirs
            and mode == "curriculum"
            and self.reward_name == "balanced"
            and self.agent_tag == "tabular"
        ):
            is_canonical_production = True
        else:
            is_canonical_production = False

        run_cp_dir = Path(self.checkpoints_dir) / self.agent_tag / f"{mode}_{self.reward_name}"
        run_cp_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir_path = Path(self.outputs_dir)
        outputs_dir_path.mkdir(parents=True, exist_ok=True)

        episode_history: List[Dict] = []
        evaluation_summary: Dict[str, Any] = {}
        states_discovered_per_level: Dict[str, int] = {}
        epsilon_decay_per_level: Dict[str, float] = {}
        eval_interval_per_level: Dict[str, Dict[str, int]] = {}
        global_episode = 0

        # 1. Compute untrained baselines for each level
        print(f"[Trainer] Computing untrained baselines ({self.untrained_eval_episodes} eps each)...")
        untrained_baselines: Dict[int, Dict] = {}
        for lvl in levels:
            if mode == "separate":
                random.seed(self.seed + lvl)
                np.random.seed(self.seed + lvl)
            temp_agent = agent_factory()
            untrained_baselines[lvl] = self.evaluate_agent(
                temp_agent, lvl, self.untrained_eval_episodes, random_baseline=True
            )

        # ----------------------------------------------------
        # CURRICULUM MODE: One shared agent trained across levels
        # ----------------------------------------------------
        alpha_start = 0.12
        trace_decay_start = float(self.trace_decay)
        lam_min = (
            min(trace_decay_start, max(0.0, self.trace_decay_min))
            if trace_decay_start > 0.0
            else 0.0
        )
        if mode == "curriculum":
            agent = agent_factory()
            alpha_start = float(getattr(agent, "alpha", 0.12))
            base_epsilon_min = float(getattr(agent, "epsilon_min", 0.01))
            if trace_decay_start <= 0.0:
                trace_decay_start = float(getattr(agent, "trace_decay", 0.0))
                lam_min = (
                    min(trace_decay_start, max(0.0, self.trace_decay_min))
                    if trace_decay_start > 0.0
                    else 0.0
                )
            # 3-tuple ranking: (len(unlocked_levels), avg_win_rate, avg_eval_reward)
            best_global_score = (0, -1.0, float("-inf"))
            best_curriculum_ret_score = (-1.0, 0, float("-inf"))
            best_curriculum_ret_dict: Optional[Dict[str, Dict]] = None
            curriculum_best_stage: Optional[int] = levels[0] if levels else None
            replay_envs: Dict[int, BaseEnvironment] = {}

            try:
                for lvl_idx, lvl in enumerate(levels):
                    stage_episodes = stage_budget[lvl]
                    print(f"[Trainer] [Curriculum] Starting Level {lvl} ({stage_episodes} episodes)...")
                    recent_wins: deque = deque(maxlen=50)  # Reset per level
                    recent_train_max_x: deque = deque(maxlen=200)
                    stage_records: List[Dict] = []
                    prior_low_retention: Dict[int, bool] = {}
                    prev_levels = levels[:lvl_idx]
                    unlocked_levels = levels[: lvl_idx + 1]
                    entry_prev_win_rate = 0.0
                    best_stage_anchor_score: Tuple[int, float, float] = (-1, -1.0, float("-inf"))
                    stage_best_q_table: Optional[Dict[Any, np.ndarray]] = None
                    best_stage_episode = 0
                    best_stage_curr_win_rate = 0.0
                    best_stage_curr_reward = float("-inf")
                    stage_best_probe_max_x = 0.0
                    last_stage_anchor_score: Optional[Tuple[int, float, float]] = None
                    main_phase_collapse_recoveries = 0
                    last_main_recovery_ep = -10000
                    polish_anchor_reloaded = False
                    polish_regression_rollbacks = 0
                    polish_alpha_cap: Optional[float] = None
                    phase1_end = max(1, int(0.75 * stage_episodes))
                    phase2_end = max(phase1_end, int(0.85 * stage_episodes))
                    stage_polish_eps_start: Optional[float] = None
                    eval_interval_per_level[f"level_{lvl}"] = (
                        self._get_stage_eval_interval_summary(stage_episodes)
                    )

                    # Restore stage-start trace_decay and epsilon_min cleanly
                    if hasattr(agent, "epsilon_min"):
                        agent.epsilon_min = base_epsilon_min
                    if hasattr(agent, "trace_decay"):
                        agent.trace_decay = trace_decay_start
                    if hasattr(agent, "reset_traces"):
                        agent.reset_traces()

                    # If transitioning to a new level
                    if lvl_idx > 0:
                        if prev_levels:
                            entry_prev_evals = [
                                self.evaluate_agent(agent, p_lvl, self.eval_episodes_periodic)
                                for p_lvl in prev_levels
                            ]
                            entry_prev_win_rate = float(
                                np.mean([r["win_rate"] for r in entry_prev_evals])
                            )

                        # Order-agnostic zero-shot evaluation probes for key progression levels
                        if lvl in (3, 5, 6):
                            print(
                                f"[Trainer] Running Level {lvl} zero-shot evaluation "
                                f"({self.final_eval_episodes} eps)..."
                            )
                            z_res = self.evaluate_agent(agent, lvl, self.final_eval_episodes)
                            evaluation_summary[f"level_{lvl}_zero_shot"] = z_res

                        rewarm_target = self._get_rewarm_epsilon(lvl)
                        current_eps = getattr(agent, "epsilon", 0.0)
                        new_eps = max(current_eps, rewarm_target)
                        setattr(agent, "epsilon", new_eps)
                        print(
                            f"[Trainer] Re-warmed exploration epsilon: {current_eps:.3f} -> {new_eps:.3f} (target: {rewarm_target:.3f})"
                        )

                    # Stage-matched epsilon decay
                    eps_start_stage = float(getattr(agent, "epsilon", 1.0))
                    if auto_epsilon_decay and hasattr(agent, "epsilon_decay"):
                        agent.epsilon_decay = self._compute_stage_epsilon_decay(
                            eps_start_stage, stage_episodes
                        )
                    epsilon_decay_per_level[f"level_{lvl}"] = round(
                        float(getattr(agent, "epsilon_decay", 0.995)), 6
                    )

                    env = self.env_factory(lvl)
                    try:
                        for lvl_ep in range(1, stage_episodes + 1):
                            global_episode += 1
                            if lvl_ep == phase2_end + 1:
                                stage_polish_eps_start = float(getattr(agent, "epsilon", self.polish_epsilon_min))

                            self._anneal_stage_hyperparams(
                                agent,
                                alpha_start,
                                trace_decay_start,
                                lam_min,
                                lvl_ep,
                                stage_episodes,
                                stage_polish_eps_start=stage_polish_eps_start,
                                polish_alpha_cap=polish_alpha_cap,
                            )

                            # 85% Onset Anchor: reload retention-safe stage-best weights if live agent drifted
                            if (
                                self.lambda_polish_zero
                                and self.polish_anchor_best
                                and lvl_ep == (phase2_end + 1)
                                and best_stage_episode > 0
                                and last_stage_anchor_score is not None
                                and last_stage_anchor_score < best_stage_anchor_score
                            ):
                                if self._reload_stage_best_weights(
                                    agent,
                                    agent_factory,
                                    run_cp_dir,
                                    lvl,
                                    stage_best_q_table=stage_best_q_table,
                                ):
                                    polish_anchor_reloaded = True

                            ep_res = self._run_episode(env, agent, training=True, decay_eps=True)
                            recent_train_max_x.append(float(ep_res.get("max_x_reached", 0.0)))

                            # Option B Interleaved Background Replay with adaptive retention protection
                            if lvl_idx > 0 and self.replay_ratio > 0.0:
                                if random.random() < self.replay_ratio:
                                    replay_candidates = levels[:lvl_idx]
                                    if replay_candidates:
                                        replay_lvl = self._select_replay_level(
                                            replay_candidates, lvl, prior_low_retention
                                        )
                                        if replay_lvl not in replay_envs:
                                            replay_envs[replay_lvl] = self.env_factory(replay_lvl)
                                        self._run_episode(
                                            replay_envs[replay_lvl],
                                            agent,
                                            training=True,
                                            decay_eps=False,
                                        )

                            recent_wins.append(1 if ep_res["won"] else 0)
                            rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                            rec = self._build_episode_record(
                                global_episode, lvl_ep, lvl, ep_res, agent, rolling_win_rate
                            )
                            episode_history.append(rec)
                            stage_records.append(rec)

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
                                    "episodes_per_level": stage_episodes,
                                    "global_episode": global_episode,
                                    "total_episodes": total_schedule_episodes,
                                    "total_reward": round(float(ep_res["total_reward"]), 2),
                                    "rolling_win_rate_50": rolling_win_rate,
                                    "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                                    "states_discovered": len(getattr(agent, "q_table", {})),
                                })

                            # Periodic greedy evaluation for checkpointing and collapse recovery
                            stage_eval_interval = self._get_stage_eval_interval(lvl_ep, stage_episodes)
                            if (lvl_ep % stage_eval_interval == 0) or (lvl_ep == stage_episodes):
                                prev_evals = [
                                    self.evaluate_agent(agent, p_lvl, self.eval_episodes_periodic)
                                    for p_lvl in prev_levels
                                ]
                                curr_eval = self.evaluate_agent(agent, lvl, self.eval_episodes_periodic)
                                stage_evals = prev_evals + [curr_eval]

                                prev_avg_win_rate = (
                                    float(np.mean([r["win_rate"] for r in prev_evals]))
                                    if prev_evals
                                    else 0.0
                                )
                                curr_win_rate = round(float(curr_eval["win_rate"]), 2)
                                curr_avg_reward = round(float(curr_eval["avg_reward"]), 2)
                                curr_probe_max_x = round(float(curr_eval.get("avg_max_x", 0.0)), 1)

                                stage_anchor_score = self._compute_stage_anchor_score(
                                    lvl_idx,
                                    prev_avg_win_rate,
                                    entry_prev_win_rate,
                                    curr_win_rate,
                                    curr_avg_reward,
                                )

                                # 1. Update retention-safe stage_best_q_table (or guard against collapse/regression)
                                if stage_anchor_score > best_stage_anchor_score:
                                    best_stage_anchor_score = stage_anchor_score
                                    best_stage_episode = lvl_ep
                                    stage_best_q_table = self._clone_q_table(agent)
                                    best_stage_curr_win_rate = curr_win_rate
                                    best_stage_curr_reward = curr_avg_reward
                                    stage_best_probe_max_x = curr_probe_max_x
                                    agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.pkl"))
                                    agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.json"))
                                elif (
                                    self.lambda_polish_zero
                                    and self.polish_anchor_best
                                    and lvl_ep > phase2_end
                                    and best_stage_episode > 0
                                ):
                                    rolled_back, polish_alpha_cap = self._handle_mid_polish_regression(
                                        agent,
                                        agent_factory,
                                        run_cp_dir,
                                        lvl,
                                        (curr_win_rate, curr_avg_reward),
                                        (best_stage_curr_win_rate, best_stage_curr_reward),
                                        polish_alpha_cap,
                                        stage_best_q_table=stage_best_q_table,
                                    )
                                    if rolled_back:
                                        polish_regression_rollbacks += 1
                                elif lvl_ep <= phase2_end:
                                    recovered = self._check_main_phase_collapse_recovery(
                                        agent=agent,
                                        stage_best_q_table=stage_best_q_table,
                                        stage_episodes=stage_episodes,
                                        lvl_ep=lvl_ep,
                                        last_main_recovery_ep=last_main_recovery_ep,
                                        main_phase_collapse_recoveries=main_phase_collapse_recoveries,
                                        curr_win_rate=curr_win_rate,
                                        curr_avg_reward=curr_avg_reward,
                                        best_stage_curr_win_rate=best_stage_curr_win_rate,
                                        best_stage_curr_reward=best_stage_curr_reward,
                                        stage_best_probe_max_x=stage_best_probe_max_x,
                                        recent_train_max_x=recent_train_max_x,
                                    )
                                    if recovered:
                                        main_phase_collapse_recoveries += 1
                                        last_main_recovery_ep = lvl_ep

                                last_stage_anchor_score = stage_anchor_score

                                # 2. Update global 3-tuple best: (len(unlocked_levels), avg_win_rate, avg_eval_reward)
                                stage_win_rate = round(
                                    float(np.mean([r["win_rate"] for r in stage_evals])), 2
                                )
                                stage_avg_reward = round(
                                    float(np.mean([r["avg_reward"] for r in stage_evals])), 2
                                )
                                candidate_global_score = (
                                    len(unlocked_levels),
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

                            # Adaptive Replay Priority check every eval_interval * 4 episodes
                            if (
                                lvl_idx > 0
                                and self.eval_episodes_periodic > 0
                                and (lvl_ep % max(1, self.eval_interval * 4) == 0)
                            ):
                                for p_lvl in levels[:lvl_idx]:
                                    p_res = self.evaluate_agent(agent, p_lvl, self.eval_episodes_periodic)
                                    prior_low_retention[p_lvl] = float(p_res["win_rate"]) < 95.0
                    finally:
                        env.close()

                    # Record states discovered up to this level
                    states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                    # Evaluate end-of-stage final agent AND saved q_table_lvl{lvl}_best
                    summary_entry, best_eval_agent = self._evaluate_stage_completion(
                        agent, agent_factory, lvl, untrained_baselines[lvl], run_cp_dir
                    )
                    summary_entry["best_checkpoint_episode"] = (
                        best_stage_episode if best_stage_episode > 0 else stage_episodes
                    )
                    summary_entry["polish_anchor_reloaded"] = polish_anchor_reloaded
                    summary_entry["polish_regression_rollbacks"] = polish_regression_rollbacks
                    summary_entry["main_phase_collapse_recoveries"] = main_phase_collapse_recoveries
                    summary_entry["stage_best_probe_max_x"] = stage_best_probe_max_x
                    summary_entry["top_death_clusters"] = self._compute_top_death_clusters(stage_records)
                    evaluation_summary[f"level_{lvl}"] = summary_entry

                    # Multi-level retention promotion on stage-best weights (q_table_lvl{lvl}_best)
                    ret_eval_target = best_eval_agent if best_eval_agent is not None else agent
                    cand_ret_dict = self._evaluate_retention_deterministic(
                        ret_eval_target, levels, self.final_eval_episodes
                    )
                    cand_ret_score = self._score_curriculum_retention(cand_ret_dict, levels)
                    if self._should_promote_curriculum_best(
                        best_curriculum_ret_score,
                        best_curriculum_ret_dict,
                        cand_ret_score,
                        cand_ret_dict,
                        levels,
                    ):
                        best_curriculum_ret_score = cand_ret_score
                        best_curriculum_ret_dict = cand_ret_dict
                        curriculum_best_stage = lvl
                        ret_eval_target.save(str(run_cp_dir / "q_table_curriculum_best.pkl"))
                        ret_eval_target.save(str(run_cp_dir / "q_table_curriculum_best.json"))

                    # Stage-best carryover: advance into stage lvl+1 from verified q_table_lvl{lvl}_best weights
                    if (lvl_idx + 1 < len(levels)) and (best_eval_agent is not None):
                        self._copy_q_table_weights(agent, best_eval_agent)
                        print(
                            f"[Trainer] [Curriculum] Carried over q_table_lvl{lvl}_best "
                            f"({len(agent.q_table)} states) into Level {levels[lvl_idx + 1]}."
                        )
            finally:
                for r_env in replay_envs.values():
                    r_env.close()

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

            # Post-curriculum retention evaluations across all levels
            print(f"[Trainer] Evaluating post-curriculum retention on final agent across levels {levels}...")
            evaluation_summary["post_curriculum_retention"] = {
                f"level_{r_lvl}": self.evaluate_agent(agent, r_lvl, self.final_eval_episodes)
                for r_lvl in levels
            }

            best_cp_pkl = run_cp_dir / "q_table_best.pkl"
            best_cp_json = run_cp_dir / "q_table_best.json"
            best_cp_path = best_cp_pkl if best_cp_pkl.exists() else best_cp_json
            if best_cp_path.exists():
                print(f"[Trainer] Evaluating best-checkpoint retention across levels {levels}...")
                best_eval_agent = agent_factory()
                best_eval_agent.load(str(best_cp_path))
                evaluation_summary["best_checkpoint_retention"] = {
                    f"level_{r_lvl}": self.evaluate_agent(
                        best_eval_agent, r_lvl, self.final_eval_episodes
                    )
                    for r_lvl in levels
                }

            curr_best_pkl = run_cp_dir / "q_table_curriculum_best.pkl"
            curr_best_json = run_cp_dir / "q_table_curriculum_best.json"
            curr_best_path = curr_best_pkl if curr_best_pkl.exists() else curr_best_json
            if curr_best_path.exists():
                print(
                    f"[Trainer] Recorded multi-level curriculum_best_retention "
                    f"(from Stage {curriculum_best_stage}) across levels {levels}."
                )
                evaluation_summary["curriculum_best_checkpoint_stage"] = curriculum_best_stage
                if best_curriculum_ret_dict is not None:
                    evaluation_summary["curriculum_best_retention"] = best_curriculum_ret_dict
                else:
                    curr_best_agent = agent_factory()
                    curr_best_agent.load(str(curr_best_path))
                    evaluation_summary["curriculum_best_retention"] = (
                        self._evaluate_retention_deterministic(
                            curr_best_agent, levels, self.final_eval_episodes
                        )
                    )

        # ----------------------------------------------------
        # SEPARATE MODE: Independent fresh agent per level
        # ----------------------------------------------------
        else:
            agent = None
            for lvl_idx, lvl in enumerate(levels):
                random.seed(self.seed + lvl)
                np.random.seed(self.seed + lvl)
                stage_episodes = stage_budget[lvl]
                print(f"[Trainer] [Separate] Training Level {lvl} ({stage_episodes} episodes) with fresh agent...")
                agent = agent_factory()
                alpha_start = float(getattr(agent, "alpha", 0.12))
                if trace_decay_start <= 0.0:
                    trace_decay_start = float(getattr(agent, "trace_decay", 0.0))
                    lam_min = (
                        min(trace_decay_start, max(0.0, self.trace_decay_min))
                        if trace_decay_start > 0.0
                        else 0.0
                    )
                if hasattr(agent, "trace_decay"):
                    agent.trace_decay = trace_decay_start
                if hasattr(agent, "reset_traces"):
                    agent.reset_traces()

                eps_start_stage = float(getattr(agent, "epsilon", 1.0))
                if auto_epsilon_decay and hasattr(agent, "epsilon_decay"):
                    agent.epsilon_decay = self._compute_stage_epsilon_decay(
                        eps_start_stage, stage_episodes
                    )
                epsilon_decay_per_level[f"level_{lvl}"] = round(
                    float(getattr(agent, "epsilon_decay", 0.995)), 6
                )
                eval_interval_per_level[f"level_{lvl}"] = (
                    self._get_stage_eval_interval_summary(stage_episodes)
                )

                recent_wins: deque = deque(maxlen=50)  # Reset per level
                recent_train_max_x: deque = deque(maxlen=200)
                stage_records: List[Dict] = []
                best_stage_anchor_score: Tuple[int, float, float] = (-1, -1.0, float("-inf"))
                stage_best_q_table: Optional[Dict[Any, np.ndarray]] = None
                best_stage_episode = 0
                best_stage_curr_win_rate = 0.0
                best_stage_curr_reward = float("-inf")
                stage_best_probe_max_x = 0.0
                last_stage_anchor_score: Optional[Tuple[int, float, float]] = None
                main_phase_collapse_recoveries = 0
                last_main_recovery_ep = -10000
                polish_anchor_reloaded = False
                polish_regression_rollbacks = 0
                polish_alpha_cap: Optional[float] = None
                phase1_end = max(1, int(0.75 * stage_episodes))
                phase2_end = max(phase1_end, int(0.85 * stage_episodes))
                stage_polish_eps_start: Optional[float] = None

                env = self.env_factory(lvl)
                try:
                    for lvl_ep in range(1, stage_episodes + 1):
                        global_episode += 1
                        if lvl_ep == phase2_end + 1:
                            stage_polish_eps_start = float(getattr(agent, "epsilon", self.polish_epsilon_min))

                        self._anneal_stage_hyperparams(
                            agent,
                            alpha_start,
                            trace_decay_start,
                            lam_min,
                            lvl_ep,
                            stage_episodes,
                            stage_polish_eps_start=stage_polish_eps_start,
                            polish_alpha_cap=polish_alpha_cap,
                        )

                        # 85% Onset Anchor: reload stage-best weights if live agent drifted
                        if (
                            self.lambda_polish_zero
                            and self.polish_anchor_best
                            and lvl_ep == (phase2_end + 1)
                            and best_stage_episode > 0
                            and last_stage_anchor_score is not None
                            and last_stage_anchor_score < best_stage_anchor_score
                        ):
                            if self._reload_stage_best_weights(
                                agent,
                                agent_factory,
                                run_cp_dir,
                                lvl,
                                stage_best_q_table=stage_best_q_table,
                            ):
                                polish_anchor_reloaded = True

                        ep_res = self._run_episode(env, agent, training=True, decay_eps=True)
                        recent_train_max_x.append(float(ep_res.get("max_x_reached", 0.0)))

                        recent_wins.append(1 if ep_res["won"] else 0)
                        rolling_win_rate = round(float(np.mean(recent_wins) * 100.0), 2)

                        rec = self._build_episode_record(
                            global_episode, lvl_ep, lvl, ep_res, agent, rolling_win_rate
                        )
                        episode_history.append(rec)
                        stage_records.append(rec)

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
                                "episodes_per_level": stage_episodes,
                                "global_episode": global_episode,
                                "total_episodes": total_schedule_episodes,
                                "total_reward": round(float(ep_res["total_reward"]), 2),
                                "rolling_win_rate_50": rolling_win_rate,
                                "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 6),
                                "states_discovered": len(getattr(agent, "q_table", {})),
                            })

                        # Periodic greedy-eval best checkpointing per level (with collapse & polish guards)
                        stage_eval_interval = self._get_stage_eval_interval(lvl_ep, stage_episodes)
                        if (lvl_ep % stage_eval_interval == 0) or (lvl_ep == stage_episodes):
                            eval_res = self.evaluate_agent(
                                agent, lvl, self.eval_episodes_periodic
                            )
                            curr_win_rate = round(float(eval_res["win_rate"]), 2)
                            curr_avg_reward = round(float(eval_res["avg_reward"]), 2)
                            curr_probe_max_x = round(float(eval_res.get("avg_max_x", 0.0)), 1)
                            stage_anchor_score = self._compute_stage_anchor_score(
                                0, 0.0, 0.0, curr_win_rate, curr_avg_reward
                            )
                            if stage_anchor_score > best_stage_anchor_score:
                                best_stage_anchor_score = stage_anchor_score
                                best_stage_episode = lvl_ep
                                stage_best_q_table = self._clone_q_table(agent)
                                best_stage_curr_win_rate = curr_win_rate
                                best_stage_curr_reward = curr_avg_reward
                                stage_best_probe_max_x = curr_probe_max_x
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.pkl"))
                                agent.save(str(run_cp_dir / f"q_table_lvl{lvl}_best.json"))
                            elif (
                                self.lambda_polish_zero
                                and self.polish_anchor_best
                                and lvl_ep > phase2_end
                                and best_stage_episode > 0
                            ):
                                rolled_back, polish_alpha_cap = self._handle_mid_polish_regression(
                                    agent,
                                    agent_factory,
                                    run_cp_dir,
                                    lvl,
                                    (curr_win_rate, curr_avg_reward),
                                    (best_stage_curr_win_rate, best_stage_curr_reward),
                                    polish_alpha_cap,
                                    stage_best_q_table=stage_best_q_table,
                                )
                                if rolled_back:
                                    polish_regression_rollbacks += 1
                            elif lvl_ep <= phase2_end:
                                recovered = self._check_main_phase_collapse_recovery(
                                    agent=agent,
                                    stage_best_q_table=stage_best_q_table,
                                    stage_episodes=stage_episodes,
                                    lvl_ep=lvl_ep,
                                    last_main_recovery_ep=last_main_recovery_ep,
                                    main_phase_collapse_recoveries=main_phase_collapse_recoveries,
                                    curr_win_rate=curr_win_rate,
                                    curr_avg_reward=curr_avg_reward,
                                    best_stage_curr_win_rate=best_stage_curr_win_rate,
                                    best_stage_curr_reward=best_stage_curr_reward,
                                    stage_best_probe_max_x=stage_best_probe_max_x,
                                    recent_train_max_x=recent_train_max_x,
                                )
                                if recovered:
                                    main_phase_collapse_recoveries += 1
                                    last_main_recovery_ep = lvl_ep

                            last_stage_anchor_score = stage_anchor_score
                finally:
                    env.close()

                # Record states discovered for this level
                states_discovered_per_level[f"level_{lvl}"] = len(agent.q_table)

                # Evaluate end-of-stage final agent AND saved q_table_lvl{lvl}_best
                summary_entry, _ = self._evaluate_stage_completion(
                    agent, agent_factory, lvl, untrained_baselines[lvl], run_cp_dir
                )
                summary_entry["best_checkpoint_episode"] = (
                    best_stage_episode if best_stage_episode > 0 else stage_episodes
                )
                summary_entry["polish_anchor_reloaded"] = polish_anchor_reloaded
                summary_entry["polish_regression_rollbacks"] = polish_regression_rollbacks
                summary_entry["main_phase_collapse_recoveries"] = main_phase_collapse_recoveries
                summary_entry["stage_best_probe_max_x"] = stage_best_probe_max_x
                summary_entry["top_death_clusters"] = self._compute_top_death_clusters(stage_records)
                evaluation_summary[f"level_{lvl}"] = summary_entry

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

        state_space_max = int(getattr(self.discretizer, "state_space_size", 55296))
        episodes_schedule_log = {f"level_{lvl}": stage_budget[lvl] for lvl in levels}
        uniform_budget = len(set(stage_budget.values())) == 1
        meta_episodes_per_level = (
            stage_budget[levels[0]] if (levels and uniform_budget) else episodes_per_level
        )

        monster_thresh = float(
            getattr(
                self.discretizer,
                "monster_threat_distance",
                getattr(self.reward_strategy, "monster_threat_distance", 95.0),
            )
        )

        board_data = {
            "metadata": {
                "agent_type": self.agent_type,
                "agent_tag": self.agent_tag,
                "hybrid_mode": self.hybrid_mode,
                "beta": round(float(self.beta), 4),
                "trace_decay": round(float(trace_decay_start), 4),
                "trace_decay_start": round(float(trace_decay_start), 4),
                "trace_decay_min": round(float(lam_min), 4),
                "lambda_polish_zero": bool(self.lambda_polish_zero),
                "polish_epsilon_min": round(float(self.polish_epsilon_min), 6),
                "polish_anchor_best": bool(self.polish_anchor_best),
                "main_collapse_guard": bool(self.main_collapse_guard),
                "trace_decay_schedule": (
                    (
                        f"two_phase_{trace_decay_start:g}_to_{lam_min:g}_polish_to_0"
                        if self.lambda_polish_zero
                        else f"linear_to_{lam_min:.2f}_at_75pct"
                    )
                    if trace_decay_start > 0.0
                    else "constant_0.0"
                ),
                "enable_wisp_sensor": bool(getattr(self.discretizer, "enable_wisp_sensor", False)),
                "monster_threat_distance": round(monster_thresh, 2),
                "training_mode": mode,
                "reward_profile": self.reward_name,
                "action_repeat": self.action_repeat,
                "max_steps_cap": self.max_steps_cap,
                "episodes_per_level": meta_episodes_per_level,
                "episodes_schedule": episodes_schedule_log,
                "levels": levels,
                "level_order": list(levels),
                "rewarm_epsilon": round(float(self.rewarm_epsilon), 4),
                "rewarm_epsilon_ghost": round(float(self.rewarm_epsilon_ghost), 4),
                "rewarm_epsilon_hard": round(float(self.rewarm_epsilon_hard), 4),
                "rewarm_epsilon_l5": round(float(self.rewarm_epsilon_l5), 4),
                "rewarm_epsilon_l6": round(float(self.rewarm_epsilon_l6), 4),
                "replay_ratio": round(float(self.replay_ratio if mode == "curriculum" else 0.0), 4),
                "state_space_max": state_space_max,
                "num_states_discovered": total_states,
                "states_discovered_per_level": states_discovered_per_level,
                "alpha": round(float(alpha_start), 4),
                "alpha_start": round(float(alpha_start), 4),
                "alpha_min": round(float(self.alpha_min), 4),
                "gamma": round(float(getattr(agent, "gamma", 0.99)), 4) if agent else 0.99,
                "epsilon_min": round(float(getattr(agent, "epsilon_min", 0.01)), 6) if agent else 0.01,
                "epsilon_decay": round(float(getattr(agent, "epsilon_decay", 0.995)), 6) if agent else 0.995,
                "epsilon_decay_per_level": epsilon_decay_per_level,
                "eval_interval_per_level": eval_interval_per_level,
                "seed": int(self.seed),
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
