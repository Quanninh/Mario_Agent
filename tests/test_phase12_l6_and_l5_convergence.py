import os
import pickle
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from agent import TabularQAgent
from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from discretizer import MarioStateDiscretizer
from env import create_env
from rewards import BalancedRewardStrategy
from training import Trainer


def make_state(**kwargs) -> RawState:
    defaults = {
        "player_x": 100.0,
        "player_y": 288.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "is_grounded": True,
        "can_double_jump": False,
        "is_crouching": False,
        "dist_to_obstacle": -1.0,
        "obstacle_height": 0.0,
        "dist_to_pit": -1.0,
        "pit_fire_state": 0,
        "dist_to_ghost": -1.0,
        "ghost_y": 0.0,
        "dist_to_item": -1.0,
        "dist_to_laser": -1.0,
        "dist_to_wisp": -1.0,
        "wisp_dir": 0,
        "dist_to_monster": -1.0,
        "monster_y": 0.0,
        "dist_to_goal": 500.0,
    }
    defaults.update(kwargs)
    return RawState(**defaults)


class _DummyEnv(BaseEnvironment):
    def __init__(self, level_id: int = 1, max_steps: int = 5):
        self.level_id = level_id
        self.max_steps = max_steps
        self.start_x = 50.0
        self.items = []
        self.step_x = 54.0

        class _Goal:
            x = 1000.0

        self.goal = _Goal()

    def reset(self) -> RawState:
        return make_state(player_x=self.start_x)

    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
        return (
            make_state(player_x=self.step_x),
            GameEvents(dx=4.0, reached_goal=False),
            True,
        )

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestPhase12L6AndL5Convergence(unittest.TestCase):
    def test_monster_threat_distance_synced_95(self) -> None:
        disc = MarioStateDiscretizer()
        rew = BalancedRewardStrategy()

        self.assertEqual(disc.monster_threat_distance, 95.0)
        self.assertEqual(rew.monster_threat_distance, 95.0)
        self.assertEqual(disc.state_space_size, 55296)

        s_near = disc.discretize(make_state(dist_to_monster=90.0))
        s_far = disc.discretize(make_state(dist_to_monster=96.0))
        self.assertEqual(s_near[11], 1)
        self.assertEqual(s_far[11], 0)

        prev_near = make_state(player_x=100.0, is_grounded=True, dist_to_monster=90.0)
        next_near = make_state(player_x=100.0, is_grounded=False, dist_to_monster=90.0)
        r_exempt = rew.calculate_reward(prev_near, Action.JUMP, next_near, GameEvents(dx=0.0), False)

        prev_far = make_state(player_x=100.0, is_grounded=True, dist_to_monster=96.0)
        next_far = make_state(player_x=100.0, is_grounded=False, dist_to_monster=96.0)
        r_penalized = rew.calculate_reward(prev_far, Action.JUMP, next_far, GameEvents(dx=0.0), False)

        self.assertAlmostEqual(r_exempt, -0.1, places=4)
        self.assertAlmostEqual(r_penalized, -15.1, places=4)

    def test_uncapped_dynamic_epsilon_decay_scales_to_30k_and_50k(self) -> None:
        decay_30k = Trainer._compute_stage_epsilon_decay(0.40, 30000)
        self.assertGreater(decay_30k, 0.9995)
        eps_at_25500 = 0.40 * (decay_30k ** 25500)
        self.assertAlmostEqual(eps_at_25500, 0.015, delta=0.001)

        decay_50k = Trainer._compute_stage_epsilon_decay(0.40, 50000)
        self.assertGreater(decay_50k, 0.9995)
        eps_at_42500 = 0.40 * (decay_50k ** 42500)
        self.assertAlmostEqual(eps_at_42500, 0.015, delta=0.001)

    def test_decoupled_stage_anchor_prefers_current_level_when_retention_ok(self) -> None:
        score_a = Trainer._compute_stage_anchor_score(4, 100.0, 100.0, 0.0, 303.0)
        score_b = Trainer._compute_stage_anchor_score(4, 100.0, 100.0, 0.0, 744.0)
        score_c = Trainer._compute_stage_anchor_score(4, 50.0, 100.0, 0.0, 900.0)
        self.assertEqual(score_a, (1, 0.0, 303.0))
        self.assertEqual(score_b, (1, 0.0, 744.0))
        self.assertEqual(score_c, (0, 0.0, 900.0))
        self.assertGreater(score_b, score_a)
        self.assertGreater(score_a, score_c)

        s_tag = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        class _AnchorTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                lvl = getattr(env, "level_id", 1)
                if training and decay_eps:
                    if not hasattr(self, "_ep_cnt"):
                        self._ep_cnt = {}
                    self._ep_cnt[lvl] = self._ep_cnt.get(lvl, 0) + 1
                    ep = self._ep_cnt[lvl]
                    if lvl <= 4:
                        agent.q_table[s_tag][Action.STAY] = 100.0
                    else:
                        # Ep 5: Cand A; Ep 10: Cand B; Ep 15: Cand C; Ep 20: Pit-1 regression (Cand A)
                        if ep <= 5:
                            agent.q_table[s_tag][Action.STAY] = 303.0
                        elif ep <= 10:
                            agent.q_table[s_tag][Action.STAY] = 744.0
                        elif ep <= 15:
                            agent.q_table[s_tag][Action.STAY] = 900.0
                        else:
                            agent.q_table[s_tag][Action.STAY] = 303.0
                return res

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                marker = float(agent.q_table.get(s_tag, np.zeros(6))[Action.STAY])
                if level_id <= 4:
                    wr = 50.0 if (850.0 < marker < 950.0) else 100.0
                    rew = 1000.0 if wr == 100.0 else 300.0
                    mx = 1200.0
                else:
                    wr = 0.0
                    if 700.0 < marker < 800.0:
                        rew, mx = 744.0, 898.0
                    elif marker >= 850.0:
                        rew, mx = 900.0, 1020.0
                    else:
                        rew, mx = 303.0, 426.0
                return {
                    "win_rate": wr,
                    "avg_reward": rew,
                    "avg_completion_steps": -1.0,
                    "avg_items_collected": 0.0,
                    "avg_max_x": mx,
                    "outcome_counts": {"pit_death": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _AnchorTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                replay_ratio=0.0,
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[1, 2, 3, 4, 5],
                episodes_schedule={1: 5, 2: 5, 3: 5, 4: 5, 5: 20},
            )

        l5 = board["evaluation_summary"]["level_5"]
        self.assertEqual(l5["best_checkpoint_episode"], 10)
        self.assertEqual(l5["best_checkpoint_avg_reward"], 744.0)
        self.assertEqual(l5["stage_best_probe_max_x"], 898.0)

    def test_main_phase_collapse_guard_triggers_restores_and_respects_cooldown(self) -> None:
        s_tag = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        recovery_events: List[Dict[str, float]] = []

        class _CollapseGuardTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps:
                    if not hasattr(self, "_ep"):
                        self._ep = 0
                    self._ep += 1
                    ep = self._ep
                    env.step_x = 900.0 if ep <= 1000 else 420.0
                    # Force low epsilon right before eligible eval checkpoints to inspect nudge
                    if ep >= 1249:
                        agent.epsilon = 0.04
                    res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                    if ep <= 1000:
                        agent.q_table[s_tag][Action.STAY] = 900.0
                    else:
                        agent.q_table[s_tag][Action.STAY] = -900.0
                        agent._traces[(s_tag, Action.STAY)] = 0.75
                    return res
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

            def _check_main_phase_collapse_recovery(self, **kwargs):
                agent = kwargs["agent"]
                pre_eps = float(agent.epsilon)
                pre_alpha = float(agent.alpha)
                fired = super()._check_main_phase_collapse_recovery(**kwargs)
                if fired:
                    recovery_events.append({
                        "ep": kwargs["lvl_ep"],
                        "restored_marker": float(agent.q_table[s_tag][Action.STAY]),
                        "traces_len": len(agent._traces),
                        "pre_alpha": pre_alpha,
                        "post_alpha": float(agent.alpha),
                        "pre_eps": pre_eps,
                        "post_eps": float(agent.epsilon),
                    })
                return fired

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                val = float(agent.q_table.get(s_tag, np.zeros(6))[Action.STAY])
                good = val > 500.0
                return {
                    "win_rate": 40.0 if good else 0.0,
                    "avg_reward": 1200.0 if good else 250.0,
                    "avg_completion_steps": 100.0 if good else -1.0,
                    "avg_items_collected": 0.0,
                    "avg_max_x": 900.0 if good else 420.0,
                    "outcome_counts": {"pit_death": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _CollapseGuardTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                main_collapse_guard=True,
                trace_decay=0.65,
                trace_decay_min=0.35,
                eval_interval=25,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(learning_rate=0.12, trace_decay=0.65, seed=42),
                mode="separate",
                levels=[1],
                episodes_per_level=5000,
            )

        self.assertEqual(len(recovery_events), 4)
        self.assertEqual([int(e["ep"]) for e in recovery_events], [1250, 1750, 2250, 2750])
        for ev in recovery_events:
            self.assertAlmostEqual(ev["restored_marker"], 900.0, places=3)
            self.assertEqual(ev["traces_len"], 0)
            self.assertAlmostEqual(ev["post_alpha"], ev["pre_alpha"], places=6)
            self.assertGreaterEqual(ev["post_eps"], ev["pre_eps"])
            self.assertLessEqual(ev["post_eps"], min(0.12, 2.0 * ev["pre_eps"]) + 1e-9)
            self.assertAlmostEqual(ev["post_eps"], min(0.12, 2.0 * ev["pre_eps"]), places=6)
        self.assertEqual(board["evaluation_summary"]["level_1"]["main_phase_collapse_recoveries"], 4)

    def test_order_agnostic_curriculum_l6_before_l5(self) -> None:
        order = [1, 2, 3, 4, 6, 5]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                env_factory=create_env,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                max_steps_cap=20,
                replay_ratio=0.0,
                eval_interval=2,
                eval_episodes_periodic=1,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=order,
                episodes_per_level=2,
            )

        self.assertEqual(board["metadata"]["levels"], order)
        self.assertEqual(board["metadata"]["level_order"], order)
        ev = board["evaluation_summary"]
        for probe in ("level_3_zero_shot", "level_6_zero_shot", "level_5_zero_shot"):
            self.assertIn(probe, ev)
        expected_keys = [f"level_{u}" for u in order]
        self.assertEqual(list(ev["post_curriculum_retention"].keys()), expected_keys)
        self.assertEqual(list(ev["curriculum_best_retention"].keys()), expected_keys)

    def test_order_agnostic_curriculum_sequential_1_to_6(self) -> None:
        order = [1, 2, 3, 4, 5, 6]
        seen_entry_eps: Dict[int, float] = {}

        class _RewarmSpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                lvl = getattr(env, "level_id", 1)
                if training and decay_eps and lvl not in seen_entry_eps:
                    seen_entry_eps[lvl] = float(agent.epsilon)
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _RewarmSpyTrainer(
                env_factory=create_env,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                max_steps_cap=20,
                replay_ratio=0.0,
                rewarm_epsilon_l5=0.40,
                rewarm_epsilon_l6=0.47,
                eval_interval=2,
                eval_episodes_periodic=1,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=order,
                episodes_per_level=2,
            )

        self.assertEqual(board["metadata"]["levels"], order)
        self.assertAlmostEqual(board["metadata"]["rewarm_epsilon_l6"], 0.47, places=4)
        ev = board["evaluation_summary"]
        self.assertIn("level_5_zero_shot", ev)
        self.assertIn("level_6_zero_shot", ev)
        self.assertGreaterEqual(seen_entry_eps[6], 0.47)

    def test_separate_mode_level_order_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cp_a = os.path.join(tmp, "cp_a")
            out_a = os.path.join(tmp, "out_a")
            cp_b = os.path.join(tmp, "cp_b")
            out_b = os.path.join(tmp, "out_b")

            trainer_a = Trainer(
                env_factory=create_env,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                max_steps_cap=30,
                eval_interval=3,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=2,
                checkpoints_dir=cp_a,
                outputs_dir=out_a,
                seed=42,
            )
            board_a = trainer_a.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[1, 6],
                episodes_per_level=6,
            )

            trainer_b = Trainer(
                env_factory=create_env,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                max_steps_cap=30,
                eval_interval=3,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=2,
                checkpoints_dir=cp_b,
                outputs_dir=out_b,
                seed=42,
            )
            board_b = trainer_b.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[6, 1],
                episodes_per_level=6,
            )

            self.assertEqual(board_a["evaluation_summary"]["level_6"], board_b["evaluation_summary"]["level_6"])

            l6_hist_a = [
                {k: v for k, v in rec.items() if k != "global_episode"}
                for rec in board_a["episode_history"]
                if rec["level_id"] == 6
            ]
            l6_hist_b = [
                {k: v for k, v in rec.items() if k != "global_episode"}
                for rec in board_b["episode_history"]
                if rec["level_id"] == 6
            ]
            self.assertEqual(l6_hist_a, l6_hist_b)

            for fname in ("q_table_lvl6_best.pkl", "q_table_lvl6_final.pkl"):
                with open(os.path.join(cp_a, "tabular", "separate_balanced", fname), "rb") as f:
                    raw_a = pickle.load(f)
                with open(os.path.join(cp_b, "tabular", "separate_balanced", fname), "rb") as f:
                    raw_b = pickle.load(f)
                qt_a = raw_a.get("q_table", raw_a)
                qt_b = raw_b.get("q_table", raw_b)
                self.assertEqual(set(qt_a.keys()), set(qt_b.keys()))
                for k in qt_a:
                    np.testing.assert_array_equal(qt_a[k], qt_b[k])

    def test_long_stage_eval_interval_scaling(self) -> None:
        eval_eps_recorded: List[int] = []

        class _CadenceSpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps:
                    if not hasattr(self, "_ep"):
                        self._ep = 0
                    self._ep += 1
                    self._cur_ep = self._ep
                return {
                    "total_reward": 10.0,
                    "steps": 1,
                    "max_x_reached": 60.0,
                    "progress_pct": 1.0,
                    "obstacle_hits": 0,
                    "items_collected": 0,
                    "items_total": 0,
                    "won": False,
                    "outcome": "timeout",
                }

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if (not random_baseline) and episodes == self.eval_episodes_periodic and hasattr(self, "_cur_ep"):
                    eval_eps_recorded.append(self._cur_ep)
                return {
                    "win_rate": 0.0,
                    "avg_reward": 10.0,
                    "avg_completion_steps": -1.0,
                    "avg_items_collected": 0.0,
                    "avg_max_x": 60.0,
                    "outcome_counts": {"timeout": max(0, episodes)},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _CadenceSpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                eval_interval=25,
                eval_episodes_periodic=2,
                final_eval_episodes=5,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            for ep in (1, 100, 15000, 25500):
                self.assertEqual(trainer._get_stage_eval_interval(ep, 30000), 100)
            for ep in (25501, 25525, 30000):
                self.assertEqual(trainer._get_stage_eval_interval(ep, 30000), 25)
            self.assertEqual(trainer._get_stage_eval_interval(5000, 7000), 25)

            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[5],
                episodes_per_level=30000,
            )

        main_evals = [e for e in eval_eps_recorded if e <= 25500]
        polish_evals = [e for e in eval_eps_recorded if e > 25500]
        self.assertTrue(main_evals)
        self.assertTrue(all(e % 100 == 0 for e in main_evals))
        self.assertNotIn(25, main_evals)
        self.assertIn(25500, main_evals)
        self.assertIn(25525, polish_evals)
        self.assertIn(30000, polish_evals)
        self.assertEqual(len(polish_evals), (30000 - 25500) // 25)
        self.assertEqual(
            board["metadata"]["eval_interval_per_level"]["level_5"],
            {"main": 100, "polish": 25},
        )


if __name__ == "__main__":
    unittest.main()
