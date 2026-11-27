import os
import random
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple
import numpy as np

from agent import HybridQSARSAAgent, TabularQAgent
from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from discretizer import MarioStateDiscretizer
from rewards import BalancedRewardStrategy, DenseRewardStrategy, SparseRewardStrategy
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

        class _Goal:
            x = 1000.0

        self.goal = _Goal()

    def reset(self) -> RawState:
        return make_state(player_x=self.start_x)

    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
        return make_state(player_x=self.start_x + 4.0), GameEvents(dx=4.0, reached_goal=True), True

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestPitMarginAndWispSuppression(unittest.TestCase):
    def test_discretizer_pit_margins_and_wisp_suppression(self) -> None:
        disc_default = MarioStateDiscretizer()
        self.assertFalse(disc_default.enable_wisp_sensor)
        self.assertEqual(disc_default.pit_immediate, 24.0)
        self.assertEqual(disc_default.state_space_size, 55296)

        # Pit margin calibration (24.0 px horizon)
        self.assertEqual(disc_default.discretize(make_state(dist_to_pit=20.0, pit_fire_state=0))[4], 2)
        self.assertEqual(disc_default.discretize(make_state(dist_to_pit=30.0, pit_fire_state=0))[4], 1)
        self.assertEqual(disc_default.discretize(make_state(dist_to_pit=50.0, pit_fire_state=1))[4], 3)

        # Wisp suppressed by default while keeping 12-tuple contract
        s_suppressed = disc_default.discretize(make_state(dist_to_wisp=10.0, wisp_dir=1))
        self.assertEqual(len(s_suppressed), 12)
        self.assertEqual(s_suppressed[8], 0)

        # Enabled wisp sensor restores detection and 110,592 state cardinality
        disc_wisp = MarioStateDiscretizer(enable_wisp_sensor=True)
        self.assertTrue(disc_wisp.enable_wisp_sensor)
        self.assertEqual(disc_wisp.state_space_size, 110592)
        self.assertEqual(disc_wisp.discretize(make_state(dist_to_wisp=10.0, wisp_dir=1))[8], 1)

    def test_balanced_reward_wisp_exemption_and_baselines(self) -> None:
        ev = GameEvents(dx=4.0)
        s_wisp = make_state(is_grounded=True, dist_to_wisp=20.0, wisp_dir=1)

        # Default wisp_threat_distance=-1.0 does NOT exempt jump near overhead wisp
        bal_default = BalancedRewardStrategy()
        r_walk = bal_default.calculate_reward(s_wisp, Action.RIGHT, s_wisp, ev, False)
        r_jump_def = bal_default.calculate_reward(s_wisp, Action.RIGHT_JUMP, s_wisp, ev, False)
        self.assertAlmostEqual(r_jump_def - r_walk, -15.0)

        # Explicit wisp_threat_distance=65.0 exempts jump near wisp
        bal_wisp = BalancedRewardStrategy(wisp_threat_distance=65.0)
        r_jump_wisp = bal_wisp.calculate_reward(s_wisp, Action.RIGHT_JUMP, s_wisp, ev, False)
        self.assertAlmostEqual(r_jump_wisp, r_walk)

        # early_double_jump_penalty and Dense/Sparse strategies remain intact
        s_fast_air = make_state(is_grounded=False, can_double_jump=True, is_crouching=False, velocity_y=-10.0)
        self.assertAlmostEqual(
            bal_default.calculate_reward(s_fast_air, Action.RIGHT_JUMP, s_fast_air, ev, False)
            - bal_default.calculate_reward(s_fast_air, Action.RIGHT, s_fast_air, ev, False),
            -3.0,
        )
        s1 = make_state(player_x=100.0, dist_to_goal=400.0)
        s2 = make_state(player_x=104.0, dist_to_goal=396.0)
        self.assertAlmostEqual(DenseRewardStrategy().calculate_reward(s1, Action.RIGHT, s2, ev, False), 4.8)
        self.assertEqual(SparseRewardStrategy().calculate_reward(s1, Action.RIGHT, s2, ev, False), 0.0)


class TestTraceDecayAndCarryover(unittest.TestCase):
    def test_linear_trace_decay_schedule_and_curriculum_carryover(self) -> None:
        discretizer = MarioStateDiscretizer()
        reward = BalancedRewardStrategy()
        observed_traces_l1: List[Tuple[int, float, int]] = []
        observed_stage_starts: Dict[int, Tuple[float, float]] = {}

        class _InstrumentedAgent(TabularQAgent):
            pass

        class _InstrumentedTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                lvl_id = getattr(env, "level_id", 1)
                if training and decay_eps and lvl_id == 1:
                    ep_idx = len(observed_traces_l1) + 1
                    observed_traces_l1.append((ep_idx, float(agent.trace_decay), len(agent._traces)))
                if training and decay_eps and lvl_id not in observed_stage_starts:
                    observed_stage_starts[lvl_id] = (float(agent.trace_decay), float(agent.epsilon))
                res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                if training and decay_eps and lvl_id == 1 and len(observed_traces_l1) > 75:
                    # Assert _traces empty after episode >= 76
                    observed_traces_l1[-1] = (
                        observed_traces_l1[-1][0],
                        float(agent.trace_decay),
                        len(agent._traces),
                    )
                return res

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _InstrumentedTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=discretizer,
                reward_strategy=reward,
                trace_decay=0.65,
                trace_decay_min=0.0,
                lambda_polish_zero=False,
                eval_interval=25,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: _InstrumentedAgent(trace_decay=0.65, seed=42),
                mode="curriculum",
                levels=[1, 2, 3, 4],
                episodes_per_level=100,
                auto_epsilon_decay=True,
            )

        # Verify trace_decay decays linearly across 1..75 and hits exact 0.0 with empty _traces for 76..100
        self.assertEqual(len(observed_traces_l1), 100)
        self.assertAlmostEqual(observed_traces_l1[0][1], 0.65, places=6)
        self.assertGreater(observed_traces_l1[37][1], 0.30)
        self.assertAlmostEqual(observed_traces_l1[74][1], 0.65 * (1.0 - 74.0 / 75.0), places=6)
        for ep_idx, lam, tr_len in observed_traces_l1[75:]:
            self.assertEqual(lam, 0.0, f"Expected 0.0 at ep {ep_idx}")
            self.assertEqual(tr_len, 0, f"Expected empty traces at ep {ep_idx}")

        # Verify trace_decay resets cleanly to 0.65 at start of every stage and 3-tier rewarm fires
        self.assertAlmostEqual(observed_stage_starts[2][0], 0.65, places=6)
        self.assertAlmostEqual(observed_stage_starts[3][0], 0.65, places=6)
        self.assertAlmostEqual(observed_stage_starts[4][0], 0.65, places=6)
        self.assertGreaterEqual(observed_stage_starts[2][1], 0.30)
        self.assertGreaterEqual(observed_stage_starts[3][1], 0.35)
        self.assertGreaterEqual(observed_stage_starts[4][1], 0.45)

        # Metadata checks
        meta = board["metadata"]
        self.assertEqual(meta["state_space_max"], 55296)
        self.assertFalse(meta["enable_wisp_sensor"])
        self.assertAlmostEqual(meta["trace_decay_start"], 0.65)
        self.assertAlmostEqual(meta["trace_decay_min"], 0.0)
        self.assertEqual(meta["trace_decay_schedule"], "linear_to_0.00_at_75pct")
        self.assertAlmostEqual(meta["rewarm_epsilon_hard"], 0.45)

        # Verify best_checkpoint_* keys present in evaluation_summary
        for lvl in range(1, 5):
            lvl_entry = board["evaluation_summary"][f"level_{lvl}"]
            for key in (
                "best_checkpoint_win_rate",
                "best_checkpoint_avg_reward",
                "best_checkpoint_completion_steps",
                "best_checkpoint_items_collected",
                "best_checkpoint_outcome_counts",
            ):
                self.assertIn(key, lvl_entry)

    def test_carryover_uses_stage_best_not_drifted_final_and_separate_has_best_keys(self) -> None:
        discretizer = MarioStateDiscretizer()
        reward = BalancedRewardStrategy()
        s_marker = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        stage2_start_marker_val: List[float] = []

        class _DriftingTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                lvl_id = getattr(env, "level_id", 1)
                if training and decay_eps and lvl_id == 2 and not stage2_start_marker_val:
                    stage2_start_marker_val.append(float(agent.q_table[s_marker][Action.STAY]))
                res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                if training and decay_eps and lvl_id == 1:
                    # Drift Action.STAY heavily negative at late episodes so best != drifted final
                    if not hasattr(self, "_l1_ep"):
                        self._l1_ep = 0
                    self._l1_ep += 1
                    if self._l1_ep == 5:
                        agent.q_table[s_marker][Action.STAY] = 999.0
                    elif self._l1_ep == 10:
                        agent.q_table[s_marker][Action.STAY] = -999.0
                return res

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                # Force periodic score at ep 5 (where STAY=999.0) higher than ep 10 (where STAY=-999.0)
                val = float(agent.q_table[s_marker][Action.STAY])
                return {
                    "win_rate": 100.0 if val > 0 else 50.0,
                    "avg_reward": 500.0 if val > 0 else 100.0,
                    "avg_completion_steps": 10.0,
                    "avg_items_collected": 0.0,
                    "outcome_counts": {"win": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _DriftingTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=discretizer,
                reward_strategy=reward,
                polish_anchor_best=False,
                replay_ratio=0.0,
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board_curr = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[1, 2],
                episodes_per_level=10,
            )
            self.assertAlmostEqual(stage2_start_marker_val[0], 999.0, places=3)
            self.assertEqual(board_curr["evaluation_summary"]["level_1"]["trained_win_rate"], 50.0)
            self.assertEqual(board_curr["evaluation_summary"]["level_1"]["best_checkpoint_win_rate"], 100.0)

            board_sep = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[1],
                episodes_per_level=10,
            )
            for key in (
                "best_checkpoint_win_rate",
                "best_checkpoint_avg_reward",
                "best_checkpoint_completion_steps",
                "best_checkpoint_items_collected",
                "best_checkpoint_outcome_counts",
            ):
                self.assertIn(key, board_sep["evaluation_summary"]["level_1"])


if __name__ == "__main__":
    unittest.main()
