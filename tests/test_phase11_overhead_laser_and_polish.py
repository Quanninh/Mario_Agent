import os
import tempfile
import unittest
from typing import Dict, List, Tuple

from agent import TabularQAgent
from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from discretizer import MarioStateDiscretizer
from rewards import BalancedRewardStrategy, DenseRewardStrategy, SparseRewardStrategy
from training import Trainer
from training.trainer import DEFAULT_LEVEL_EPISODES


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


class TestPhase11OverheadLaserAndPolish(unittest.TestCase):
    def test_airborne_ghost_alt_uses_standing_reference(self) -> None:
        disc = MarioStateDiscretizer()

        # Airborne with overhead LaserGhost -> stays HIGH (1)
        s_air_overhead = disc.discretize(
            make_state(is_grounded=False, player_y=240.0, ghost_y=232.0, dist_to_ghost=20.0)
        )
        self.assertEqual(s_air_overhead[6], 1)

        # Grounded on top of tombstone at y=232 -> ghost at y=232 is eye-level LOW (0)
        s_tomb_eye = disc.discretize(
            make_state(is_grounded=True, player_y=232.0, ghost_y=232.0, dist_to_ghost=20.0)
        )
        self.assertEqual(s_tomb_eye[6], 0)

        # Low patrolling ghost (ghost_y=280.0) -> LOW (0) whether airborne or grounded
        s_low_air = disc.discretize(
            make_state(is_grounded=False, player_y=240.0, ghost_y=280.0, dist_to_ghost=20.0)
        )
        s_low_ground = disc.discretize(
            make_state(is_grounded=True, player_y=288.0, ghost_y=280.0, dist_to_ghost=20.0)
        )
        self.assertEqual(s_low_air[6], 0)
        self.assertEqual(s_low_ground[6], 0)

        # State space sizes remain 55,296 / 110,592
        self.assertEqual(disc.state_space_size, 55296)
        self.assertEqual(MarioStateDiscretizer(enable_wisp_sensor=True).state_space_size, 110592)

    def test_balanced_reward_laser_jump_penalty_default_and_override(self) -> None:
        ev = GameEvents(dx=4.0)
        s_laser = make_state(is_grounded=True, dist_to_laser=40.0)

        bal_default = BalancedRewardStrategy()
        self.assertEqual(bal_default.laser_threat_distance, -1.0)
        r_walk = bal_default.calculate_reward(s_laser, Action.RIGHT, s_laser, ev, False)
        r_jump_def = bal_default.calculate_reward(s_laser, Action.RIGHT_JUMP, s_laser, ev, False)
        self.assertAlmostEqual(r_jump_def - r_walk, -15.0)

        bal_override = BalancedRewardStrategy(laser_threat_distance=85.0)
        r_jump_ovr = bal_override.calculate_reward(s_laser, Action.RIGHT_JUMP, s_laser, ev, False)
        self.assertAlmostEqual(r_jump_ovr, r_walk)

        # Dense and Sparse baselines untouched
        s1 = make_state(player_x=100.0, dist_to_goal=400.0)
        s2 = make_state(player_x=104.0, dist_to_goal=396.0)
        self.assertAlmostEqual(DenseRewardStrategy().calculate_reward(s1, Action.RIGHT, s2, ev, False), 4.8)
        self.assertEqual(SparseRewardStrategy().calculate_reward(s1, Action.RIGHT, s2, ev, False), 0.0)

    def test_polish_anchor_reloads_stage_best_at_85pct(self) -> None:
        s_tag = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        pre_ep86_snapshot: Dict[str, float] = {}

        class _OnsetSpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps:
                    if not hasattr(self, "_ep"):
                        self._ep = 0
                    self._ep += 1
                    ep = self._ep
                    if ep == 86:
                        # On ep 86 (onset of 85%-100% polish tail), anchor reload must have already fired
                        pre_ep86_snapshot["marker"] = float(agent.q_table[s_tag][Action.STAY])
                        pre_ep86_snapshot["traces_len"] = len(agent._traces)
                        pre_ep86_snapshot["alpha"] = float(agent.alpha)
                        pre_ep86_snapshot["epsilon"] = float(agent.epsilon)
                        pre_ep86_snapshot["trace_decay"] = float(agent.trace_decay)
                    res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                    # Ep 50 captures high-performing best; Ep 60..75 degrade Q-table + plant trace
                    if ep == 50:
                        agent.q_table[s_tag][Action.STAY] = 777.0
                    elif ep == 70:
                        agent.q_table[s_tag][Action.STAY] = -555.0
                        agent._traces[(s_tag, Action.STAY)] = 0.5
                    return res
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                val = float(agent.q_table.get(s_tag, [0.0] * 6)[Action.STAY])
                good = val > 500.0
                return {
                    "win_rate": 100.0 if good else 0.0,
                    "avg_reward": 900.0 if good else 100.0,
                    "avg_completion_steps": 100.0 if good else -1.0,
                    "avg_items_collected": 0.0,
                    "outcome_counts": {"win": episodes} if good else {"pit_death": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _OnsetSpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                trace_decay=0.65,
                trace_decay_min=0.35,
                lambda_polish_zero=True,
                polish_anchor_best=True,
                alpha_min=0.003,
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
                episodes_per_level=100,
            )

        self.assertAlmostEqual(pre_ep86_snapshot["marker"], 777.0, places=3)
        self.assertEqual(pre_ep86_snapshot["traces_len"], 0)
        self.assertLess(pre_ep86_snapshot["alpha"], 0.05)
        self.assertLess(pre_ep86_snapshot["trace_decay"], 0.35)
        self.assertTrue(board["evaluation_summary"]["level_1"]["polish_anchor_reloaded"])

    def test_polish_anti_regression_guard_rolls_back_and_cools_alpha(self) -> None:
        s_tag = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ep91_snapshot: Dict[str, float] = {}
        pre_rollback_alpha: List[float] = []

        class _MidPolishSpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps:
                    if not hasattr(self, "_ep"):
                        self._ep = 0
                    self._ep += 1
                    ep = self._ep
                    if ep == 91:
                        ep91_snapshot["marker"] = float(agent.q_table[s_tag][Action.STAY])
                        ep91_snapshot["alpha"] = float(agent.alpha)
                    res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                    if ep == 50:
                        agent.q_table[s_tag][Action.STAY] = 888.0
                    elif ep == 90:
                        # Simulate sudden reward drop > 25% right before periodic eval at ep 90
                        pre_rollback_alpha.append(float(agent.alpha))
                        agent.q_table[s_tag][Action.STAY] = -888.0
                    return res
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                val = float(agent.q_table.get(s_tag, [0.0] * 6)[Action.STAY])
                good = val > 500.0
                return {
                    "win_rate": 0.0,
                    "avg_reward": 1705.0 if good else 275.0,
                    "avg_completion_steps": -1.0,
                    "avg_items_collected": 0.0,
                    "outcome_counts": {"ghost_death": episodes} if good else {"timeout": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _MidPolishSpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                trace_decay=0.65,
                trace_decay_min=0.35,
                lambda_polish_zero=True,
                polish_anchor_best=True,
                alpha_min=0.003,
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(learning_rate=0.12, trace_decay=0.65, seed=42),
                mode="curriculum",
                levels=[1],
                episodes_per_level=100,
            )

        self.assertAlmostEqual(ep91_snapshot["marker"], 888.0, places=3)
        self.assertLessEqual(ep91_snapshot["alpha"], pre_rollback_alpha[0] * 0.5 + 1e-6)
        self.assertGreaterEqual(ep91_snapshot["alpha"], max(0.001, 0.003 * 0.5) - 1e-9)
        self.assertGreaterEqual(board["evaluation_summary"]["level_1"]["polish_regression_rollbacks"], 1)

    def test_default_level5_schedule_and_metadata(self) -> None:
        self.assertEqual(
            DEFAULT_LEVEL_EPISODES,
            {1: 800, 2: 1200, 3: 1800, 4: 7000, 5: 30000, 6: 3500},
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                eval_interval=5,
                eval_episodes_periodic=1,
                final_eval_episodes=1,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[1],
                episodes_per_level=10,
            )
        self.assertTrue(board["metadata"]["polish_anchor_best"])


if __name__ == "__main__":
    unittest.main()
