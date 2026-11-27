import os
import tempfile
import unittest
from typing import Dict, List, Tuple

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


class TestLaserMotionAndEnvFov(unittest.TestCase):
    def test_level5_ground_ghost_not_occluded_by_sky_maiden(self) -> None:
        env = create_env(5)
        try:
            env.reset()
            for px in (270.0, 285.0, 300.0):
                env.player.x = px
                env.player.y = 288.0
                raw = env._get_raw_state()
                self.assertGreaterEqual(raw.dist_to_ghost, 0.0)
                self.assertGreater(
                    raw.ghost_y,
                    200.0,
                    f"Expected ground ghost (y>200) at player_x={px}, got ghost_y={raw.ghost_y}",
                )
        finally:
            env.close()

    def test_pit_margin_24_and_motion_aware_laser_bin(self) -> None:
        disc = MarioStateDiscretizer()
        self.assertEqual(disc.pit_immediate, 24.0)
        self.assertEqual(disc.laser_imminent_distance, 28.0)
        self.assertEqual(disc.state_space_size, 55296)
        self.assertEqual(MarioStateDiscretizer(enable_wisp_sensor=True).state_space_size, 110592)

        # Pit margin calibration
        self.assertEqual(disc.discretize(make_state(dist_to_pit=20.0, pit_fire_state=0))[4], 2)
        self.assertEqual(disc.discretize(make_state(dist_to_pit=30.0, pit_fire_state=0))[4], 1)
        self.assertEqual(disc.discretize(make_state(dist_to_pit=50.0, pit_fire_state=1))[4], 3)

        # World-frame motion-aware laser bin
        # Tick 1: first contact -> CLOSING / IMMINENT (1)
        s1 = disc.discretize(make_state(player_x=1140.0, dist_to_laser=65.0))
        self.assertEqual(s1[7], 1)
        self.assertAlmostEqual(disc._prev_laser_world_x, 1205.0)

        # Tick 2: beam sweeps rightward away while Hoa walks right (world_lx 1205 -> 1206.5, +1.5 > 0.1) -> 2
        s2 = disc.discretize(make_state(player_x=1144.0, dist_to_laser=62.5))
        self.assertEqual(s2[7], 2)
        self.assertAlmostEqual(disc._prev_laser_world_x, 1206.5)

        # Tick 3: beam sweeps leftward toward Hoa (world_lx 1206.5 -> 1205.0, -1.5 <= 0.1) -> 1
        s3 = disc.discretize(make_state(player_x=1148.0, dist_to_laser=57.0))
        self.assertEqual(s3[7], 1)
        self.assertAlmostEqual(disc._prev_laser_world_x, 1205.0)

        # Tick 4: point-blank <= 20.0 px even if world_lx increased -> 1
        s4 = disc.discretize(make_state(player_x=1152.0, dist_to_laser=18.0))
        self.assertEqual(s4[7], 1)

        # reset() clears _prev_laser_world_x so next tick has no stale velocity
        disc.reset()
        self.assertIsNone(disc._prev_laser_world_x)
        s_after_reset = disc.discretize(make_state(player_x=1144.0, dist_to_laser=62.5))
        self.assertEqual(s_after_reset[7], 1)


class TestTraceFloorAlphaCoolingAndRetentionGuard(unittest.TestCase):
    def test_non_zero_trace_floor_alpha_cooling_and_zero_fast_path(self) -> None:
        discretizer = MarioStateDiscretizer()
        reward = BalancedRewardStrategy()
        l1_obs: List[Tuple[int, float, float]] = []

        class _SpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps and getattr(env, "level_id", 1) == 1:
                    l1_obs.append((len(l1_obs) + 1, float(agent.trace_decay), float(agent.alpha)))
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _SpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=discretizer,
                reward_strategy=reward,
                trace_decay=0.65,
                trace_decay_min=0.35,
                lambda_polish_zero=False,
                alpha_min=0.005,
                eval_interval=25,
                eval_episodes_periodic=2,
                final_eval_episodes=3,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(learning_rate=0.12, trace_decay=0.65, seed=42),
                mode="curriculum",
                levels=[1, 2],
                episodes_per_level=100,
            )

            self.assertEqual(len(l1_obs), 100)
            self.assertAlmostEqual(l1_obs[0][1], 0.65, places=6)
            expected_ep75 = 0.65 - (0.65 - 0.35) * (74.0 / 75.0)
            self.assertAlmostEqual(l1_obs[74][1], expected_ep75, places=6)
            for _, lam, _ in l1_obs[75:]:
                self.assertAlmostEqual(lam, 0.35, places=6)
            self.assertAlmostEqual(l1_obs[0][2], 0.12 - (0.12 - 0.005) / 100.0, places=6)
            self.assertAlmostEqual(l1_obs[-1][2], 0.005, places=6)

            meta = board["metadata"]
            self.assertAlmostEqual(meta["trace_decay_start"], 0.65)
            self.assertAlmostEqual(meta["trace_decay_min"], 0.35)
            self.assertEqual(meta["trace_decay_schedule"], "linear_to_0.35_at_75pct")
            self.assertAlmostEqual(meta["alpha_min"], 0.005)

            cp_best = os.path.join(tmp, "cp", "tabular", "curriculum_balanced", "q_table_curriculum_best.pkl")
            self.assertTrue(os.path.exists(cp_best))
            ev_sum = board["evaluation_summary"]
            self.assertIn("curriculum_best_checkpoint_stage", ev_sum)
            self.assertIn("curriculum_best_retention", ev_sum)
            for lvl in (1, 2):
                self.assertIn(f"level_{lvl}", ev_sum["curriculum_best_retention"])

        # Zero lambda stays exact 0.0 across all episodes
        l1_zero: List[float] = []

        class _ZeroSpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps and getattr(env, "level_id", 1) == 1:
                    l1_zero.append(float(agent.trace_decay))
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

        with tempfile.TemporaryDirectory() as tmp:
            t_zero = _ZeroSpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=discretizer,
                reward_strategy=reward,
                trace_decay=0.0,
                trace_decay_min=0.35,
                alpha_min=0.005,
                eval_interval=50,
                eval_episodes_periodic=1,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            b_zero = t_zero.run_experiment(
                agent_factory=lambda: TabularQAgent(trace_decay=0.0, seed=42),
                mode="separate",
                levels=[1],
                episodes_per_level=20,
            )
            self.assertTrue(all(lam == 0.0 for lam in l1_zero))
            self.assertAlmostEqual(b_zero["metadata"]["trace_decay_min"], 0.0)
            self.assertEqual(b_zero["metadata"]["trace_decay_schedule"], "constant_0.0")

    def test_curriculum_retention_guard_locks_peak_stage_against_later_collapse(self) -> None:
        s_marker = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        class _StageCollapseTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                lvl = getattr(env, "level_id", 1)
                res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                if training and decay_eps:
                    agent.q_table[s_marker][Action.RIGHT] = 1.0 if lvl <= 2 else -1.0
                return res

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                ok = float(agent.q_table.get(s_marker, [0.0] * 6)[Action.RIGHT]) > 0.0 and level_id <= 2
                return {
                    "win_rate": 100.0 if ok else 0.0,
                    "avg_reward": 500.0 if ok else -50.0,
                    "avg_completion_steps": 10.0 if ok else -1.0,
                    "avg_items_collected": 0.0,
                    "outcome_counts": {"win": episodes} if ok else {"pit_death": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _StageCollapseTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                replay_ratio=0.0,
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=3,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[1, 2, 3],
                episodes_per_level=5,
            )
            ev_sum = board["evaluation_summary"]
            self.assertEqual(ev_sum["curriculum_best_checkpoint_stage"], 2)
            self.assertEqual(ev_sum["curriculum_best_retention"]["level_1"]["win_rate"], 100.0)
            self.assertEqual(ev_sum["curriculum_best_retention"]["level_2"]["win_rate"], 100.0)
            self.assertEqual(ev_sum["curriculum_best_retention"]["level_3"]["win_rate"], 0.0)

    def test_trainer_resets_discretizer_between_episodes(self) -> None:
        class _TwoTickLaserEnv(BaseEnvironment):
            def __init__(self):
                self.level_id = 4
                self.max_steps = 5
                self.start_x = 1140.0
                self.items = []
                self.tick = 0

                class _Goal:
                    x = 2000.0

                self.goal = _Goal()

            def reset(self) -> RawState:
                self.tick = 0
                return make_state(player_x=1140.0, dist_to_laser=65.0)

            def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
                self.tick += 1
                if self.tick == 1:
                    return make_state(player_x=1144.0, dist_to_laser=62.5), GameEvents(dx=4.0), True
                return make_state(player_x=1148.0, dist_to_laser=60.0), GameEvents(dx=4.0), True

            def render(self) -> None:
                pass

            def close(self) -> None:
                pass

        disc = MarioStateDiscretizer()
        trainer = Trainer(
            env_factory=lambda lvl: _TwoTickLaserEnv(),
            discretizer=disc,
            reward_strategy=BalancedRewardStrategy(),
        )
        # Seed _prev_laser_world_x with stale lower value that WOULD falsely trigger bin=2 if not reset()
        disc._prev_laser_world_x = 1190.0
        seen_first_bins: List[int] = []
        orig_discretize = disc.discretize

        def spy_discretize(raw):
            res = orig_discretize(raw)
            if raw.player_x == 1140.0:
                seen_first_bins.append(res[7])
            return res

        disc.discretize = spy_discretize  # type: ignore[method-assign]
        trainer._run_episode(_TwoTickLaserEnv(), TabularQAgent(seed=1), training=False)
        self.assertEqual(seen_first_bins[0], 1)


if __name__ == "__main__":
    unittest.main()
