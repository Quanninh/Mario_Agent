import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent import TabularQAgent
from core.base_env import BaseEnvironment, GameEvents, RawState
from discretizer import MarioStateDiscretizer
from env import create_env
from env.level_generator import LevelGenerator
from rewards import BalancedRewardStrategy
from train import parse_args as parse_tabular_args
from training.trainer import Trainer, resolve_map_seed


def _make_raw_state(player_x: float = 50.0, goal_dist: float = 500.0) -> RawState:
    return RawState(
        player_x=player_x,
        player_y=288.0,
        velocity_x=0.0,
        velocity_y=0.0,
        is_grounded=True,
        can_double_jump=False,
        is_crouching=False,
        dist_to_obstacle=-1.0,
        obstacle_height=0.0,
        dist_to_pit=-1.0,
        pit_fire_state=0,
        dist_to_ghost=-1.0,
        ghost_y=0.0,
        dist_to_item=-1.0,
        dist_to_laser=-1.0,
        dist_to_wisp=-1.0,
        wisp_dir=0,
        dist_to_monster=-1.0,
        monster_y=0.0,
        dist_to_goal=goal_dist,
    )


class _SeedSpyEnv(BaseEnvironment):
    def __init__(self, level_id: int, seed=None, win: bool = True):
        self.level_id = level_id
        self.seed = seed
        self.win = win
        self.start_x = 50.0
        self.items = []
        self.obstacles = []
        self.pits = []
        self.ghosts = []
        self.monsters = []
        self.reset_start_xs = []

        class _Goal:
            def __init__(self, x: float):
                self.x = x

        self.goal = _Goal(1000.0 if seed is None else 1000.0 + float(seed) * 100.0)

    def reset(self) -> RawState:
        self.reset_start_xs.append(self.start_x)
        return _make_raw_state(player_x=self.start_x, goal_dist=self.goal.x - self.start_x)

    def step(self, action: int):
        return (
            _make_raw_state(player_x=self.start_x + 4.0, goal_dist=0.0 if self.win else 500.0),
            GameEvents(dx=4.0, reached_goal=self.win, fell_in_pit=not self.win),
            True,
        )

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestMapSeedRotation(unittest.TestCase):
    def test_resolve_map_seed_rotation_and_fixed_fallback(self) -> None:
        self.assertIsNone(resolve_map_seed(1, 1, map_seed=None, map_seed_interval=0))
        self.assertEqual(resolve_map_seed(1, 1, map_seed=100, map_seed_interval=0), 100)
        self.assertEqual(resolve_map_seed(6, 5000, map_seed=100, map_seed_interval=0), 100)
        self.assertEqual(resolve_map_seed(7, 5000, map_seed=100, map_seed_interval=0), 100)

        # Block 0 respects explicit initial map_seed across Levels 1-7
        for lvl in (4, 6, 7):
            self.assertEqual(resolve_map_seed(lvl, 1, map_seed=100, map_seed_interval=3000), 100)
            self.assertEqual(resolve_map_seed(lvl, 3000, map_seed=100, map_seed_interval=3000), 100)

        # Subsequent intervals draw reproducible non-sequential random seeds per level
        s1_l4_a = resolve_map_seed(4, 3001, map_seed=100, map_seed_interval=3000)
        s1_l4_b = resolve_map_seed(4, 6000, map_seed=100, map_seed_interval=3000)
        s2_l4 = resolve_map_seed(4, 6001, map_seed=100, map_seed_interval=3000)
        s1_l6 = resolve_map_seed(6, 3001, map_seed=100, map_seed_interval=3000)
        s1_l7 = resolve_map_seed(7, 3001, map_seed=100, map_seed_interval=3000)

        self.assertIsNotNone(s1_l4_a)
        self.assertEqual(s1_l4_a, s1_l4_b)
        self.assertNotEqual(s1_l4_a, 100)
        self.assertNotEqual(s1_l4_a, 101)
        self.assertNotEqual(s2_l4, s1_l4_a)
        self.assertNotEqual(s2_l4, s1_l4_a + 1)
        self.assertNotEqual(s1_l4_a, s1_l6)
        self.assertNotEqual(s1_l6, s1_l7)

        # When map_seed is omitted with rotation active, random seeds stay reproducible from fallback_seed
        r0 = resolve_map_seed(3, 1, map_seed=None, map_seed_interval=5000, fallback_seed=77)
        r0_end = resolve_map_seed(3, 5000, map_seed=None, map_seed_interval=5000, fallback_seed=77)
        r1 = resolve_map_seed(3, 5001, map_seed=None, map_seed_interval=5000, fallback_seed=77)
        self.assertIsNotNone(r0)
        self.assertEqual(r0, r0_end)
        self.assertNotEqual(r1, r0)

        # Level 8 remains on fixed handcrafted layout (None)
        self.assertIsNone(resolve_map_seed(8, 1, map_seed=100, map_seed_interval=3000))
        self.assertIsNone(resolve_map_seed(8, 3001, map_seed=100, map_seed_interval=3000))
        self.assertIsNone(resolve_map_seed(8, 1, map_seed=100, map_seed_interval=0))

    def test_level_generator_changes_layouts_across_random_rotated_seeds_l1_to_l7(self) -> None:
        s_a = 100
        for lvl in (1, 6, 7):
            s_b = resolve_map_seed(lvl, 3001, map_seed=s_a, map_seed_interval=3000)
            cfg_a = LevelGenerator.generate(lvl, seed=s_a)
            cfg_b = LevelGenerator.generate(lvl, seed=s_b)
            obs_a = [(round(o.x, 2), round(o.height, 2), o.style) for o in cfg_a.obstacles]
            obs_b = [(round(o.x, 2), round(o.height, 2), o.style) for o in cfg_b.obstacles]
            self.assertNotEqual(obs_a, obs_b)
            if lvl in (6, 7):
                mon_a = [round(m.x, 2) for m in (cfg_a.monsters or [])]
                mon_b = [round(m.x, 2) for m in (cfg_b.monsters or [])]
                self.assertNotEqual(mon_a, mon_b)

    def test_trainer_rotates_procedural_map_seed_in_curriculum_and_separate(self) -> None:
        s0 = 100
        s1 = resolve_map_seed(1, 3, map_seed=100, map_seed_interval=2)
        s2 = resolve_map_seed(1, 5, map_seed=100, map_seed_interval=2)
        self.assertEqual(len({s0, s1, s2}), 3)

        for mode in ("curriculum", "separate"):
            calls = []

            def spy_factory(level_id: int, seed=None):
                calls.append((level_id, seed))
                return create_env(level_id=level_id, seed=seed)

            with tempfile.TemporaryDirectory() as tmp:
                trainer = Trainer(
                    env_factory=spy_factory,
                    discretizer=MarioStateDiscretizer(),
                    reward_strategy=BalancedRewardStrategy(),
                    eval_interval=3,
                    eval_episodes_periodic=1,
                    final_eval_episodes=2,
                    untrained_eval_episodes=1,
                    checkpoints_dir=f"{tmp}/cp",
                    outputs_dir=f"{tmp}/out",
                    map_seed=100,
                    map_seed_interval=2,
                )
                board = trainer.run_experiment(
                    agent_factory=lambda: TabularQAgent(seed=42),
                    mode=mode,
                    levels=[1],
                    episodes_per_level=6,
                )

            seed_calls = [s for _, s in calls if s is not None]
            eval_calls = [s for _, s in calls if s is None]
            self.assertEqual(seed_calls, [s0, s1, s2])
            self.assertGreaterEqual(len(eval_calls), 3)
            self.assertEqual(board["metadata"]["map_seed"], 100)
            self.assertEqual(board["metadata"]["map_seed_interval"], 2)
            self.assertEqual(
                [rec["map_seed"] for rec in board["episode_history"]],
                [s0, s0, s1, s1, s2, s2],
            )

    def test_trainer_recomputes_reverse_spawns_on_l7_and_keeps_level8_fixed(self) -> None:
        created_envs = []

        def spy_env_factory(level_id: int, seed=None):
            env = _SeedSpyEnv(level_id=level_id, seed=seed, win=False)
            created_envs.append((level_id, seed, env))
            return env

        s0 = resolve_map_seed(7, 1, map_seed=10, map_seed_interval=3)
        s1 = resolve_map_seed(7, 4, map_seed=10, map_seed_interval=3)
        self.assertEqual(s0, 10)
        self.assertNotEqual(s1, 10)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                env_factory=spy_env_factory,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                replay_ratio=0.5,
                eval_interval=60,
                eval_episodes_periodic=1,
                final_eval_episodes=1,
                untrained_eval_episodes=1,
                checkpoints_dir=f"{tmp}/cp",
                outputs_dir=f"{tmp}/out",
                map_seed=10,
                map_seed_interval=3,
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[7, 8],
                episodes_per_level=60,
            )

        l7_s0 = [env for lid, s, env in created_envs if lid == 7 and s == s0][0]
        l7_s1 = [env for lid, s, env in created_envs if lid == 7 and s == s1][0]
        self.assertGreater(l7_s1.reset_start_xs[0], l7_s0.reset_start_xs[0])

        l8_seeds = [s for lid, s, _ in created_envs if lid == 8]
        self.assertTrue(all(s is None for s in l8_seeds))
        l8_recs = [r for r in board["episode_history"] if r["level_id"] == 8]
        self.assertTrue(all(r["map_seed"] is None for r in l8_recs))

    def test_dqn_curriculum_rotates_l6_map_seed_and_evaluates_fixed(self) -> None:
        from train_dqn import train_dqn_curriculum

        s0 = 200
        s1 = resolve_map_seed(6, 4, map_seed=200, map_seed_interval=3)
        calls = []

        def spy_create_env(level_id: int = 1, seed=None):
            calls.append((level_id, seed))
            return create_env(level_id=level_id, seed=seed)

        with tempfile.TemporaryDirectory() as tmp, patch("train_dqn.create_env", side_effect=spy_create_env):
            res = train_dqn_curriculum(
                levels=[6],
                episodes_per_level=6,
                batch_size=2,
                warmup_steps=2,
                checkpoint_dir=tmp,
                device="cpu",
                map_seed=200,
                map_seed_interval=3,
                seed=42,
                eval_episodes=2,
            )

        self.assertEqual([r["map_seed"] for r in res["history"]], [s0, s0, s0, s1, s1, s1])
        self.assertEqual(calls, [(6, s0), (6, s1), (6, None)])
        self.assertIn(6, res["eval_summary"])

    def test_cli_parses_map_seed_flags(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--map-seed", "123", "--map-seed-interval", "3000"]):
            args = parse_tabular_args()
        self.assertEqual(args.map_seed, 123)
        self.assertEqual(args.map_seed_interval, 3000)

        proc = subprocess.run(
            [sys.executable, "train_dqn.py", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--map-seed", proc.stdout)
        self.assertIn("--map-seed-interval", proc.stdout)

    def test_train_py_rotates_map_seed_after_interval(self) -> None:
        from train import JobConfig, _run_single_job

        with tempfile.TemporaryDirectory() as tmp:
            # 1. Explicit --map-seed with interval=2 over 6 episodes -> 3 distinct blocks
            job_explicit = JobConfig(
                agent_type="tabular",
                agent_tag="tabular",
                mode="curriculum",
                reward_name="balanced",
                seed=42,
                levels=[1],
                episodes_per_level=6,
                eval_interval=6,
                eval_episodes_periodic=1,
                final_eval_episodes=1,
                untrained_eval_episodes=1,
                checkpoint_interval=6,
                alpha=0.12,
                alpha_min=0.02,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.997,
                hybrid_mode="expected",
                beta=0.25,
                action_repeat=1,
                rewarm_epsilon=0.30,
                rewarm_epsilon_ghost=0.35,
                replay_ratio=0.35,
                checkpoints_dir=f"{tmp}/cp1",
                outputs_dir=f"{tmp}/out1",
                map_seed=100,
                map_seed_interval=2,
                use_inline=False,
                is_multi_job=False,
            )
            _, res1 = _run_single_job(job_explicit)
            seeds1 = [r["map_seed"] for r in res1["episode_history"]]
            self.assertEqual(seeds1[0], 100)
            self.assertEqual(seeds1[0], seeds1[1])
            self.assertEqual(seeds1[2], seeds1[3])
            self.assertEqual(seeds1[4], seeds1[5])
            self.assertEqual(len({seeds1[0], seeds1[2], seeds1[4]}), 3)

            # 2. Omitted --map-seed (None) with interval=3 over 6 episodes -> derives from job.seed and rotates
            job_auto_42 = JobConfig(
                agent_type="tabular",
                agent_tag="tabular",
                mode="separate",
                reward_name="balanced",
                seed=42,
                levels=[1],
                episodes_per_level=6,
                eval_interval=6,
                eval_episodes_periodic=1,
                final_eval_episodes=1,
                untrained_eval_episodes=1,
                checkpoint_interval=6,
                alpha=0.12,
                alpha_min=0.02,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.997,
                hybrid_mode="expected",
                beta=0.25,
                action_repeat=1,
                rewarm_epsilon=0.30,
                rewarm_epsilon_ghost=0.35,
                replay_ratio=0.35,
                checkpoints_dir=f"{tmp}/cp2",
                outputs_dir=f"{tmp}/out2",
                map_seed=None,
                map_seed_interval=3,
                use_inline=False,
                is_multi_job=True,
            )
            _, res2 = _run_single_job(job_auto_42)
            seeds2 = [r["map_seed"] for r in res2["episode_history"]]
            self.assertIsNotNone(seeds2[0])
            self.assertEqual(seeds2[0], seeds2[1])
            self.assertEqual(seeds2[1], seeds2[2])
            self.assertEqual(seeds2[3], seeds2[4])
            self.assertEqual(seeds2[4], seeds2[5])
            self.assertNotEqual(seeds2[0], seeds2[3])


if __name__ == "__main__":
    unittest.main()
