import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from agent import HybridQSARSAAgent, TabularQAgent
from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from discretizer import MarioStateDiscretizer
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


class TestPhase10Convergence(unittest.TestCase):
    def test_airborne_pit_disambiguation(self) -> None:
        disc = MarioStateDiscretizer()
        self.assertEqual(disc.state_space_size, 55296)
        self.assertEqual(MarioStateDiscretizer(enable_wisp_sensor=True).state_space_size, 110592)

        # Grounded at dist_to_pit=20.0 -> IMMEDIATE_SAFE (2)
        self.assertEqual(
            disc.discretize(make_state(is_grounded=True, dist_to_pit=20.0, pit_fire_state=0))[4], 2
        )
        # Airborne approaching at dist_to_pit=20.0 -> FAR_SAFE (1)
        self.assertEqual(
            disc.discretize(make_state(is_grounded=False, dist_to_pit=20.0, pit_fire_state=0))[4], 1
        )
        # Airborne spanning pit at dist_to_pit=0.0 (safe) -> MID_PIT_AIRBORNE (2)
        self.assertEqual(
            disc.discretize(make_state(is_grounded=False, dist_to_pit=0.0, pit_fire_state=0))[4], 2
        )
        # Airborne over pit with active flame -> FIRE_DANGER (3)
        self.assertEqual(
            disc.discretize(make_state(is_grounded=False, dist_to_pit=0.0, pit_fire_state=2))[4], 3
        )

    def test_fire_cycle_prewarn_guard_and_reset(self) -> None:
        disc = MarioStateDiscretizer()
        # Observe active flame phase (2) before safe transition (2 -> 0)
        disc.discretize(make_state(player_x=800.0, dist_to_pit=40.0, pit_fire_state=2))
        # Edge tick (2 -> 0 sets _fire_safe_ticks = 0), followed by 47 more ticks (_fire_safe_ticks 1..47)
        for i in range(48):
            st = disc.discretize(make_state(player_x=810.0, dist_to_pit=30.0, pit_fire_state=0))
            self.assertEqual(st[4], 1, f"Expected FAR_SAFE (1) before tick 48 at iter {i}")
        # Tick 48 (_fire_safe_ticks == 48) triggers pre-warning -> FIRE_DANGER (3)
        st_pre = disc.discretize(make_state(player_x=810.0, dist_to_pit=30.0, pit_fire_state=0))
        self.assertEqual(st_pre[4], 3)

        # reset() clears _fire_phase_known so tick 1 returns FAR_SAFE (1)
        disc.reset()
        self.assertFalse(disc._fire_phase_known)
        st_post_reset = disc.discretize(make_state(player_x=810.0, dist_to_pit=30.0, pit_fire_state=0))
        self.assertEqual(st_post_reset[4], 1)

        # World-x jump > 40px (switching to next pit) clears _fire_phase_known
        disc.discretize(make_state(player_x=810.0, dist_to_pit=30.0, pit_fire_state=2))
        disc.discretize(make_state(player_x=810.0, dist_to_pit=30.0, pit_fire_state=0))
        self.assertTrue(disc._fire_phase_known)
        # Jump from pit_world_x=840 to next pit at pit_world_x=960 (delta=120 > 40)
        disc.discretize(make_state(player_x=920.0, dist_to_pit=40.0, pit_fire_state=0))
        self.assertFalse(disc._fire_phase_known)

    def test_laser_glitch_filter_rejects_dual_beam_swap(self) -> None:
        disc = MarioStateDiscretizer()
        # Tick 1: first contact (world_x = 1245) -> 1 (conservative threat)
        self.assertEqual(disc.discretize(make_state(player_x=1200.0, dist_to_laser=45.0))[7], 1)
        # Tick 2: world_x = 1248 (delta = +3.0, within [0.5, 15.0]) -> 2 (valid receding)
        self.assertEqual(disc.discretize(make_state(player_x=1204.0, dist_to_laser=44.0))[7], 2)
        # Tick 3: dual-beam swap world_x = 1278 (delta = +30.0 > 15.0) -> 1 (rejected glitch)
        self.assertEqual(disc.discretize(make_state(player_x=1208.0, dist_to_laser=70.0))[7], 1)
        # Tick 4: dist_to_laser = 25.0 <= 28.0 -> 1 (imminent threat)
        self.assertEqual(disc.discretize(make_state(player_x=1212.0, dist_to_laser=25.0))[7], 1)

    def test_hybrid_expected_hazard_gating_and_watkins_trace_cut(self) -> None:
        agent = HybridQSARSAAgent(
            mode="expected",
            epsilon=0.50,
            discount_factor=0.9,
            learning_rate=0.5,
            trace_decay=0.65,
            hazard_safe_expected=True,
        )
        s_safe_next = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        s_haz_pit = (1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0)
        s_haz_laser = (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)

        for st in (s_safe_next, s_haz_pit, s_haz_laser):
            agent.q_table[st] = np.array([0.0, 0.0, 0.0, 10.0, 20.0, 0.0], dtype=np.float32)

        # Safe next_state uses Expected SARSA: max=20, mean=5.0 -> 0.5*20 + 0.5*5 = 12.5 -> 2 + 0.9*12.5 = 13.25
        t_safe = agent.compute_td_target(2.0, s_safe_next, None, False)
        self.assertAlmostEqual(t_safe, 13.25, places=5)

        # Hazardous next_state (pit=2 or laser=1) uses pure max_next_q=20 -> 2 + 0.9*20 = 20.0
        self.assertAlmostEqual(agent.compute_td_target(2.0, s_haz_pit, None, False), 20.0, places=5)
        self.assertAlmostEqual(agent.compute_td_target(2.0, s_haz_laser, None, False), 20.0, places=5)

        # Strictly suboptimal exploratory action in hazardous state cuts prior eligibility traces
        s_safe = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        agent.q_table[s_safe][Action.RIGHT] = 5.0
        agent.reset_traces()
        agent.update(s_safe, Action.RIGHT, 0.0, s_haz_pit, False)
        self.assertIn((s_safe, Action.RIGHT), agent._traces)
        q_safe_before = float(agent.q_table[s_safe][Action.RIGHT])

        # Take strictly suboptimal Action.STAY (0.0 < max 20.0) in hazardous state -> -75.0 death
        agent.update(s_haz_pit, Action.STAY, -75.0, s_haz_pit, True)
        # Prior trace from s_safe must have been cut before applying -75.0 penalty
        self.assertAlmostEqual(float(agent.q_table[s_safe][Action.RIGHT]), q_safe_before, places=5)
        self.assertLess(float(agent.q_table[s_haz_pit][Action.STAY]), 0.0)

        # Serialization preserves hazard_safe_expected & hazard_expected_eps_cap
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "h_safe.json")
            agent.save(p)
            loaded = HybridQSARSAAgent(mode="expected", hazard_safe_expected=False, hazard_expected_eps_cap=0.5)
            loaded.load(p)
            self.assertTrue(loaded.hazard_safe_expected)
            self.assertAlmostEqual(loaded.hazard_expected_eps_cap, 0.0)

    def test_two_phase_trace_polish_tail_schedule(self) -> None:
        observed: List[Tuple[int, float, float, float]] = []

        class _SpyTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                if training and decay_eps:
                    observed.append(
                        (
                            len(observed) + 1,
                            float(agent.trace_decay),
                            float(agent.epsilon),
                            float(agent.alpha),
                        )
                    )
                return super()._run_episode(env, agent, training=training, decay_eps=decay_eps)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _SpyTrainer(
                env_factory=lambda lvl: _DummyEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                trace_decay=0.65,
                trace_decay_min=0.35,
                lambda_polish_zero=True,
                polish_epsilon_min=0.005,
                alpha_min=0.003,
                eval_interval=50,
                eval_episodes_periodic=1,
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

        self.assertEqual(len(observed), 100)
        self.assertAlmostEqual(observed[0][1], 0.65, places=6)   # Ep 1
        self.assertAlmostEqual(observed[74][1], 0.35, places=6)  # Ep 75
        for ep, lam, _, _ in observed[75:85]:                    # Ep 76..85 plateau
            self.assertAlmostEqual(lam, 0.35, places=6, msg=f"ep {ep}")
        self.assertAlmostEqual(observed[99][1], 0.0, places=6)   # Ep 100 lambda == 0.0
        self.assertLessEqual(observed[99][2], 0.005 + 1e-6)      # Ep 100 epsilon <= 0.005
        self.assertLessEqual(observed[99][3], 0.003 + 1e-6)      # Ep 100 alpha <= 0.003

        meta = board["metadata"]
        self.assertEqual(meta["trace_decay_schedule"], "two_phase_0.65_to_0.35_polish_to_0")
        self.assertAlmostEqual(meta["polish_epsilon_min"], 0.005)
        self.assertAlmostEqual(meta["alpha_min"], 0.003)

    def test_curriculum_best_promotes_stage4_from_stage_best_weights_and_logs_diagnostics(self) -> None:
        s_tag = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        class _SyntheticEnv(BaseEnvironment):
            def __init__(self, level_id: int):
                self.level_id = level_id
                self.max_steps = 5
                self.start_x = 50.0
                self.items = []

                class _Goal:
                    x = 1000.0

                self.goal = _Goal()

            def reset(self) -> RawState:
                return make_state(player_x=426.0 if self.level_id == 4 else 50.0)

            def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
                if self.level_id == 4:
                    return make_state(player_x=426.0), GameEvents(dx=0.0, fell_in_pit=True), True
                return make_state(player_x=54.0), GameEvents(dx=4.0, reached_goal=True), True

            def render(self) -> None:
                pass

            def close(self) -> None:
                pass

        class _Stage4BestTrainer(Trainer):
            def _run_episode(self, env, agent, training=True, decay_eps=True):
                res = super()._run_episode(env, agent, training=training, decay_eps=decay_eps)
                lvl = getattr(env, "level_id", 1)
                if training and decay_eps:
                    if not hasattr(self, "_ep_cnt"):
                        self._ep_cnt = {}
                    self._ep_cnt[lvl] = self._ep_cnt.get(lvl, 0) + 1
                    # During Stage 4, ep 5 sets q_table_lvl4_best marker (+999); ep 10 drifts final negative (-999)
                    if lvl <= 3:
                        agent.q_table[s_tag][Action.STAY] = 10.0 * lvl
                    elif lvl == 4:
                        agent.q_table[s_tag][Action.STAY] = 999.0 if self._ep_cnt[4] <= 5 else -999.0
                return res

            def evaluate_agent(self, agent, level_id, episodes, random_baseline=False):
                if random_baseline or episodes <= 0:
                    return super().evaluate_agent(agent, level_id, episodes, random_baseline=random_baseline)
                marker = float(agent.q_table.get(s_tag, np.zeros(6))[Action.STAY])
                # Stage 4 best checkpoint (marker ~ 999): retains 100/100/100 on L1-3 AND 40% on L4
                if marker > 500.0:
                    wr = 100.0 if level_id <= 3 else (40.0 if level_id == 4 else 0.0)
                    rew = 1000.0 if level_id <= 3 else 1500.0
                elif marker < -500.0:
                    # Drifted final weights at ep 10 of Stage 4: 100/100/0/0
                    wr = 100.0 if level_id <= 2 else 0.0
                    rew = 1000.0 if level_id <= 2 else 200.0
                else:
                    # Stages 1..3
                    max_solved = 1 if marker < 15.0 else (2 if marker < 25.0 else 3)
                    wr = 100.0 if level_id <= max_solved else 0.0
                    rew = 1000.0 if wr == 100.0 else 100.0
                return {
                    "win_rate": wr,
                    "avg_reward": rew,
                    "avg_completion_steps": 100.0 if wr > 0 else -1.0,
                    "avg_items_collected": 0.0,
                    "outcome_counts": {"win": episodes} if wr == 100.0 else {"pit_death": episodes},
                }

        with tempfile.TemporaryDirectory() as tmp:
            trainer = _Stage4BestTrainer(
                env_factory=lambda lvl: _SyntheticEnv(level_id=lvl),
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                replay_ratio=0.0,
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=5,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmp, "cp"),
                outputs_dir=os.path.join(tmp, "out"),
            )
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[1, 2, 3, 4],
                episodes_per_level=10,
            )

        ev = board["evaluation_summary"]
        self.assertEqual(ev["curriculum_best_checkpoint_stage"], 4)
        self.assertEqual(ev["curriculum_best_retention"]["level_3"]["win_rate"], 100.0)
        self.assertEqual(ev["curriculum_best_retention"]["level_4"]["win_rate"], 40.0)
        l4_entry = ev["level_4"]
        self.assertEqual(l4_entry["best_checkpoint_episode"], 5)
        self.assertIn("pit_death", l4_entry["top_death_clusters"])
        self.assertEqual(l4_entry["top_death_clusters"]["pit_death"][0]["x_bin"], 420)
        self.assertEqual(l4_entry["top_death_clusters"]["pit_death"][0]["count"], 10)


if __name__ == "__main__":
    unittest.main()
