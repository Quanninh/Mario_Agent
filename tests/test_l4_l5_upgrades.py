import os
import random
import tempfile
import unittest
from typing import List, Optional, Tuple
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


class _SyntheticEventEnv(BaseEnvironment):
    """Minimal mock environment emitting a single preconfigured terminal GameEvents."""

    def __init__(self, level_id: int = 1, max_steps: int = 1000, event_on_step: Optional[GameEvents] = None):
        self.level_id = level_id
        self.max_steps = max_steps
        self.start_x = 50.0
        self.items = []
        self.event_on_step = event_on_step
        self.step_count = 0
        self.closed = False

        class _Goal:
            x = 1000.0

        self.goal = _Goal()

    def reset(self) -> RawState:
        self.step_count = 0
        return make_state(player_x=self.start_x)

    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
        self.step_count += 1
        if self.event_on_step is not None:
            done = bool(
                self.event_on_step.reached_goal
                or self.event_on_step.hit_ghost
                or self.event_on_step.hit_fire
                or self.event_on_step.hit_laser
                or self.event_on_step.hit_wisp
                or self.event_on_step.hit_monster
                or self.event_on_step.fell_in_pit
            )
            return make_state(player_x=self.start_x + 4.0), self.event_on_step, done

        # Stalling environment (tests max_steps_cap timeout termination)
        ev = GameEvents(dx=0.0)
        done = self.step_count >= self.max_steps
        return make_state(player_x=self.start_x), ev, done

    def render(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class TestDiscretizerUpgrades(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = MarioStateDiscretizer()

    def test_state_space_size_and_tuple_length(self) -> None:
        self.assertEqual(self.discretizer.state_space_size, 55296)
        self.assertEqual(MarioStateDiscretizer(enable_wisp_sensor=True).state_space_size, 110592)
        d_state = self.discretizer.discretize(make_state())
        self.assertEqual(len(d_state), 12)

    def test_pit_fire_warning_bins(self) -> None:
        # Warning embers at 50px -> FIRE_DANGER (3)
        s_warn = self.discretizer.discretize(make_state(dist_to_pit=50.0, pit_fire_state=1))
        self.assertEqual(s_warn[4], 3)

        # Safe pit at 20px (<= 24px) -> IMMEDIATE_SAFE (2)
        s_imm_safe = self.discretizer.discretize(make_state(dist_to_pit=20.0, pit_fire_state=0))
        self.assertEqual(s_imm_safe[4], 2)

        # Safe pit at 30px (> 24px) -> FAR_SAFE (1)
        s_mid_safe = self.discretizer.discretize(make_state(dist_to_pit=30.0, pit_fire_state=0))
        self.assertEqual(s_mid_safe[4], 1)

        # Far pit at 80px even with embers -> FAR_SAFE (1)
        s_far = self.discretizer.discretize(make_state(dist_to_pit=80.0, pit_fire_state=1))
        self.assertEqual(s_far[4], 1)

    def test_vy_double_jump_window_bins(self) -> None:
        self.assertEqual(self.discretizer.discretize(make_state(is_grounded=False, velocity_y=-10.0))[1], 1)
        self.assertEqual(self.discretizer.discretize(make_state(is_grounded=False, velocity_y=-1.5))[1], 0)
        self.assertEqual(self.discretizer.discretize(make_state(is_grounded=False, velocity_y=4.0))[1], -1)
        self.assertEqual(self.discretizer.discretize(make_state(is_grounded=True, velocity_y=-5.0))[1], 0)

    def test_three_bin_laser_proximity(self) -> None:
        self.discretizer.reset()
        self.assertEqual(self.discretizer.discretize(make_state(player_x=100.0, dist_to_laser=70.0))[7], 1)
        # Receding beam in world frame: world_lx 170 -> 172 => RECEDING (2)
        self.assertEqual(self.discretizer.discretize(make_state(player_x=104.0, dist_to_laser=68.0))[7], 2)
        # Imminent <= 20px always CLOSING / IMMINENT (1)
        self.assertEqual(self.discretizer.discretize(make_state(player_x=108.0, dist_to_laser=18.0))[7], 1)
        self.assertEqual(self.discretizer.discretize(make_state(player_x=112.0, dist_to_laser=100.0))[7], 0)


class TestBalancedRewardUpgrades(unittest.TestCase):
    def setUp(self) -> None:
        self.balanced = BalancedRewardStrategy()
        self.dense = DenseRewardStrategy()
        self.sparse = SparseRewardStrategy()

    def test_early_double_jump_penalty(self) -> None:
        ev = GameEvents(dx=4.0)
        s_fast = make_state(is_grounded=False, can_double_jump=True, is_crouching=False, velocity_y=-10.0)
        s_apex = make_state(is_grounded=False, can_double_jump=True, is_crouching=False, velocity_y=-1.0)

        r_fast_jump = self.balanced.calculate_reward(s_fast, Action.RIGHT_JUMP, s_fast, ev, False)
        r_fast_stay = self.balanced.calculate_reward(s_fast, Action.RIGHT, s_fast, ev, False)
        self.assertAlmostEqual(r_fast_jump - r_fast_stay, -3.0)

        r_apex_jump = self.balanced.calculate_reward(s_apex, Action.RIGHT_JUMP, s_apex, ev, False)
        r_apex_stay = self.balanced.calculate_reward(s_apex, Action.RIGHT, s_apex, ev, False)
        self.assertAlmostEqual(r_apex_jump - r_apex_stay, 0.0)

    def test_synced_pit_and_laser_exemption_horizons(self) -> None:
        ev = GameEvents(dx=4.0)
        s_open = make_state(is_grounded=True)
        r_walk = self.balanced.calculate_reward(s_open, Action.RIGHT, s_open, ev, False)

        # Open ground -> -15.0
        r_open_jump = self.balanced.calculate_reward(s_open, Action.RIGHT_JUMP, s_open, ev, False)
        self.assertAlmostEqual(r_open_jump - r_walk, -15.0)

        # Pit at 55px (<= 65px synced horizon) -> exempt
        s_pit = make_state(is_grounded=True, dist_to_pit=55.0)
        r_pit_jump = self.balanced.calculate_reward(s_pit, Action.RIGHT_JUMP, s_pit, ev, False)
        self.assertAlmostEqual(r_pit_jump, r_walk)

        # Laser at 75px -> penalized (-15.0) by default (-1.0), exempt when laser_threat_distance=85.0
        s_laser = make_state(is_grounded=True, dist_to_laser=75.0)
        r_laser_jump_default = self.balanced.calculate_reward(s_laser, Action.RIGHT_JUMP, s_laser, ev, False)
        self.assertAlmostEqual(r_laser_jump_default - r_walk, -15.0)
        bal_laser_exempt = BalancedRewardStrategy(laser_threat_distance=85.0)
        r_laser_jump_exempt = bal_laser_exempt.calculate_reward(s_laser, Action.RIGHT_JUMP, s_laser, ev, False)
        self.assertAlmostEqual(r_laser_jump_exempt, r_walk)

    def test_dense_and_sparse_untouched(self) -> None:
        s1 = make_state(player_x=100.0, dist_to_goal=400.0)
        s2 = make_state(player_x=104.0, dist_to_goal=396.0)
        ev = GameEvents(dx=4.0)
        self.assertAlmostEqual(self.dense.calculate_reward(s1, Action.RIGHT, s2, ev, False), 4.8)
        self.assertEqual(self.sparse.calculate_reward(s1, Action.RIGHT, s2, ev, False), 0.0)


class TestAgentFixesAndTraces(unittest.TestCase):
    def test_readonly_greedy_selection_does_not_inflate_q_table(self) -> None:
        for agent in (TabularQAgent(seed=42), HybridQSARSAAgent(mode="expected", seed=42)):
            agent.set_mode(False)
            for i in range(10):
                unseen = (1, 0, i % 4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                act = agent.select_action(unseen)
                self.assertIn(act, range(Action.NUM_ACTIONS))
            self.assertEqual(len(agent.q_table), 0)

    def test_hybrid_update_positional_args_preserved(self) -> None:
        agent = HybridQSARSAAgent(beta=0.5, mode="linear", learning_rate=0.1, discount_factor=0.9)
        s = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        agent.q_table[s2][Action.RIGHT] = 10.0
        agent.q_table[s2][Action.RIGHT_JUMP] = 20.0

        # Positional call: update_hybrid(state, action, reward, next_state, next_action, done)
        td_err = agent.update_hybrid(s, Action.RIGHT, 2.0, s2, Action.RIGHT, False)
        self.assertAlmostEqual(td_err, 15.5, places=5)
        self.assertAlmostEqual(float(agent.q_table[s][Action.RIGHT]), 1.55, places=5)

    def test_replacing_traces_propagate_and_watkins_cuts_exploratory(self) -> None:
        s0 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        s1 = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # 1. Greedy trajectory propagates terminal +100 back to s0 and s1 in 1 episode
        for agent in (
            TabularQAgent(learning_rate=0.5, discount_factor=0.9, trace_decay=0.65, epsilon=0.0, seed=1),
            HybridQSARSAAgent(mode="expected", learning_rate=0.5, discount_factor=0.9, trace_decay=0.65, epsilon=0.0, seed=1),
        ):
            agent.reset_traces()
            agent.update(s0, Action.RIGHT, 0.0, s1, False)
            agent.update(s1, Action.RIGHT, 0.0, s2, False)
            agent.update(s2, Action.RIGHT, 100.0, s2, True)
            self.assertGreater(float(agent.q_table[s0][Action.RIGHT]), 0.0)
            self.assertGreater(float(agent.q_table[s1][Action.RIGHT]), 0.0)
            self.assertEqual(len(agent._traces), 0)

        # 2. Watkins's Q(lambda) cuts prior traces on strictly suboptimal exploratory action
        q_agent = TabularQAgent(learning_rate=0.5, discount_factor=0.9, trace_decay=0.65, epsilon=1.0, seed=42)
        q_agent.q_table[s1][Action.RIGHT] = 10.0  # Make STAY/LEFT/JUMP strictly suboptimal
        q_agent.reset_traces()
        q_agent.update(s0, Action.RIGHT, 0.0, s1, False)
        q_s0_before_exp = float(q_agent.q_table[s0][Action.RIGHT])
        # Choose suboptimal exploratory actions via select_action until we get a non-RIGHT action
        chosen = Action.RIGHT
        for _ in range(50):
            chosen = q_agent.select_action(s1)
            if chosen != Action.RIGHT:
                break
        self.assertNotEqual(chosen, Action.RIGHT)
        self.assertTrue(q_agent._last_action_exploratory)
        q_agent.update(s1, chosen, 100.0, s2, True)
        # s0 must NOT receive any of the +100 exploratory backup because Watkins cut the trace
        self.assertAlmostEqual(float(q_agent.q_table[s0][Action.RIGHT]), q_s0_before_exp)
        self.assertGreater(float(q_agent.q_table[s1][chosen]), 0.0)

        # 3. reset_traces clears traces between episodes
        q_agent.update(s0, Action.RIGHT, 1.0, s1, False)
        self.assertGreater(len(q_agent._traces), 0)
        q_agent.reset_traces()
        self.assertEqual(len(q_agent._traces), 0)

    def test_trace_decay_serialization_and_legacy_fallback(self) -> None:
        agent = HybridQSARSAAgent(mode="expected", trace_decay=0.65)
        s = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        agent.q_table[s][Action.RIGHT] = 12.0
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "h.json")
            agent.save(p)
            loaded = HybridQSARSAAgent(mode="expected", trace_decay=0.0)
            loaded.load(p)
            self.assertAlmostEqual(loaded.trace_decay, 0.65)


class TestTrainerUpgrades(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = MarioStateDiscretizer()
        self.reward = BalancedRewardStrategy()

    def test_outcome_classification_all_event_types(self) -> None:
        cases = [
            (GameEvents(reached_goal=True), "win"),
            (GameEvents(hit_ghost=True), "ghost_death"),
            (GameEvents(hit_fire=True), "fire_death"),
            (GameEvents(hit_laser=True), "laser_death"),
            (GameEvents(hit_wisp=True), "wisp_death"),
            (GameEvents(hit_monster=True), "monster_death"),
            (GameEvents(fell_in_pit=True), "pit_death"),
            (None, "timeout"),
        ]
        trainer = Trainer(
            env_factory=lambda lvl: _SyntheticEventEnv(lvl),
            discretizer=self.discretizer,
            reward_strategy=self.reward,
        )
        agent = TabularQAgent(seed=1)
        for ev, expected in cases:
            if ev is None:
                env = _SyntheticEventEnv(max_steps=3, event_on_step=None)
            else:
                env = _SyntheticEventEnv(max_steps=100, event_on_step=ev)
            res = trainer._run_episode(env, agent, training=False)
            self.assertEqual(res["outcome"], expected)

        # Verify evaluate_agent returns outcome_counts
        trainer_fire = Trainer(
            env_factory=lambda lvl: _SyntheticEventEnv(lvl, event_on_step=GameEvents(hit_fire=True)),
            discretizer=self.discretizer,
            reward_strategy=self.reward,
        )
        eval_res = trainer_fire.evaluate_agent(agent, level_id=4, episodes=3)
        self.assertEqual(eval_res["outcome_counts"]["fire_death"], 3)

    def test_max_steps_cap_terminates_level5_at_2400_with_timeout_penalty(self) -> None:
        from env import create_env

        stalling_l5 = _SyntheticEventEnv(level_id=5, max_steps=10000, event_on_step=None)
        trainer = Trainer(
            env_factory=lambda lvl: stalling_l5,
            discretizer=self.discretizer,
            reward_strategy=self.reward,
            max_steps_cap=2400,
        )
        agent = TabularQAgent(epsilon=0.0, seed=42)
        d_start = self.discretizer.discretize(make_state(player_x=stalling_l5.start_x))
        agent.q_table[d_start][Action.STAY] = 1.0  # Prevent random jumps on open ground
        res = trainer._run_episode(stalling_l5, agent, training=False)
        self.assertEqual(res["steps"], 2400)
        self.assertEqual(res["outcome"], "timeout")
        # Every step gives -0.1 step cost (2400 * -0.1 = -240.0) plus -50.0 timeout penalty = -290.0
        self.assertAlmostEqual(res["total_reward"], -290.0, places=2)

        # Confirm Level 5 native max_steps is indeed 10000
        real_l5 = create_env(5)
        self.assertEqual(real_l5.max_steps, 10000)
        real_l5.close()

    def test_curriculum_replay_covers_immediate_previous_levels_and_caches_envs(self) -> None:
        created_levels: List[int] = []
        created_envs: List[_SyntheticEventEnv] = []
        replayed_level_counts: dict = {}

        class _TrackingEnv(_SyntheticEventEnv):
            def reset(self) -> RawState:
                replayed_level_counts[self.level_id] = replayed_level_counts.get(self.level_id, 0) + 1
                return super().reset()

        def factory(lvl: int) -> BaseEnvironment:
            created_levels.append(lvl)
            e = _TrackingEnv(level_id=lvl, max_steps=5, event_on_step=GameEvents(reached_goal=True))
            created_envs.append(e)
            return e

        random.seed(42)
        np.random.seed(42)
        trainer = Trainer(
            env_factory=factory,
            discretizer=self.discretizer,
            reward_strategy=self.reward,
            replay_ratio=1.0,
            eval_interval=100,
            eval_episodes_periodic=0,
            final_eval_episodes=0,
            untrained_eval_episodes=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer.checkpoints_dir = os.path.join(tmp, "cp")
            trainer.outputs_dir = os.path.join(tmp, "out")
            board = trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="curriculum",
                levels=[1, 2, 3, 4],
                episodes_per_level=20,
            )

        # Caching check: exactly 1 training env + 1 cached replay env for levels 1..3; only 1 training env for level 4
        self.assertEqual(created_levels.count(1), 2)
        self.assertEqual(created_levels.count(2), 2)
        self.assertEqual(created_levels.count(3), 2)
        self.assertEqual(created_levels.count(4), 1)

        # Replay check: L2 replays L1, L3 replays {1,2}, L4 replays {1,2,3}
        self.assertGreaterEqual(replayed_level_counts[1] - 20, 20)
        self.assertGreater(replayed_level_counts[2] - 20, 0)
        self.assertGreater(replayed_level_counts[3] - 20, 0)
        self.assertEqual(replayed_level_counts[4], 20)
        # All created envs (including cached replay envs) were closed cleanly
        self.assertTrue(all(e.closed for e in created_envs))
        self.assertEqual(board["metadata"]["state_space_max"], 55296)
        self.assertIn("post_curriculum_retention", board["evaluation_summary"])
        self.assertIn("best_checkpoint_retention", board["evaluation_summary"])


if __name__ == "__main__":
    unittest.main()
