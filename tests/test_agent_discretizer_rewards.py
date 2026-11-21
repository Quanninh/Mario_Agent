import os
import tempfile
import unittest

from core.base_env import Action, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy
from discretizer import MarioStateDiscretizer, PlayerStateDiscretizer
from rewards import BalancedRewardStrategy, DenseRewardStrategy, SparseRewardStrategy
from agent import TabularQAgent


class TestDiscretizer(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = MarioStateDiscretizer()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.discretizer, BaseDiscretizer)
        self.assertTrue(issubclass(PlayerStateDiscretizer, BaseDiscretizer))

    def test_discretize_structure_and_types(self) -> None:
        raw = RawState(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=50.0,
            obstacle_height=48.0,
            dist_to_pit=60.0,
            dist_to_ghost=50.0,
            ghost_y=280.0,
            ghost_vx=-1.5,
            dist_to_item=30.0,
            dist_to_goal=500.0,
        )
        state = self.discretizer.discretize(raw)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 8)
        self.assertTrue(all(isinstance(x, int) for x in state))
        self.assertEqual(state[7], 1)  # ghost_approaching

    def test_legacy_attribute_compatibility(self) -> None:
        # If legacy state has mario_y instead of player_y, it should not crash
        class LegacyState:
            mario_y = 288.0
            is_grounded = True
            velocity_y = 0.0
            dist_to_obstacle = 20.0
            obstacle_height = 40.0
            dist_to_pit = -1.0
            dist_to_ghost = -1.0
            ghost_y = 0.0
            ghost_vx = 0.0

        state = self.discretizer.discretize(LegacyState())
        self.assertEqual(len(state), 8)

    def test_ghost_dist_and_alt_bins(self) -> None:
        # Low ghost (jump over)
        raw_low = RawState(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=280.0,
            ghost_vx=0.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_low = self.discretizer.discretize(raw_low)
        self.assertEqual(len(s_low), 8)
        self.assertEqual(s_low[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_low[6], 0)  # ghost_alt_bin: LOW
        self.assertEqual(s_low[7], 0)  # ghost_approaching: 0 (speed 0.0)

        # High ghost (walk under)
        raw_high = RawState(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=234.0,
            ghost_vx=-2.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_high = self.discretizer.discretize(raw_high)
        self.assertEqual(len(s_high), 8)
        self.assertEqual(s_high[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_high[6], 1)  # ghost_alt_bin: HIGH
        self.assertEqual(s_high[7], 1)  # ghost_approaching: 1 (speed -2.0)

    def test_ghost_approaching_directionality(self) -> None:
        """Verifies state[7] encodes ghost approach velocity vector to eliminate perceptual aliasing."""
        base_kwargs = dict(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            ghost_y=280.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )

        # 1. Oncoming ghost (approaching Hoá, vx < -0.01) -> state[7] == 1
        raw_approaching = RawState(dist_to_ghost=50.0, ghost_vx=-1.5, **base_kwargs)
        s_app = self.discretizer.discretize(raw_approaching)
        self.assertEqual(s_app[7], 1)

        # 2. Retreating ghost (flying away from Hoá, vx > 0) -> state[7] == 0
        raw_retreating = RawState(dist_to_ghost=50.0, ghost_vx=1.5, **base_kwargs)
        s_ret = self.discretizer.discretize(raw_retreating)
        self.assertEqual(s_ret[7], 0)

        # 3. No ghost ahead (dist_to_ghost == -1.0) -> state[7] == 0
        raw_no_ghost = RawState(dist_to_ghost=-1.0, ghost_vx=-1.5, **base_kwargs)
        s_none = self.discretizer.discretize(raw_no_ghost)
        self.assertEqual(s_none[7], 0)

        # 4. Stationary ghost (vx == 0.0) -> state[7] == 0
        raw_stationary = RawState(dist_to_ghost=50.0, ghost_vx=0.0, **base_kwargs)
        s_stat = self.discretizer.discretize(raw_stationary)
        self.assertEqual(s_stat[7], 0)


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.s1 = RawState(50.0, 288.0, 0.0, True, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 500.0)
        self.s2 = RawState(54.0, 288.0, 0.0, True, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 496.0)

    def test_balanced_reward_ghost_and_item(self) -> None:
        strategy = BalancedRewardStrategy(
            goal_bonus=200.0,
            pit_penalty=-75.0,
            ghost_penalty=-75.0,
            item_bonus=25.0,
            step_penalty=-0.1,
        )
        self.assertIsInstance(strategy, BaseRewardStrategy)

        # Normal forward step
        ev_norm = GameEvents(dx=4.0)
        r_norm = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_norm, False)
        self.assertAlmostEqual(r_norm, 3.9)  # 4.0 - 0.1

        # Death by ghost: MUST be severely penalized
        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        r_ghost = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True)
        self.assertAlmostEqual(r_ghost, -71.1)  # 4.0 - 75.0 - 0.1

        # Death by pit: MUST be severely penalized
        ev_pit = GameEvents(dx=4.0, fell_in_pit=True)
        r_pit = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_pit, True)
        self.assertAlmostEqual(r_pit, -71.1)

        # Warmth item collection bonus
        ev_item = GameEvents(dx=4.0, collected_item=True)
        r_item = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False)
        self.assertAlmostEqual(r_item, 28.9)  # 4.0 + 25.0 - 0.1

        # Goal reached
        ev_goal = GameEvents(dx=4.0, reached_goal=True)
        r_goal = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_goal, True)
        self.assertAlmostEqual(r_goal, 203.9)  # 4.0 + 200.0 - 0.1

    def test_balanced_reward_safe_waiting_dynamics(self) -> None:
        """Verifies tactical hesitation (Action.STAY) reward calibration and waiting incentive."""
        strategy = BalancedRewardStrategy(step_penalty=-0.02, wait_bonus=0.05)

        s_oncoming = RawState(50.0, 288.0, 0.0, True, -1.0, 0.0, -1.0, 50.0, 280.0, -1.5, -1.0, 400.0)
        s_retreating = RawState(50.0, 288.0, 0.0, True, -1.0, 0.0, -1.0, 50.0, 280.0, 1.5, -1.0, 400.0)

        # 1. Action.STAY with oncoming ghost in FAR zone: offsets step penalty with wait_bonus
        ev_stay = GameEvents(dx=0.0)
        r_wait = strategy.calculate_reward(s_oncoming, Action.STAY, s_oncoming, ev_stay, False)
        self.assertAlmostEqual(r_wait, -0.02 + 0.05)  # +0.03

        # 2. Action.RIGHT towards oncoming ghost: gets forward progress minus step penalty (no wait bonus)
        ev_right = GameEvents(dx=4.0)
        r_right = strategy.calculate_reward(s_oncoming, Action.RIGHT, s_oncoming, ev_right, False)
        self.assertAlmostEqual(r_right, 4.0 - 0.02)

        # 3. Action.STAY with retreating ghost: only receives mild step penalty (no wait bonus)
        r_stay_retreat = strategy.calculate_reward(s_retreating, Action.STAY, s_retreating, ev_stay, False)
        self.assertAlmostEqual(r_stay_retreat, -0.02)

    def test_dense_reward_ghost_and_item(self) -> None:
        strategy = DenseRewardStrategy(death_penalty=-60.0, item_bonus=20.0)
        self.assertIsInstance(strategy, BaseRewardStrategy)

        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        r_ghost = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True)
        self.assertLess(r_ghost, 0.0)

        ev_item = GameEvents(dx=4.0, collected_item=True)
        r_item = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False)
        self.assertGreater(r_item, 20.0)

    def test_sparse_reward_ghost_and_item(self) -> None:
        strategy = SparseRewardStrategy(goal_bonus=100.0, death_penalty=-50.0, item_bonus=10.0)
        self.assertIsInstance(strategy, BaseRewardStrategy)

        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        self.assertEqual(strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True), -50.0)

        ev_item = GameEvents(dx=4.0, collected_item=True)
        self.assertEqual(strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False), 10.0)


class TestTabularQAgent(unittest.TestCase):
    def test_implements_interface(self) -> None:
        agent = TabularQAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_action_selection_and_update(self) -> None:
        agent = TabularQAgent(seed=42)
        s1 = (1, 0, 0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 1, 0, 0, 0, 0, 0)

        action = agent.select_action(s1)
        self.assertIn(action, list(range(Action.NUM_ACTIONS)))

        # Update Q-table
        td_err = agent.update(s1, action, reward=10.0, next_state=s2, done=False)
        self.assertIsInstance(td_err, float)
        self.assertGreater(agent.q_table[s1][action], 0.0)

    def test_save_and_load_pickle_and_json(self) -> None:
        agent = TabularQAgent(seed=123)
        s = (1, 0, 2, 0, 1, 0, 0, 0)
        agent.q_table[s][Action.RIGHT] = 42.5

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "agent.pkl")
            json_path = os.path.join(tmpdir, "agent.json")

            # Test Pickle
            agent.save(pkl_path)
            new_agent_pkl = TabularQAgent()
            new_agent_pkl.load_q_table(pkl_path)  # Test alias
            self.assertAlmostEqual(new_agent_pkl.q_table[s][Action.RIGHT], 42.5)

            # Test JSON
            agent.save(json_path)
            new_agent_json = TabularQAgent()
            new_agent_json.load(json_path)
            self.assertAlmostEqual(new_agent_json.q_table[s][Action.RIGHT], 42.5)


class TestFullSystemIntegration(unittest.TestCase):
    """End-to-end headless integration of Environment, Discretizer, Reward, and Agent."""

    def test_train_steps_all_5_levels(self) -> None:
        from env import LEVEL_BUILDERS, create_env

        discretizer = MarioStateDiscretizer()
        reward_strategy = BalancedRewardStrategy()
        agent = TabularQAgent(seed=42)

        for level_id in LEVEL_BUILDERS:
            env = create_env(level_id)
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)

            for _ in range(100):
                action = agent.select_action(discrete_state)
                next_raw_state, events, done = env.step(action)
                next_discrete_state = discretizer.discretize(next_raw_state)

                reward = reward_strategy.calculate_reward(
                    raw_state, action, next_raw_state, events, done
                )
                agent.update(discrete_state, action, reward, next_discrete_state, done)

                if done:
                    raw_state = env.reset()
                    discrete_state = discretizer.discretize(raw_state)
                else:
                    raw_state = next_raw_state
                    discrete_state = next_discrete_state

        # Assert agent discovered discrete states across the 5 levels
        self.assertGreater(len(agent.q_table), 0)


if __name__ == "__main__":
    unittest.main()
