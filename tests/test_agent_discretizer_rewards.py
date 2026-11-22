import os
import tempfile
import unittest

from core.base_env import Action, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy
from discretizer import MarioStateDiscretizer, PlayerStateDiscretizer
from rewards import BalancedRewardStrategy, DenseRewardStrategy, SparseRewardStrategy
from agent import TabularQAgent


def make_raw_state(
    player_x: float = 50.0,
    player_y: float = 288.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    is_grounded: bool = True,
    can_double_jump: bool = False,
    is_crouching: bool = False,
    dist_to_obstacle: float = -1.0,
    obstacle_height: float = 0.0,
    dist_to_pit: float = -1.0,
    pit_fire_state: int = 0,
    dist_to_ghost: float = -1.0,
    ghost_y: float = 0.0,
    dist_to_item: float = -1.0,
    dist_to_laser: float = -1.0,
    dist_to_goal: float = 500.0,
) -> RawState:
    return RawState(
        player_x=player_x,
        player_y=player_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        is_grounded=is_grounded,
        can_double_jump=can_double_jump,
        is_crouching=is_crouching,
        dist_to_obstacle=dist_to_obstacle,
        obstacle_height=obstacle_height,
        dist_to_pit=dist_to_pit,
        pit_fire_state=pit_fire_state,
        dist_to_ghost=dist_to_ghost,
        ghost_y=ghost_y,
        dist_to_item=dist_to_item,
        dist_to_laser=dist_to_laser,
        dist_to_goal=dist_to_goal,
    )


class TestDiscretizer(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = MarioStateDiscretizer()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.discretizer, BaseDiscretizer)
        self.assertTrue(issubclass(PlayerStateDiscretizer, BaseDiscretizer))

    def test_discretize_structure_and_types(self) -> None:
        raw = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=50.0,
            obstacle_height=48.0,
            dist_to_pit=60.0,
            dist_to_ghost=50.0,
            ghost_y=280.0,
            dist_to_item=30.0,
            dist_to_goal=500.0,
        )
        state = self.discretizer.discretize(raw)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 9)
        self.assertTrue(all(isinstance(x, int) for x in state))

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

        state = self.discretizer.discretize(LegacyState())
        self.assertEqual(len(state), 9)

    def test_ghost_dist_and_alt_bins(self) -> None:
        # Low ghost (jump over)
        raw_low = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=280.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_low = self.discretizer.discretize(raw_low)
        self.assertEqual(s_low[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_low[6], 0)  # ghost_alt_bin: LOW

        # High ghost (walk under)
        raw_high = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=234.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_high = self.discretizer.discretize(raw_high)
        self.assertEqual(s_high[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_high[6], 1)  # ghost_alt_bin: HIGH


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.s1 = make_raw_state(player_x=50.0, player_y=288.0, dist_to_goal=500.0)
        self.s2 = make_raw_state(player_x=54.0, player_y=288.0, dist_to_goal=496.0)

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
        s1 = (1, 0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 1, 0, 0, 0, 0)

        action = agent.select_action(s1)
        self.assertIn(action, list(range(Action.NUM_ACTIONS)))

        # Update Q-table
        td_err = agent.update(s1, action, reward=10.0, next_state=s2, done=False)
        self.assertIsInstance(td_err, float)
        self.assertGreater(agent.q_table[s1][action], 0.0)

    def test_save_and_load_pickle_and_json(self) -> None:
        agent = TabularQAgent(seed=123)
        s = (1, 0, 2, 0, 1, 0, 0)
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
