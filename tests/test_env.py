import random
import unittest

from core import Action
from env import GameEnvironment, Ghost, GoalFlag, LEVEL_BUILDERS, Pit, WarmthItem, create_env

class TestGameEnvironment(unittest.TestCase):
    """Automated headless tests for Hoá's Midnight Escape environment."""
    def test_all_levels_headless_smoke(self) -> None:
        """All 5 levels must run 500 random steps headlessly with zero crashes."""
        rng = random.Random(42)
        for level_id in LEVEL_BUILDERS:
            env = create_env(level_id)
            state = env.reset()
            self.assertEqual(state.player_x, env.start_x)
            for _ in range(500):
                action = rng.randrange(Action.NUM_ACTIONS)
                state, events, done = env.step(action)
                if done:
                    state = env.reset()
            # Verify Pygame window was NEVER opened during headless stepping
            self.assertIsNone(env._screen)

    def test_reset_restores_player_ghosts_and_items(self) -> None:
        """reset() must restore Hoá, all ghosts, and all collected warmth items."""
        env = create_env(level_id=4)
        env.reset()
        initial_ghost_x = [g.x for g in env.ghosts]
        for _ in range(25):
            env.step(Action.RIGHT)
        env.items[0].is_collected = True
        # Reset and verify everything returned to initial state
        state = env.reset()
        self.assertEqual(state.player_x, env.start_x)
        self.assertEqual(env.step_count, 0)
        self.assertEqual([g.x for g in env.ghosts], initial_ghost_x)
        self.assertTrue(all(not item.is_collected for item in env.items))

    def test_collect_warmth_item_once(self) -> None:
        """Walking into a WarmthItem sets collected_item=True only once."""
        item = WarmthItem(x=74.0, y=296.0)
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[item])
        env.reset()
        _, events_1, done_1 = env.step(Action.RIGHT)
        self.assertTrue(events_1.collected_item)
        self.assertTrue(item.is_collected)
        self.assertFalse(done_1)
        # Next step should NOT double-collect the same item
        _, events_2, _ = env.step(Action.STAY)
        self.assertFalse(events_2.collected_item)

    def test_ghost_collision_ends_episode(self) -> None:
        """Touching a ghost must set hit_ghost=True, is_alive=False, and done=True."""
        ghost = Ghost(
            x=74.0, y=290.0,
            patrol_min_x=50.0, patrol_max_x=200.0,
            speed=0.0, float_amplitude=0.0,
        )
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[ghost], items=[])
        env.reset()
        _, events, done = env.step(Action.RIGHT)
        self.assertTrue(events.hit_ghost)
        self.assertTrue(done)
        self.assertFalse(env.player.is_alive)

    def test_pit_fall_ends_episode(self) -> None:
        """Falling into a grave pit must set fell_in_pit=True and done=True."""
        pit = Pit(x_start=50.0, x_end=150.0)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], items=[])
        env.reset()
        done = False
        events = None
        for _ in range(40):
            _, events, done = env.step(Action.RIGHT)
            if done:
                break
        self.assertTrue(done)
        self.assertIsNotNone(events)
        self.assertTrue(events.fell_in_pit)
        self.assertFalse(env.player.is_alive)

    def test_reach_home_wins_episode(self) -> None:
        """Reaching Hoá's house must set reached_goal=True and done=True."""
        goal = GoalFlag(x=80.0, y=232.0)
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[], goal=goal)
        env.reset()
        done = False
        events = None
        for _ in range(5):
            _, events, done = env.step(Action.RIGHT)
            if done:
                break
        self.assertTrue(done)
        self.assertIsNotNone(events)
        self.assertTrue(events.reached_goal)

if __name__ == "__main__":
    unittest.main()