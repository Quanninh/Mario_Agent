import random
import unittest

from core import Action, GameEvents
from env import GameEnvironment, Ghost, GoalFlag, LEVEL_BUILDERS, Pit, WarmthItem, create_env

class TestGameEnvironment(unittest.TestCase):
    """Automated headless tests for Hoá's Midnight Escape environment."""
    def test_all_levels_headless_smoke(self) -> None:
        """All registered curriculum levels must run 500 random steps headlessly with zero crashes."""
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
            self.assertIsNone(env._renderer)
            self.assertFalse(env.is_running)

    def test_reset_restores_player_ghosts_and_items(self) -> None:
        """reset() must restore Hoá, all ghosts, and all collected warmth items."""
        level_id = next(lvl for lvl, builder in LEVEL_BUILDERS.items() if builder().ghosts and builder().items)
        env = create_env(level_id=level_id)
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

    def test_double_jump_and_recovery(self) -> None:
        """Double jump works in mid-air and ledge-slip recovery grants 1 jump."""
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[])
        env.reset()

        # 1. Ground jump unlocks air jump
        state, _, _ = env.step(Action.JUMP)
        self.assertTrue(state.can_double_jump)

        # 2. Air jump consumes ability
        state, _, _ = env.step(Action.JUMP)
        self.assertFalse(state.can_double_jump)

        # 3. Subsequent air jump does nothing extra
        state, _, _ = env.step(Action.JUMP)
        self.assertFalse(state.can_double_jump)

    def test_ledge_slip_recovery_over_pit(self) -> None:
        """Walking into a pit grants 1 air jump, allowing Hoá to jump mid-air to recover."""
        # A pit located right ahead of the starting position (x=50.0)
        pit = Pit(x_start=55.0, x_end=100.0)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], items=[])
        env.reset()

        # Step 1: Walk right off the ledge into the pit without jumping
        state, _, done = env.step(Action.RIGHT)
        self.assertFalse(done)
        self.assertFalse(env.player.is_grounded)
        # Slipped into air -> recovery jump MUST be unlocked
        self.assertTrue(state.can_double_jump)

        # Step 2: Trigger mid-air recovery jump
        state, _, done = env.step(Action.RIGHT_JUMP)
        # Ability consumed, vertical velocity boosted upward (vy < 0)
        self.assertFalse(state.can_double_jump)
        self.assertLess(env.player.vy, 0)

    def test_pit_death_after_double_jump(self) -> None:
        """Exhausting double jump over a wide pit still results in fell_in_pit death."""
        # A very wide pit that cannot be cleared
        pit = Pit(x_start=50.0, x_end=500.0)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], items=[])
        env.reset()

        # Double jump as high as possible into the pit
        env.step(Action.RIGHT_JUMP)  # First jump from ground
        env.step(Action.RIGHT_JUMP)  # Double jump in mid-air

        # Keep falling until hitting bottom of canvas
        done = False
        events = None
        for _ in range(50):
            _, events, done = env.step(Action.STAY)
            if done:
                break

        self.assertTrue(done)
        self.assertTrue(events.fell_in_pit)
        self.assertFalse(env.player.is_alive)
        self.assertFalse(env.player.can_double_jump)

    def test_crouch_and_momentum_slide(self) -> None:
        """Crouching shrinks hitbox height to 18px and moving momentum slides with friction."""
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[])
        env.reset()

        # 1. Standing still: crouching drops height to 18px and stays still (vx=0)
        state, _, _ = env.step(Action.CROUCH)
        self.assertTrue(state.is_crouching)
        self.assertEqual(env.player.height, 18.0)
        self.assertEqual(env.player.vx, 0.0)

        # 2. Release crouch: height returns to 32px
        state, _, _ = env.step(Action.STAY)
        self.assertFalse(state.is_crouching)
        self.assertEqual(env.player.height, 32.0)

        # 3. Running right then crouch: slides with decaying momentum
        env.step(Action.RIGHT)
        self.assertEqual(env.player.vx, env.WALK_SPEED)
        state, _, _ = env.step(Action.CROUCH)
        self.assertTrue(state.is_crouching)
        # Velocity should be WALK_SPEED * SLIDE_FRICTION (3.28)
        self.assertAlmostEqual(env.player.vx, env.WALK_SPEED * env.SLIDE_FRICTION, places=2)

    def test_pit_fire_phases_and_bounding_box(self) -> None:
        """Pits with fire follow dormant (0) -> warning (1) -> active flame (2) cycles."""
        pit = Pit(x_start=100.0, x_end=170.0, has_fire=True, fire_offset=0)
        floor_y = 320.0
        canvas_h = 400.0

        # Phase 0: Safe/Off (ticks 0..64)
        self.assertEqual(pit.get_fire_phase(0), 0)
        self.assertEqual(pit.get_fire_phase(64), 0)
        self.assertIsNone(pit.get_flame_rect(0, floor_y, canvas_h))

        # Phase 1: Warning embers (ticks 65..79)
        self.assertEqual(pit.get_fire_phase(65), 1)
        self.assertEqual(pit.get_fire_phase(79), 1)
        self.assertIsNone(pit.get_flame_rect(65, floor_y, canvas_h))

        # Phase 2: Active lethal flame (ticks 80..99)
        self.assertEqual(pit.get_fire_phase(80), 2)
        self.assertEqual(pit.get_fire_phase(99), 2)
        expected_rect = (100.0, floor_y - pit.FIRE_HEIGHT, 70.0, canvas_h - (floor_y - pit.FIRE_HEIGHT))
        self.assertEqual(pit.get_flame_rect(80, floor_y, canvas_h), expected_rect)

        # Cycle wraps back to 0 at tick 100
        self.assertEqual(pit.get_fire_phase(100), 0)

    def test_pit_fire_lethal_collision(self) -> None:
        """Jumping or walking into active fire pillar triggers hit_fire and ends episode."""
        pit = Pit(x_start=50.0, x_end=120.0, has_fire=True, fire_offset=0)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], items=[])
        env.reset()

        # Fast-forward to flame active phase (tick 80)
        env.step_count = 79
        state, events, done = env.step(Action.RIGHT)

        self.assertTrue(events.hit_fire)
        self.assertTrue(done)
        self.assertFalse(env.player.is_alive)
        self.assertEqual(state.pit_fire_state, 2)

    def test_pit_fire_safe_crossing_during_dormant_phase(self) -> None:
        """Moving over a pit during dormant or warning phase does NOT trigger hit_fire."""
        pit = Pit(x_start=55.0, x_end=90.0, has_fire=True, fire_offset=0)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], items=[])
        env.reset()

        # Step count is at tick 1 (dormant phase 0)
        state, events, done = env.step(Action.RIGHT)
        self.assertFalse(events.hit_fire)
        self.assertEqual(state.pit_fire_state, 0)

    def test_show_game_over_headless_safe(self) -> None:
        """Calling show_game_over() headlessly must be safe and never crash."""
        env = GameEnvironment()
        events = GameEvents(reached_goal=True)
        # Should gracefully do nothing because _renderer is None
        env.show_game_over(events)
        self.assertFalse(env.is_running)

    def test_ghost_laser_lethal_collision(self) -> None:
        """Walking or jumping into a ghost's rotating laser beam must trigger hit_laser and end episode."""
        level_id = next(lvl for lvl, builder in LEVEL_BUILDERS.items() if any(getattr(g, "has_laser", False) or hasattr(g, "get_laser_segment") for g in builder().ghosts))
        env = create_env(level_id=level_id)
        env.reset()
        laser_ghost = next(g for g in env.ghosts if getattr(g, "has_laser", False) or hasattr(g, "get_laser_segment"))

        seg = laser_ghost.get_laser_segment(env.step_count)
        self.assertIsNotNone(seg)
        (sx, sy), (ex, ey) = seg

        # Place Hoá directly along the midpoint of the laser beam
        env.player.x = (sx + ex) / 2.0 - env.player.width / 2.0
        env.player.y = (sy + ey) / 2.0 - env.player.height / 2.0

        state, events, done = env.step(Action.STAY)
        self.assertTrue(events.hit_laser)
        self.assertFalse(env.player.is_alive)
        self.assertTrue(done)

    def test_platform_obstacle_landing(self) -> None:
        """Hoá can jump onto and land on top of a Mario-style platform."""
        from env.entities import Obstacle
        platform = Obstacle(x=50.0, y=260.0, width=60.0, height=18.0, style="platform")
        env = GameEnvironment(obstacles=[platform], pits=[], ghosts=[], items=[])
        env.reset()
        # Place player above platform such that next gravity tick crosses the platform top
        env.player.x = 60.0
        env.player.y = 225.0
        env.player.vy = 4.0
        env.player.is_grounded = False

        state, _, _ = env.step(Action.STAY)
        # Should land on top of platform: y = platform.y - player.height = 260 - 32 = 228
        self.assertEqual(env.player.y, 228.0)
        self.assertTrue(env.player.is_grounded)
        self.assertEqual(env.player.vy, 0.0)

    def test_tombstone_visual_styles_rendering_headless(self) -> None:
        """GameRenderer draws all tombstone styles ('cross', 'buddhist', 'platform') headlessly without error."""
        import pygame
        from env.entities import Obstacle
        from env.game_renderer import GameRenderer

        renderer = GameRenderer()
        canvas = pygame.Surface((800, 400))
        tomb_cross = Obstacle(x=100.0, y=256.0, width=32.0, height=64.0, style="cross")
        tomb_buddhist = Obstacle(x=200.0, y=256.0, width=32.0, height=64.0, style="buddhist")
        platform = Obstacle(x=300.0, y=200.0, width=50.0, height=18.0, style="platform")

        # Must execute without exception
        renderer._draw_tombstone(canvas, tomb_cross, camera_x=0.0)
        renderer._draw_tombstone(canvas, tomb_buddhist, camera_x=0.0)
        renderer._draw_tombstone(canvas, platform, camera_x=0.0)

if __name__ == "__main__":
    unittest.main()
