import random
import unittest

from core import Action, GameEvents
from env import GameEnvironment, Ghost, LaserGhost, WispGhost, GoalFlag, LEVEL_BUILDERS, Pit, WarmthItem, create_env

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
        # Velocity should match initial boosted slide speed
        self.assertAlmostEqual(env.player.vx, env.SLIDE_INITIAL_SPEED, places=2)

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
        level_id = next(lvl for lvl, builder in LEVEL_BUILDERS.items() if any(isinstance(g, LaserGhost) for g in builder().ghosts))
        env = create_env(level_id=level_id)
        env.reset()
        laser_ghost = next(g for g in env.ghosts if isinstance(g, LaserGhost))

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

    def test_level_7_elements_and_single_monster_per_grave(self) -> None:
        """Level 7 combines all entities, exceeds Level 5 difficulty, and ensures 1 monster per grave."""
        from env.entities import CuongThi, QuyNhapTrang, LaserGhost, WispGhost, Obstacle
        cfg7 = LEVEL_BUILDERS[7]()
        self.assertEqual(cfg7.name, "Level 7: Cataclysm of the Undead")
        self.assertGreater(len(cfg7.obstacles), 5)
        self.assertGreater(len(cfg7.pits), 4)
        self.assertGreater(len(cfg7.ghosts), 8)
        self.assertGreater(len(cfg7.items), 6)
        self.assertIsNotNone(cfg7.monsters)
        self.assertEqual(len(cfg7.monsters), 4)

        # Confirm all entity types are present
        has_cuong_thi = any(isinstance(m, CuongThi) for m in cfg7.monsters)
        has_quy_nhap_trang = any(isinstance(m, QuyNhapTrang) for m in cfg7.monsters)
        has_laser = any(isinstance(g, LaserGhost) for g in cfg7.ghosts)
        has_wisp = any(isinstance(g, WispGhost) for g in cfg7.ghosts)
        has_fire_pit = any(p.has_fire for p in cfg7.pits)
        has_platform = any(o.style == "platform" for o in cfg7.obstacles)

        self.assertTrue(has_cuong_thi)
        self.assertTrue(has_quy_nhap_trang)
        self.assertTrue(has_laser)
        self.assertTrue(has_wisp)
        self.assertTrue(has_fire_pit)
        self.assertTrue(has_platform)

        # Constraint check: Exactly one monster per tombstone obstacle
        grave_coords = [m._start_x for m in cfg7.monsters]
        self.assertEqual(len(grave_coords), len(set(grave_coords)), "Every monster must have a distinct grave coordinate!")
        obstacle_xs = [o.x for o in cfg7.obstacles]
        for gx in grave_coords:
            self.assertIn(gx, obstacle_xs, f"Ambush origin {gx} must match an existing obstacle tombstone!")

    def test_cuong_thi_hopping_and_pit_vault(self) -> None:
        """CuongThi pauses on ground, hops leftward, and automatically vaults pits."""
        from env.entities import CuongThi, Pit, GraveMonsterState
        ct = CuongThi(x=200.0, y=288.0, trigger_dist=150.0, hop_speed_x=2.5, hop_force_y=-5.5, pause_duration=5)
        pit = Pit(x_start=140.0, x_end=190.0)
        floor_y = 320.0

        # Trigger emergence
        ct.update(player_x=100.0, floor_y=floor_y, pits=[pit])
        self.assertEqual(ct.state, GraveMonsterState.EMERGING)

        # Fast forward emergence to ACTIVE
        for _ in range(50):
            ct.update(player_x=100.0, floor_y=floor_y, pits=[pit])

        self.assertEqual(ct.state, GraveMonsterState.ACTIVE)

        # Step through hops and verify vaulting
        vaulted = False
        for _ in range(60):
            ct.update(player_x=100.0, floor_y=floor_y, pits=[pit])
            if not ct._is_grounded and ct.y < floor_y - ct.height:
                vaulted = True
                break
        self.assertTrue(vaulted)

    def test_cuong_thi_renderer_all_states_headless(self) -> None:
        """GameRenderer draws CuongThi in dormant, emerging, and hopping charge states without error."""
        import pygame
        from env.entities import CuongThi, GraveMonsterState
        from env.game_renderer import GameRenderer

        renderer = GameRenderer()
        canvas = pygame.Surface((800, 400))
        ct = CuongThi(x=200.0, y=288.0)

        # 1. Dormant omen on grave
        self.assertEqual(ct.state, GraveMonsterState.DORMANT)
        renderer._draw_grave_monster(canvas, ct, camera_x=0.0, step_count=10, floor_y=320.0)

        # 2. Emerging state
        ct.state = GraveMonsterState.EMERGING
        renderer._draw_grave_monster(canvas, ct, camera_x=0.0, step_count=20, floor_y=320.0)

        # 3. Active hopping charge (airborne hop)
        ct.state = GraveMonsterState.ACTIVE
        ct._is_grounded = False
        renderer._draw_grave_monster(canvas, ct, camera_x=0.0, step_count=30, floor_y=320.0)

        # 4. Active standing pause on ground (draped talisman)
        ct._is_grounded = True
        renderer._draw_grave_monster(canvas, ct, camera_x=0.0, step_count=35, floor_y=320.0)

    def test_level_8_marathon_architecture_and_sentry_monsters(self) -> None:
        """Level 8 features 7,200px length, multiple slide tunnels, and sentry monsters on wide platforms."""
        from env.entities import CuongThi, QuyNhapTrang, LaserGhost, WispGhost
        cfg8 = LEVEL_BUILDERS[8]()
        self.assertEqual(cfg8.name, "Level 8: The Celestial Purgatory")
        self.assertGreater(cfg8.goal.x, 7000.0)
        self.assertGreaterEqual(len(cfg8.obstacles), 40)
        self.assertGreaterEqual(len(cfg8.pits), 10)
        self.assertGreaterEqual(len(cfg8.monsters), 10)
        self.assertGreaterEqual(len(cfg8.ghosts), 15)

        # Check for hanging slide tunnels (y=0.0, height=296.0)
        slide_tunnels = [obs for obs in cfg8.obstacles if obs.y == 0.0 and obs.height == 320.0 - 24.0]
        self.assertGreaterEqual(len(slide_tunnels), 4)

        # Check platform sentries
        platform_sentries = [m for m in cfg8.monsters if m.patrol_min_x is not None]
        self.assertGreaterEqual(len(platform_sentries), 3)

        # Check that platform sentry boundaries match wide platforms
        for sentry in platform_sentries:
            catwalk = sentry.patrol_max_x - sentry.patrol_min_x
            self.assertGreaterEqual(catwalk, 100.0)

    def test_god_mode_super_jump(self) -> None:
        """God Mode preserves standard jump velocity for playability testing, but allows infinite mid-air leaps."""
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[])
        env.reset()
        env.god_mode = True

        # First ground jump: strictly standard jump force (no artificial height boost)
        env.step(Action.JUMP)
        self.assertAlmostEqual(env.player.vy, env.JUMP_FORCE + env.GRAVITY, places=2)

        # In mid-air: infinite leaps allowed
        for _ in range(5):
            env.step(Action.JUMP)
            self.assertTrue(env.player.vy < 0.0)

    def test_monster_specific_death_attribution(self) -> None:
        """Killed by CuongThi sets monster_type='cuong_thi'; QuyNhapTrang sets 'quy_nhap_trang'."""
        from env.entities import CuongThi, QuyNhapTrang, GraveMonsterState
        # Test CuongThi death
        ct = CuongThi(x=100.0, y=320.0 - 32.0)
        ct.state = GraveMonsterState.ACTIVE
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[], monsters=[ct])
        env.reset()
        env.player.x = 100.0
        env.player.y = 320.0 - 32.0
        _, events, done = env.step(Action.STAY)
        self.assertTrue(done)
        self.assertTrue(events.hit_monster)
        self.assertEqual(events.monster_type, "cuong_thi")

        # Test QuyNhapTrang death
        qnt = QuyNhapTrang(x=100.0, y=320.0 - 32.0)
        qnt.state = GraveMonsterState.ACTIVE
        env2 = GameEnvironment(obstacles=[], pits=[], ghosts=[], items=[], monsters=[qnt])
        env2.reset()
        env2.player.x = 100.0
        env2.player.y = 320.0 - 32.0
        _, events2, done2 = env2.step(Action.STAY)
        self.assertTrue(done2)
        self.assertTrue(events2.hit_monster)
        self.assertEqual(events2.monster_type, "quy_nhap_trang")

    def test_procedural_generator_solvability_and_reproducibility(self) -> None:
        """Procedural generator for Levels 1-5 produces deterministic, fair, physics-compliant levels."""
        from env import LevelGenerator

        # 1. Reproducibility test: Same seed must produce identical objects
        cfg_a1 = LevelGenerator.generate(1, seed=42)
        cfg_a2 = LevelGenerator.generate(1, seed=42)
        self.assertEqual(len(cfg_a1.obstacles), len(cfg_a2.obstacles))
        for o1, o2 in zip(cfg_a1.obstacles, cfg_a2.obstacles):
            self.assertEqual((o1.x, o1.y, o1.width, o1.height), (o2.x, o2.y, o2.width, o2.height))

        # Different seed produces distinct layout
        cfg_b = LevelGenerator.generate(1, seed=99)
        self.assertNotEqual(cfg_a1.obstacles[0].x, cfg_b.obstacles[0].x)

        # 2. Physics & Invariant Fairness Checks across multiple random seeds
        for lvl in range(1, 6):
            for seed in [101, 202, 303]:
                cfg = LevelGenerator.generate(lvl, seed=seed)
                env = create_env(lvl, seed=seed)
                self.assertEqual(env.level_id, lvl)

                # Check level concepts
                if lvl == 1:
                    # Level 1: Static tombstones only, no pits, no ghosts
                    self.assertEqual(len(cfg.pits), 0)
                    self.assertEqual(len(cfg.ghosts), 0)
                    self.assertTrue(all(o.height <= 65.0 for o in cfg.obstacles))
                elif lvl == 2:
                    # Level 2: Tombs + Pits
                    self.assertGreater(len(cfg.pits), 0)
                    self.assertEqual(len(cfg.ghosts), 0)
                    # All pits must be strictly <= 85px (well below 128px jump reach)
                    self.assertTrue(all((p.x_end - p.x_start) <= 85.0 for p in cfg.pits))
                elif lvl == 3:
                    # Level 3: Tombs + Pits + Items + Standard Ghosts
                    self.assertGreater(len(cfg.ghosts), 0)
                    self.assertTrue(all(type(g) is Ghost for g in cfg.ghosts))
                elif lvl == 4:
                    # Level 4: Has LaserGhost
                    self.assertTrue(any(isinstance(g, LaserGhost) for g in cfg.ghosts))
                elif lvl == 5:
                    # Level 5: Has WispGhost
                    self.assertTrue(any(isinstance(g, WispGhost) for g in cfg.ghosts))

    def test_dqn_raw_state_bug_fixes_and_kinematics(self) -> None:
        """Verify RawState fixes (monster_is_grounded, continuous LaserGhost, obstacle_dy/width, wisp_dy/vx)."""
        import math
        from env.entities import CuongThi, GraveMonsterState, Obstacle, QuyNhapTrang, WillOWisp

        # 1. Obstacle vertical clearance (obstacle_dy) and width
        floating_obs = Obstacle(x=120.0, y=220.0, width=64.0, height=20.0)
        env = GameEnvironment(obstacles=[floating_obs], pits=[], ghosts=[], items=[])
        state = env.reset()
        self.assertAlmostEqual(state.obstacle_dy, floating_obs.y - env.player.y)
        self.assertAlmostEqual(state.obstacle_width, 64.0)

        # 2. Continuous LaserGhost tracking even when beam points straight upward (-pi/2)
        laser_ghost = LaserGhost(
            x=200.0, y=260.0, patrol_min_x=200.0, patrol_max_x=200.0,
            speed=0.0, float_amplitude=0.0,
            laser_length=90.0, laser_rotation_speed=0.05, laser_initial_angle=-math.pi / 2.0,
        )
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[laser_ghost], items=[])
        state = env.reset()
        self.assertEqual(state.dist_to_laser, -1.0)  # Beam points upward, not overlapping player Y
        self.assertGreater(state.laser_ghost_dx, 0.0)
        self.assertAlmostEqual(state.laser_angle_sin, -1.0, places=5)
        self.assertAlmostEqual(state.laser_angle_cos, 0.0, places=5)
        self.assertAlmostEqual(state.laser_angular_vel, 0.05, places=5)
        self.assertAlmostEqual(state.laser_length, 90.0)

        # 3. Monster grounded state, type, speed ratio, pause progress, and pit distance
        pit = Pit(x_start=150.0, x_end=210.0)
        ct = CuongThi(x=300.0, y=288.0, state=GraveMonsterState.ACTIVE, pause_duration=10)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], monsters=[ct], items=[])
        env.reset()
        ct._is_grounded = False
        ct.vy = -4.0
        state = env._get_raw_state()
        self.assertFalse(state.monster_is_grounded)
        self.assertEqual(state.monster_type, 2)
        self.assertAlmostEqual(state.monster_vy, -4.0)
        self.assertAlmostEqual(state.monster_dist_to_pit, 300.0 - 210.0)

        qnt = QuyNhapTrang(x=280.0, y=288.0, state=GraveMonsterState.ACTIVE)
        env = GameEnvironment(obstacles=[], pits=[pit], ghosts=[], monsters=[qnt], items=[])
        env.reset()
        qnt._is_jumping = True
        qnt._current_speed = qnt.max_speed
        state = env._get_raw_state()
        self.assertFalse(state.monster_is_grounded)
        self.assertEqual(state.monster_type, 1)
        self.assertAlmostEqual(state.monster_speed_ratio, 1.0)

        # 4. WispGhost casting phase & WillOWisp dy/vx
        wg = WispGhost(
            x=300.0, y=240.0, patrol_min_x=300.0, patrol_max_x=300.0,
            speed=0.0, float_amplitude=0.0, wisp_cycle=100, wisp_offset=20, wisp_mode="alternating",
        )
        env = GameEnvironment(obstacles=[], pits=[], ghosts=[wg], items=[])
        state = env.reset()
        self.assertAlmostEqual(state.wisp_cast_progress, 0.2)
        self.assertAlmostEqual(state.wisp_ticks_to_cast, 80.0)
        self.assertFalse(state.wisp_next_is_high)  # shot 0 at 0..99, next cast is shot 1 (low)

        wisp = WillOWisp(x=180.0, y=250.0, vx=-3.4, vy=-2.0)
        env.wisps.append(wisp)
        state = env._get_raw_state()
        self.assertAlmostEqual(state.wisp_dy, wisp.y - env.player.y)
        self.assertAlmostEqual(state.wisp_vx, -3.4)

if __name__ == "__main__":
    unittest.main()
