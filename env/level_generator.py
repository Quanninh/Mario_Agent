"""
Procedural Level Generator for Fair Blind Testing and Curriculum Reinforcement Learning.

Key Design Principles (SOLID & Invariant Physics Bounds):
1. Single Responsibility: Dedicated exclusively to dynamic procedural level synthesis.
2. Physics-Bounded Invariants:
   - Walk speed = 4.0, Gravity = 0.8, Jump force = -13.0
   - Single jump span <= 128px, height <= 105.6px
   - Double jump span <= 188px, height <= 153.6px
   - All pits, obstacles, and run-up landing buffers strictly respect these bounds,
     guaranteeing 100% fair, solvable maps across all seeds.
3. Deterministic Seed Reproducibility:
   - Same seed yields identical geometry on any machine.
   - Different seeds yield diverse, generalized layouts for fair blind evaluation.
"""

import random
from typing import Optional, List
from .entities import (
    Obstacle, Pit, Ghost, LaserGhost, WispGhost,
    WarmthItem, GoalFlag, GraveMonster, QuyNhapTrang, CuongThi,
)
from .map_levels import LevelConfig, DEFAULT_FLOOR_Y


class LevelGenerator:
    """
    Synthesizes procedurally generated levels for Levels 1 through 7
    based on the exact game curriculum concepts.
    """

    @staticmethod
    def generate_level_1(seed: int) -> LevelConfig:
        """
        Level 1: Static tombstones only.
        Teaches basic obstacle jumping and horizontal movement timing.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []
        map_length = rng.uniform(1300.0, 1700.0)
        cursor_x = 280.0

        while cursor_x < map_length - 200.0:
            # Run-up buffer between obstacles: 140px to 250px
            cursor_x += rng.uniform(140.0, 250.0)
            h = rng.choice([36.0, 44.0, 50.0, 58.0])  # Strictly < 105.6px max single jump height
            w = rng.choice([30.0, 32.0, 36.0])
            style = rng.choice(["cross", "buddhist"])
            obstacles.append(Obstacle(x=cursor_x, y=DEFAULT_FLOOR_Y - h, width=w, height=h, style=style))

        return LevelConfig(
            name=f"Level 1 (Seed {seed}): Quiet Graveyard",
            obstacles=obstacles,
            pits=[],
            ghosts=[],
            items=[],
            goal=GoalFlag(x=map_length, y=DEFAULT_FLOOR_Y - 88.0),
            floor_y=DEFAULT_FLOOR_Y,
            max_steps=1200,
            level_id=1,
        )

    @staticmethod
    def generate_level_2(seed: int) -> LevelConfig:
        """
        Level 2: Static tombstones + Pit chasms with fire hazard.
        Teaches pit distance estimation and jump apex landing.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []
        pits: List[Pit] = []
        map_length = rng.uniform(1700.0, 2200.0)
        cursor_x = 280.0

        while cursor_x < map_length - 220.0:
            choice = rng.choice(["tomb", "pit", "tomb_then_pit"])

            if choice == "tomb":
                h = rng.uniform(38.0, 54.0)
                obstacles.append(Obstacle(x=cursor_x, y=DEFAULT_FLOOR_Y - h, width=32.0, height=h))
                cursor_x += rng.uniform(140.0, 210.0)

            elif choice == "pit":
                # Pit gap 65px - 85px (well below 128px max single jump reach)
                pit_w = rng.uniform(65.0, 85.0)
                pits.append(Pit(x_start=cursor_x, x_end=cursor_x + pit_w, has_fire=True, fire_offset=rng.randint(0, 80)))
                cursor_x += pit_w + rng.uniform(140.0, 200.0)

            elif choice == "tomb_then_pit":
                # Compound challenge: vault over tombstone, take run-up, then leap over pit
                obstacles.append(Obstacle(x=cursor_x, y=DEFAULT_FLOOR_Y - 42.0, width=32.0, height=42.0))
                cursor_x += 135.0  # Run-up runway
                pits.append(Pit(x_start=cursor_x, x_end=cursor_x + 75.0, has_fire=True, fire_offset=rng.randint(0, 80)))
                cursor_x += 75.0 + 160.0

        return LevelConfig(
            name=f"Level 2 (Seed {seed}): Open Graves Pits",
            obstacles=obstacles,
            pits=pits,
            ghosts=[],
            items=[],
            goal=GoalFlag(x=map_length, y=DEFAULT_FLOOR_Y - 88.0),
            floor_y=DEFAULT_FLOOR_Y,
            max_steps=1600,
            level_id=2,
        )

    @staticmethod
    def generate_level_3(seed: int) -> LevelConfig:
        """
        Level 3: Tombs + Pits + Warmth items (Garlic) + Regular patrol Ghosts.
        Teaches moving hazard avoidance and optional item collection rewards.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []
        pits: List[Pit] = []
        ghosts: List[Ghost] = []
        items: List[WarmthItem] = []
        map_length = rng.uniform(2000.0, 2600.0)
        cursor_x = 280.0

        while cursor_x < map_length - 240.0:
            if rng.random() < 0.45:
                # Pit hazard
                pit_w = rng.uniform(70.0, 85.0)
                pits.append(Pit(x_start=cursor_x, x_end=cursor_x + pit_w, has_fire=True, fire_offset=rng.randint(0, 80)))
                if rng.random() < 0.5:
                    items.append(WarmthItem(x=cursor_x + pit_w / 2.0 - 8.0, y=DEFAULT_FLOOR_Y - 80.0))
                cursor_x += pit_w + rng.uniform(130.0, 180.0)
            else:
                # Tombstone obstacle
                h = rng.uniform(38.0, 56.0)
                obstacles.append(Obstacle(x=cursor_x, y=DEFAULT_FLOOR_Y - h, width=32.0, height=h))
                if rng.random() < 0.6:
                    items.append(WarmthItem(x=cursor_x + 8.0, y=DEFAULT_FLOOR_Y - h - 28.0))
                cursor_x += rng.uniform(130.0, 190.0)

            # Spawn standard patrolling ghost along flat segments
            if rng.random() < 0.38 and cursor_x < map_length - 300.0:
                patrol_len = rng.uniform(80.0, 130.0)
                ghost_y = rng.uniform(DEFAULT_FLOOR_Y - 88.0, DEFAULT_FLOOR_Y - 48.0)
                ghosts.append(Ghost(
                    x=cursor_x,
                    y=ghost_y,
                    patrol_min_x=cursor_x - patrol_len / 2.0,
                    patrol_max_x=cursor_x + patrol_len / 2.0,
                    speed=rng.choice([-1.4, -1.2, 1.2, 1.4]),
                ))
                cursor_x += patrol_len + 80.0

        return LevelConfig(
            name=f"Level 3 (Seed {seed}): Whispering Graves",
            obstacles=obstacles,
            pits=pits,
            ghosts=ghosts,
            items=items,
            goal=GoalFlag(x=map_length, y=DEFAULT_FLOOR_Y - 88.0),
            floor_y=DEFAULT_FLOOR_Y,
            max_steps=2000,
            level_id=3,
        )

    @staticmethod
    def generate_level_4(seed: int) -> LevelConfig:
        """
        Level 4: Tombs + Pits + Items + Standard Ghosts + Rotating/Sweeping LaserGhosts.
        Teaches timing leaps under or around rotating laser hazards.
        """
        rng = random.Random(seed)
        # Inherit foundation from Level 3 structure
        base_cfg = LevelGenerator.generate_level_3(seed)
        ghosts = list(base_cfg.ghosts)

        # Distribute 3 to 5 LaserGhosts across mid-to-low altitude
        num_lasers = rng.randint(3, 5)
        step = (base_cfg.goal.x - 600.0) / (num_lasers + 1)
        for i in range(num_lasers):
            lx = 380.0 + (i + 1) * step + rng.uniform(-40.0, 40.0)
            is_sweeper = rng.random() < 0.5
            if is_sweeper:
                laser = LaserGhost(
                    x=lx,
                    y=rng.uniform(140.0, 180.0),
                    patrol_min_x=lx - 55.0,
                    patrol_max_x=lx + 55.0,
                    speed=rng.choice([-1.0, 1.0]),
                    laser_length=rng.uniform(70.0, 85.0),
                    laser_rotation_speed=rng.choice([0.035, -0.035]),
                    laser_initial_angle=1.57,
                    laser_sweep_amplitude=0.80,
                )
            else:
                laser = LaserGhost(
                    x=lx,
                    y=rng.uniform(140.0, 175.0),
                    patrol_min_x=lx - 50.0,
                    patrol_max_x=lx + 50.0,
                    speed=rng.choice([-1.0, 1.0]),
                    laser_length=rng.uniform(70.0, 80.0),
                    laser_rotation_speed=rng.choice([0.034, -0.034]),
                    laser_initial_angle=rng.uniform(0.0, 3.14),
                )
            ghosts.append(laser)

        return LevelConfig(
            name=f"Level 4 (Seed {seed}): Nightmare Catacombs",
            obstacles=base_cfg.obstacles,
            pits=base_cfg.pits,
            ghosts=ghosts,
            items=base_cfg.items,
            goal=base_cfg.goal,
            floor_y=base_cfg.floor_y,
            max_steps=2200,
            level_id=4,
        )

    @staticmethod
    def generate_level_5(seed: int) -> LevelConfig:
        """
        Level 5: Full combination + Multi-tier WispGhosts (sinusoidal & phase-cycling wisps).
        Teaches navigating complex air-space with phased visibility hazards.
        """
        rng = random.Random(seed)
        # Inherit foundation from Level 4 structure
        base_cfg = LevelGenerator.generate_level_4(seed)
        ghosts = list(base_cfg.ghosts)

        # Distribute 4 to 6 WispGhosts across sky tiers (High, Alternating, Low)
        num_wisps = rng.randint(4, 6)
        step = (base_cfg.goal.x - 500.0) / (num_wisps + 1)
        for i in range(num_wisps):
            wx = 320.0 + (i + 1) * step + rng.uniform(-35.0, 35.0)
            mode = rng.choice(["high", "alternating", "low"])
            wy = 55.0 if mode == "high" else (125.0 if mode == "alternating" else 180.0)
            wisp = WispGhost(
                x=wx,
                y=wy,
                patrol_min_x=wx - 65.0,
                patrol_max_x=wx + 65.0,
                speed=rng.choice([-1.3, 1.3]),
                float_amplitude=rng.uniform(5.0, 7.0),
                float_speed=0.08,
                wisp_cycle=rng.randint(120, 140),
                wisp_offset=rng.randint(0, 100),
                wisp_mode=mode,
            )
            ghosts.append(wisp)

        return LevelConfig(
            name=f"Level 5 (Seed {seed}): Valley of the Wisp Maidens",
            obstacles=base_cfg.obstacles,
            pits=base_cfg.pits,
            ghosts=ghosts,
            items=base_cfg.items,
            goal=base_cfg.goal,
            floor_y=base_cfg.floor_y,
            max_steps=2500,
            level_id=5,
        )

    @staticmethod
    def _ensure_min_ground_tombstones(
        rng: random.Random,
        obstacles: List[Obstacle],
        ground_tombs: List[Obstacle],
        pits: List[Pit],
        map_length: float,
        min_count: int,
    ) -> None:
        """Tops up distinct ground tombstones in safe gaps if procedural rolls placed too few."""
        candidate_x = 300.0
        while len(ground_tombs) < min_count and candidate_x < map_length - 220.0:
            near_pit = any((p.x_start - 90.0) <= candidate_x <= (p.x_end + 90.0) for p in pits)
            near_obs = any(abs(o.x - candidate_x) < 110.0 for o in obstacles)
            if not near_pit and not near_obs:
                h = rng.uniform(48.0, 62.0)
                tomb = Obstacle(
                    x=candidate_x,
                    y=DEFAULT_FLOOR_Y - h,
                    width=32.0,
                    height=h,
                    style=rng.choice(["cross", "buddhist"]),
                )
                obstacles.append(tomb)
                ground_tombs.append(tomb)
            candidate_x += 95.0

    @staticmethod
    def generate_level_6(seed: int) -> LevelConfig:
        """
        Level 6 (Procedural): Corpse Demon (QuyNhapTrang) grave ambushes paired with
        fire/normal pits, base Ghosts, low Wisp Maidens, and rotating LaserGhosts.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []
        ground_tombs: List[Obstacle] = []
        pits: List[Pit] = []
        ghosts: List[Ghost] = []
        items: List[WarmthItem] = []

        map_length = rng.uniform(1650.0, 2150.0)
        cursor_x = 220.0

        while cursor_x < map_length - 230.0:
            roll = rng.random()
            if roll < 0.40:
                pit_w = rng.uniform(68.0, 82.0)
                has_fire = rng.random() < 0.55
                pits.append(Pit(
                    x_start=cursor_x,
                    x_end=cursor_x + pit_w,
                    has_fire=has_fire,
                    fire_offset=rng.randint(0, 90),
                ))
                if rng.random() < 0.65:
                    items.append(WarmthItem(
                        x=cursor_x + pit_w / 2.0 - 8.0,
                        y=DEFAULT_FLOOR_Y - rng.uniform(82.0, 90.0),
                    ))
                cursor_x += pit_w + rng.uniform(140.0, 200.0)
            else:
                h = rng.uniform(46.0, 60.0)
                style = rng.choice(["buddhist", "cross"])
                tomb = Obstacle(x=cursor_x, y=DEFAULT_FLOOR_Y - h, width=32.0, height=h, style=style)
                obstacles.append(tomb)
                ground_tombs.append(tomb)
                if rng.random() < 0.45:
                    items.append(WarmthItem(x=cursor_x + 8.0, y=DEFAULT_FLOOR_Y - h - 38.0))
                cursor_x += rng.uniform(145.0, 220.0)

            if cursor_x < map_length - 260.0 and rng.random() < 0.45:
                patrol_len = rng.uniform(90.0, 140.0)
                gx = cursor_x + patrol_len * 0.4
                g_kind = rng.choice(["base_low", "base_high", "wisp_low", "laser"])
                if g_kind in ("base_low", "base_high"):
                    gy = DEFAULT_FLOOR_Y - (rng.uniform(38.0, 44.0) if g_kind == "base_low" else rng.uniform(84.0, 90.0))
                    ghosts.append(Ghost(
                        x=gx, y=gy,
                        patrol_min_x=gx - patrol_len / 2.0,
                        patrol_max_x=gx + patrol_len / 2.0,
                        speed=rng.choice([-1.4, -1.3, 1.3, 1.4]),
                        float_amplitude=rng.uniform(6.0, 8.0),
                        float_speed=0.08,
                    ))
                    if g_kind == "base_high" and rng.random() < 0.6:
                        items.append(WarmthItem(x=gx, y=DEFAULT_FLOOR_Y - 28.0))
                elif g_kind == "wisp_low":
                    ghosts.append(WispGhost(
                        x=gx, y=DEFAULT_FLOOR_Y - rng.uniform(72.0, 80.0),
                        patrol_min_x=gx - patrol_len / 2.0,
                        patrol_max_x=gx + patrol_len / 2.0,
                        speed=rng.choice([-1.3, 1.3]),
                        float_amplitude=8.0,
                        float_speed=0.08,
                        wisp_cycle=rng.randint(130, 150),
                        wisp_offset=rng.randint(0, 90),
                        wisp_mode=rng.choice(["low", "alternating"]),
                    ))
                else:
                    ghosts.append(LaserGhost(
                        x=gx, y=DEFAULT_FLOOR_Y - rng.uniform(82.0, 96.0),
                        patrol_min_x=gx - patrol_len / 2.0,
                        patrol_max_x=gx + patrol_len / 2.0,
                        speed=rng.choice([-1.3, 1.3]),
                        float_amplitude=7.0,
                        float_speed=0.08,
                        laser_length=rng.uniform(76.0, 85.0),
                        laser_rotation_speed=rng.choice([0.036, -0.036, 0.038]),
                        laser_initial_angle=rng.uniform(0.0, 3.14),
                    ))
                cursor_x += patrol_len + 60.0

        # Ensure essential Level 6 signature hazards exist on every seed
        if not pits:
            px = map_length * 0.45
            pits.append(Pit(x_start=px, x_end=px + 75.0, has_fire=True, fire_offset=rng.randint(0, 60)))
        if not any(type(g) is Ghost for g in ghosts):
            gx = rng.uniform(260.0, 360.0)
            ghosts.append(Ghost(x=gx, y=DEFAULT_FLOOR_Y - 40.0, patrol_min_x=gx - 40.0, patrol_max_x=gx + 40.0, speed=1.4, float_amplitude=7.0, float_speed=0.08))
        if not any(isinstance(g, LaserGhost) for g in ghosts):
            lx = map_length * 0.58
            ghosts.append(LaserGhost(x=lx, y=DEFAULT_FLOOR_Y - 86.0, patrol_min_x=lx - 50.0, patrol_max_x=lx + 50.0, speed=1.3, float_amplitude=8.0, float_speed=0.08, laser_length=82.0, laser_rotation_speed=0.038, laser_initial_angle=0.0))
        if not any(isinstance(g, WispGhost) for g in ghosts):
            wx = map_length * 0.35
            ghosts.append(WispGhost(x=wx, y=DEFAULT_FLOOR_Y - 76.0, patrol_min_x=wx - 45.0, patrol_max_x=wx + 45.0, speed=-1.3, float_amplitude=8.0, float_speed=0.08, wisp_cycle=140, wisp_offset=30, wisp_mode="low"))

        LevelGenerator._ensure_min_ground_tombstones(rng, obstacles, ground_tombs, pits, map_length, min_count=3)
        obstacles.sort(key=lambda o: o.x)
        ground_tombs.sort(key=lambda o: o.x)

        # Spawn 2 to 3 QuyNhapTrang ambushes anchored cleanly 1-to-1 to distinct ground tombstones
        eligible = [t for t in ground_tombs if t.x >= 340.0] or ground_tombs
        num_monsters = min(len(eligible), rng.randint(2, 3))
        ambush_tombs = sorted(rng.sample(eligible, k=num_monsters), key=lambda t: t.x)
        monsters: List[GraveMonster] = [
            QuyNhapTrang(
                x=tomb.x,
                y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=rng.uniform(165.0, 185.0),
                emerge_speed=rng.uniform(1.5, 1.8),
                base_speed=rng.uniform(1.4, 1.6),
                max_speed=rng.uniform(4.7, 5.2),
                acceleration=rng.uniform(0.04, 0.05),
            )
            for tomb in ambush_tombs
        ]

        if len(items) < 3:
            items.append(WarmthItem(x=map_length - 120.0, y=DEFAULT_FLOOR_Y - 28.0))

        return LevelConfig(
            name=f"Level 6 (Seed {seed}): Awakening of the Corpse Demon",
            obstacles=obstacles,
            pits=pits,
            ghosts=ghosts,
            monsters=monsters,
            items=items,
            goal=GoalFlag(x=map_length, y=DEFAULT_FLOOR_Y - 88.0),
            floor_y=DEFAULT_FLOOR_Y,
            max_steps=1600,
            level_id=6,
        )

    @staticmethod
    def generate_level_7(seed: int) -> LevelConfig:
        """
        Level 7 (Procedural): Master challenge combining CuongThi and QuyNhapTrang
        grave ambushes (1 monster per ground tombstone), floating stone platforms,
        hellfire chasms, counter-rotating laser scissors, and multi-tier wisps.
        """
        rng = random.Random(seed)
        obstacles: List[Obstacle] = []
        ground_tombs: List[Obstacle] = []
        pits: List[Pit] = []
        ghosts: List[Ghost] = []
        items: List[WarmthItem] = []

        map_length = rng.uniform(2650.0, 3200.0)
        cursor_x = 240.0

        while cursor_x < map_length - 260.0:
            segment = rng.choice(["tomb", "pit", "platform_pit", "tomb_laser"])

            if segment == "tomb":
                h = rng.uniform(48.0, 66.0)
                tomb = Obstacle(
                    x=cursor_x,
                    y=DEFAULT_FLOOR_Y - h,
                    width=32.0,
                    height=h,
                    style=rng.choice(["buddhist", "cross"]),
                )
                obstacles.append(tomb)
                ground_tombs.append(tomb)
                if rng.random() < 0.6:
                    items.append(WarmthItem(x=cursor_x + 8.0, y=DEFAULT_FLOOR_Y - h - 38.0))
                cursor_x += rng.uniform(145.0, 210.0)

            elif segment == "pit":
                pit_w = rng.uniform(70.0, 85.0)
                has_fire = rng.random() < 0.70
                pits.append(Pit(
                    x_start=cursor_x,
                    x_end=cursor_x + pit_w,
                    has_fire=has_fire,
                    fire_offset=rng.randint(0, 90),
                ))
                if rng.random() < 0.55:
                    items.append(WarmthItem(x=cursor_x + pit_w / 2.0 - 8.0, y=DEFAULT_FLOOR_Y - 86.0))
                if rng.random() < 0.55:
                    ghosts.append(WispGhost(
                        x=cursor_x + pit_w * 0.5,
                        y=DEFAULT_FLOOR_Y - rng.uniform(122.0, 134.0),
                        patrol_min_x=cursor_x,
                        patrol_max_x=cursor_x + pit_w + 50.0,
                        speed=rng.choice([-1.3, 1.3]),
                        float_amplitude=8.0,
                        float_speed=0.08,
                        wisp_cycle=rng.randint(125, 140),
                        wisp_offset=rng.randint(0, 80),
                        wisp_mode=rng.choice(["high", "alternating"]),
                    ))
                cursor_x += pit_w + rng.uniform(145.0, 200.0)

            elif segment == "platform_pit":
                pit_w = rng.uniform(72.0, 84.0)
                pits.append(Pit(
                    x_start=cursor_x,
                    x_end=cursor_x + pit_w,
                    has_fire=True,
                    fire_offset=rng.randint(0, 90),
                ))
                plat_x = cursor_x + pit_w + rng.uniform(45.0, 75.0)
                plat_h_off = rng.uniform(70.0, 76.0)
                plat_y = DEFAULT_FLOOR_Y - plat_h_off
                obstacles.append(Obstacle(x=plat_x, y=plat_y, width=48.0, height=16.0, style="platform"))
                if rng.random() < 0.75:
                    items.append(WarmthItem(x=plat_x + 16.0, y=plat_y - 38.0))
                cursor_x = plat_x + 48.0 + rng.uniform(135.0, 185.0)

            else:  # "tomb_laser"
                h = rng.uniform(50.0, 64.0)
                tomb = Obstacle(
                    x=cursor_x,
                    y=DEFAULT_FLOOR_Y - h,
                    width=32.0,
                    height=h,
                    style=rng.choice(["cross", "buddhist"]),
                )
                obstacles.append(tomb)
                ground_tombs.append(tomb)
                lx = cursor_x + rng.uniform(115.0, 155.0)
                if lx < map_length - 220.0:
                    ghosts.append(LaserGhost(
                        x=lx,
                        y=DEFAULT_FLOOR_Y - rng.uniform(108.0, 118.0),
                        patrol_min_x=lx - 45.0,
                        patrol_max_x=lx + 45.0,
                        speed=rng.choice([-1.2, 1.1, 1.2]),
                        float_amplitude=6.5,
                        float_speed=0.07,
                        laser_length=rng.uniform(80.0, 88.0),
                        laser_rotation_speed=rng.choice([0.035, -0.035, 0.038]),
                        laser_initial_angle=rng.uniform(0.0, 3.14),
                    ))
                cursor_x += rng.uniform(220.0, 280.0)

            if cursor_x < map_length - 250.0 and rng.random() < 0.45:
                gx = cursor_x
                gy = DEFAULT_FLOOR_Y - rng.choice([40.0, 42.0, 86.0])
                ghosts.append(Ghost(
                    x=gx, y=gy,
                    patrol_min_x=gx - 55.0,
                    patrol_max_x=gx + 55.0,
                    speed=rng.choice([-1.5, -1.3, 1.3, 1.5]),
                    float_amplitude=rng.uniform(7.0, 9.0),
                    float_speed=0.08,
                ))
                cursor_x += 90.0

        # Ensure essential Level 7 entity diversity on every seed
        if not any(o.style == "platform" for o in obstacles):
            plat_x = map_length * 0.45
            obstacles.append(Obstacle(x=plat_x, y=DEFAULT_FLOOR_Y - 72.0, width=48.0, height=16.0, style="platform"))
            items.append(WarmthItem(x=plat_x + 16.0, y=DEFAULT_FLOOR_Y - 110.0))

        while len(pits) < 5:
            px = 420.0 + len(pits) * 390.0
            near_obs = any(abs(o.x - px) < 95.0 for o in obstacles)
            near_pit = any((p.x_start - 110.0) <= px <= (p.x_end + 110.0) for p in pits)
            if px < map_length - 240.0 and not near_obs and not near_pit:
                pits.append(Pit(x_start=px, x_end=px + 75.0, has_fire=(len(pits) % 2 == 1), fire_offset=rng.randint(0, 90)))
            else:
                break

        if not any(p.has_fire for p in pits):
            if pits:
                pits[0].has_fire = True
            else:
                pits.append(Pit(x_start=500.0, x_end=580.0, has_fire=True, fire_offset=20))

        # Guarantee >= 9 ghosts covering LaserGhost + WispGhost + base Ghost
        if not any(isinstance(g, LaserGhost) for g in ghosts):
            lx = map_length * 0.50
            ghosts.append(LaserGhost(x=lx, y=DEFAULT_FLOOR_Y - 115.0, patrol_min_x=lx - 40.0, patrol_max_x=lx + 40.0, speed=1.1, float_amplitude=6.0, float_speed=0.06, laser_length=85.0, laser_rotation_speed=0.035, laser_initial_angle=0.0))
        if not any(isinstance(g, WispGhost) for g in ghosts):
            wx = map_length * 0.62
            ghosts.append(WispGhost(x=wx, y=DEFAULT_FLOOR_Y - 128.0, patrol_min_x=wx - 45.0, patrol_max_x=wx + 45.0, speed=-1.3, float_amplitude=8.0, float_speed=0.08, wisp_cycle=130, wisp_offset=20, wisp_mode="alternating"))
        while len(ghosts) < 9:
            gx = 310.0 + ((len(ghosts) * 230.0) % max(400.0, map_length - 500.0))
            ghosts.append(Ghost(
                x=gx,
                y=DEFAULT_FLOOR_Y - rng.choice([40.0, 86.0]),
                patrol_min_x=gx - 50.0,
                patrol_max_x=gx + 50.0,
                speed=rng.choice([-1.5, 1.5]),
                float_amplitude=7.5,
                float_speed=0.08,
            ))

        LevelGenerator._ensure_min_ground_tombstones(rng, obstacles, ground_tombs, pits, map_length, min_count=6)
        obstacles.sort(key=lambda o: o.x)
        ground_tombs.sort(key=lambda o: o.x)

        # Spawn 4 monsters (2 CuongThi + 2 QuyNhapTrang), each anchored to a distinct ground tombstone
        eligible = [t for t in ground_tombs if t.x >= 350.0] or ground_tombs
        num_monsters = min(len(eligible), 4)
        ambush_tombs = sorted(rng.sample(eligible, k=num_monsters), key=lambda t: t.x)
        monsters: List[GraveMonster] = []
        for idx, tomb in enumerate(ambush_tombs):
            if idx % 2 == 0:
                monsters.append(CuongThi(
                    x=tomb.x,
                    y=DEFAULT_FLOOR_Y - 32.0,
                    trigger_dist=rng.uniform(180.0, 200.0),
                    hop_speed_x=rng.uniform(2.4, 2.8),
                    hop_force_y=rng.uniform(-5.8, -5.5),
                    pause_duration=rng.randint(13, 17),
                ))
            else:
                monsters.append(QuyNhapTrang(
                    x=tomb.x,
                    y=DEFAULT_FLOOR_Y - 32.0,
                    trigger_dist=rng.uniform(175.0, 195.0),
                    emerge_speed=rng.uniform(1.7, 1.9),
                    base_speed=rng.uniform(1.5, 1.65),
                    max_speed=rng.uniform(5.0, 5.4),
                    acceleration=rng.uniform(0.045, 0.05),
                ))

        while len(items) < 7:
            ix = 280.0 + len(items) * 290.0
            if ix < map_length - 120.0:
                items.append(WarmthItem(x=ix, y=DEFAULT_FLOOR_Y - rng.choice([28.0, 86.0])))
            else:
                break

        return LevelConfig(
            name=f"Level 7 (Seed {seed}): Cataclysm of the Undead",
            obstacles=obstacles,
            pits=pits,
            ghosts=ghosts,
            monsters=monsters,
            items=items,
            goal=GoalFlag(x=map_length, y=DEFAULT_FLOOR_Y - 88.0),
            floor_y=DEFAULT_FLOOR_Y,
            max_steps=10000,
            level_id=7,
        )

    @classmethod
    def generate(cls, level_id: int, seed: int) -> LevelConfig:
        """Dispatcher to synthesize a procedurally generated level by ID and seed."""
        generators = {
            1: cls.generate_level_1,
            2: cls.generate_level_2,
            3: cls.generate_level_3,
            4: cls.generate_level_4,
            5: cls.generate_level_5,
            6: cls.generate_level_6,
            7: cls.generate_level_7,
        }
        if level_id not in generators:
            raise ValueError(f"Procedural generation currently supported for Levels 1-7. Received level_id={level_id}")
        return generators[level_id](seed)
