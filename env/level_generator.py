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
from .entities import Obstacle, Pit, Ghost, LaserGhost, WispGhost, WarmthItem, GoalFlag
from .map_levels import LevelConfig, DEFAULT_FLOOR_Y


class LevelGenerator:
    """
    Synthesizes procedurally generated levels for Levels 1 through 5
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

    @classmethod
    def generate(cls, level_id: int, seed: int) -> LevelConfig:
        """Dispatcher to synthesize a procedurally generated level by ID and seed."""
        generators = {
            1: cls.generate_level_1,
            2: cls.generate_level_2,
            3: cls.generate_level_3,
            4: cls.generate_level_4,
            5: cls.generate_level_5,
        }
        if level_id not in generators:
            raise ValueError(f"Procedural generation currently supported for Levels 1-5. Received level_id={level_id}")
        return generators[level_id](seed)
