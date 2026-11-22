"""Curriculum level definitions for Hoá's Midnight Escape.

Encapsulates map layouts (obstacles, pits, ghosts, items, goal) and
stage-specific rules into immutable LevelConfig dataclasses.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from core import BaseEnvironment
from env.entities import Ghost, GoalFlag, Obstacle, Pit, WarmthItem
from env.game_env import GameEnvironment


DEFAULT_FLOOR_Y: float = 320.0


@dataclass
class LevelConfig:
    """Configuration bundle defining an individual curriculum stage."""
    name: str
    obstacles: List[Obstacle]
    pits: List[Pit]
    goal: GoalFlag
    ghosts: List[Ghost]
    items: List[WarmthItem]
    floor_y: float = DEFAULT_FLOOR_Y
    max_steps: int = 1000


def _build_level_1() -> LevelConfig:
    """Level 1 (Quiet Graveyard): Introduction to basic jumping over static tombstones."""
    return LevelConfig(
        name="Level 1: Quiet Graveyard",
        obstacles=[
            Obstacle(x=320.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=640.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
        ],
        pits=[],
        ghosts=[],
        items=[],
        goal=GoalFlag(x=900.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=800,
    )


def _build_level_2() -> LevelConfig:
    """Level 2 (Open Grave Pits): Introduces lethal gap hazards requiring timed leaps."""
    return LevelConfig(
        name="Level 2: Open Graves Pits",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=560.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
            Obstacle(x=880.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),  # 70px wide gap
            Pit(x_start=680.0, x_end=755.0),  # 75px wide gap
        ],
        ghosts=[],
        items=[],
        goal=GoalFlag(x=1100.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_3() -> LevelConfig:
    """Level 3 (Whispering Graves): 3 tombstones, 2 pits (including 1 hellfire), 2 uniform ghosts (jump-over & walk-under), and warmth items."""
    return LevelConfig(
        name="Level 3: Whispering Graves",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),
            Obstacle(x=560.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),
            Obstacle(x=920.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),  # 70px wide
            Pit(x_start=760.0, x_end=830.0, has_fire=True, fire_offset=0),  # 70px wide with fire
        ],
        ghosts=[
            # Ghost 1 (low -> jump OVER): default speed=1.5, float_amplitude=8.0, float_speed=0.08
            Ghost(x=640.0, y=DEFAULT_FLOOR_Y - 40.0, patrol_min_x=610.0, patrol_max_x=740.0),
            # Ghost 2 (high -> walk UNDER): same default speed=1.5, float_amplitude=8.0, float_speed=0.08
            Ghost(x=1020.0, y=DEFAULT_FLOOR_Y - 84.0, patrol_min_x=980.0, patrol_max_x=1150.0),
        ],
        items=[
            WarmthItem(x=249.0, y=DEFAULT_FLOOR_Y - 86.0),   # Above tombstone 1
            WarmthItem(x=480.0, y=DEFAULT_FLOOR_Y - 28.0),   # After pit 1
            WarmthItem(x=1050.0, y=DEFAULT_FLOOR_Y - 28.0),  # Underneath high ghost 2
        ],
        goal=GoalFlag(x=1220.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_4() -> LevelConfig:
    """Level 4 (Nightmare Catacombs): 5 tombstones, 4 hellfire pits, 5 diverse ghosts including dual counter-rotating laser scissors, fast twitcher demon, and 5 warmth items."""
    return LevelConfig(
        name="Level 4: Nightmare Catacombs",
        obstacles=[
            # Tight stone pillars and narrow landing islands between pits
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="buddhist"),
            Obstacle(x=980.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),       # Island between Pit 2 & 3
            Obstacle(x=1420.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="buddhist"),
            Obstacle(x=1850.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),
        ],
        pits=[
            # Pit 1: Hellmouth with active flame
            Pit(x_start=400.0, x_end=480.0, has_fire=True, fire_offset=0),
            # Pit 2 & 3: Cascading alternating fire trap with narrow island at x=980
            Pit(x_start=880.0, x_end=960.0, has_fire=True, fire_offset=25),
            Pit(x_start=1020.0, x_end=1100.0, has_fire=True, fire_offset=65),
            # Pit 4: Final wide pit gap
            Pit(x_start=1620.0, x_end=1710.0, has_fire=True, fire_offset=45),
        ],
        ghosts=[
            # 1. Low ceiling guard patrolling directly over Pit 1
            Ghost(
                x=390.0, y=DEFAULT_FLOOR_Y - 42.0,
                patrol_min_x=360.0, patrol_max_x=510.0,
                speed=1.4, float_amplitude=6.0, float_speed=0.08,
            ),
            # 2. Fast twitching speed demon
            Ghost(
                x=750.0, y=DEFAULT_FLOOR_Y - 80.0,
                patrol_min_x=680.0, patrol_max_x=840.0,
                speed=-3.0, float_amplitude=8.0, float_speed=0.18,
            ),
            # 3 & 4: Dual Counter-Rotating "Laser Scissors" (Crossfire)
            Ghost(
                x=1200.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1150.0, patrol_max_x=1360.0,
                speed=1.5, float_amplitude=10.0, float_speed=0.09,
                has_laser=True, laser_length=95.0, laser_rotation_speed=0.045, laser_initial_angle=0.0,
            ),
            Ghost(
                x=1320.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1220.0, patrol_max_x=1400.0,
                speed=-1.5, float_amplitude=10.0, float_speed=0.09,
                has_laser=True, laser_length=95.0, laser_rotation_speed=-0.045, laser_initial_angle=3.14,
            ),
            # 5. Final long-range sweeper guarding Khuê Văn Các
            Ghost(
                x=1800.0, y=DEFAULT_FLOOR_Y - 85.0,
                patrol_min_x=1740.0, patrol_max_x=1960.0,
                speed=1.8, float_amplitude=12.0, float_speed=0.11,
                has_laser=True, laser_length=110.0, laser_rotation_speed=0.035,
            ),
        ],
        items=[
            WarmthItem(x=229.0, y=DEFAULT_FLOOR_Y - 90.0),   # Perched high on tombstone 1
            WarmthItem(x=440.0, y=DEFAULT_FLOOR_Y - 85.0),   # Mid-air above Hellmouth pit
            WarmthItem(x=990.0, y=DEFAULT_FLOOR_Y - 86.0),   # On narrow island between fire pits
            WarmthItem(x=1260.0, y=DEFAULT_FLOOR_Y - 28.0),  # Under the laser scissors
            WarmthItem(x=1750.0, y=DEFAULT_FLOOR_Y - 28.0),  # Before final sanctuary
        ],
        goal=GoalFlag(x=2050.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1400,
    )


def _build_level_5() -> LevelConfig:
    """Level 5 (The Eternal Abyss): Marathon gauntlet with variable flame heights, multi-speed laser ghosts, varied tombs, and mandatory crouch tunnels."""
    return LevelConfig(
        name="Level 5: The Eternal Abyss",
        obstacles=[
            # Section 1: Varied tombstone heights (low step vs tall monolith)
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 36.0, width=32.0, height=36.0, style="buddhist"),    # Low step
            Obstacle(x=600.0, y=DEFAULT_FLOOR_Y - 72.0, width=32.0, height=72.0, style="cross"),       # Tall monolith
            Obstacle(x=920.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
            
            # Section 2: Stepping stones over the Great Inferno
            Obstacle(x=1300.0, y=DEFAULT_FLOOR_Y - 84.0, width=36.0, height=84.0, style="cross"),     # High sanctuary perch
            
            # Section 3: THE MANDATORY CROUCH TUNNEL (Low clearance = 20px)
            Obstacle(x=1750.0, y=DEFAULT_FLOOR_Y - 72.0, width=90.0, height=52.0, style="cross"),      # Low ceiling: MUST SLIDE!
            
            # Section 4: Deep catacomb pillars
            Obstacle(x=2220.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),
            Obstacle(x=2580.0, y=DEFAULT_FLOOR_Y - 92.0, width=32.0, height=92.0, style="cross"),     # Colossal obelisk
            Obstacle(x=2980.0, y=DEFAULT_FLOOR_Y - 44.0, width=32.0, height=44.0, style="buddhist"),
        ],
        pits=[
            # Pit 1: Low ember flame (fire_height = 40px, can be hopped over)
            Pit(x_start=380.0, x_end=450.0, has_fire=True, fire_offset=0, fire_height=40.0),
            
            # Pit 2: Deep bottomless grave (no fire, wide leap)
            Pit(x_start=760.0, x_end=845.0, has_fire=False),
            
            # Pit 3: Towering Volcanic Inferno (fire_height = 100px! Must jump from high perch at x=1300)
            Pit(x_start=1180.0, x_end=1265.0, has_fire=True, fire_offset=30, fire_height=100.0),
            Pit(x_start=1336.0, x_end=1420.0, has_fire=True, fire_offset=70, fire_height=50.0),
            
            # Pit 4 & 5: Rapid fire gauntlet
            Pit(x_start=2050.0, x_end=2130.0, has_fire=True, fire_offset=15, fire_height=65.0),
            Pit(x_start=2420.0, x_end=2500.0, has_fire=False),
            Pit(x_start=2780.0, x_end=2870.0, has_fire=True, fire_offset=50, fire_height=85.0),
        ],
        ghosts=[
            # 1. Slow, heavy ground stalker (no laser, jump over)
            Ghost(
                x=480.0, y=DEFAULT_FLOOR_Y - 36.0,
                patrol_min_x=455.0, patrol_max_x=580.0,
                speed=0.9, float_amplitude=4.0, float_speed=0.04,
            ),
            # 2. Fast-spinning rapid laser ghost (clockwise 0.06 rad/tick)
            Ghost(
                x=860.0, y=DEFAULT_FLOOR_Y - 82.0,
                patrol_min_x=835.0, patrol_max_x=980.0,
                speed=-2.0, float_amplitude=6.0, float_speed=0.12,
                has_laser=True, laser_length=80.0, laser_rotation_speed=0.06, laser_initial_angle=0.0,
            ),
            # 3. High soaring poltergeist above the Towering Inferno
            Ghost(
                x=1240.0, y=DEFAULT_FLOOR_Y - 96.0,
                patrol_min_x=1160.0, patrol_max_x=1380.0,
                speed=1.5, float_amplitude=16.0, float_speed=0.07,
            ),
            # 4. Low Tunnel Guard: hovers right before the crouch tunnel at head height
            Ghost(
                x=1680.0, y=DEFAULT_FLOOR_Y - 38.0,
                patrol_min_x=1640.0, patrol_max_x=1740.0,
                speed=-1.2, float_amplitude=4.0, float_speed=0.08,
            ),
            # 5. Slow wide-reach counter-clockwise laser ghost (laser_length = 115px, counter-clockwise)
            Ghost(
                x=2300.0, y=DEFAULT_FLOOR_Y - 86.0,
                patrol_min_x=2220.0, patrol_max_x=2440.0,
                speed=1.6, float_amplitude=10.0, float_speed=0.10,
                has_laser=True, laser_length=115.0, laser_rotation_speed=-0.028, laser_initial_angle=1.57,
            ),
            # 6. Ultra-fast erratic twitcher (speed = -3.2!)
            Ghost(
                x=2680.0, y=DEFAULT_FLOOR_Y - 80.0,
                patrol_min_x=2600.0, patrol_max_x=2780.0,
                speed=-3.2, float_amplitude=12.0, float_speed=0.20,
            ),
            # 7. Final Grand Sentinel guarding Khuê Văn Các with continuous laser
            Ghost(
                x=3100.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=3020.0, patrol_max_x=3240.0,
                speed=1.8, float_amplitude=11.0, float_speed=0.09,
                has_laser=True, laser_length=100.0, laser_rotation_speed=0.04, laser_initial_angle=3.14,
            ),
        ],
        items=[
            WarmthItem(x=249.0, y=DEFAULT_FLOOR_Y - 74.0),    # On low tombstone 1
            WarmthItem(x=620.0, y=DEFAULT_FLOOR_Y - 110.0),   # High above monolith 2
            WarmthItem(x=1310.0, y=DEFAULT_FLOOR_Y - 122.0),  # High above volcanic perch
            WarmthItem(x=1790.0, y=DEFAULT_FLOOR_Y - 14.0),   # Inside the low crouch tunnel!
            WarmthItem(x=2240.0, y=DEFAULT_FLOOR_Y - 28.0),   # Under wide laser ghost
            WarmthItem(x=2600.0, y=DEFAULT_FLOOR_Y - 130.0),  # Top of colossal obelisk
            WarmthItem(x=3050.0, y=DEFAULT_FLOOR_Y - 28.0),   # Courtyard before sanctuary
        ],
        goal=GoalFlag(x=3350.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=2400,
    )


# Alias so callers requesting Level 7 continue to resolve cleanly to The Eternal Abyss
_build_level_7 = _build_level_5


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
    4: _build_level_4,
    5: _build_level_5,
    7: _build_level_7,
}


def get_level(level_id: int) -> LevelConfig:
    """Return a fresh LevelConfig for the requested level_id."""
    if level_id not in LEVEL_BUILDERS:
        available = list(LEVEL_BUILDERS.keys())
        raise ValueError(f"Unknown level_id={level_id}. Available levels: {available}")
    return LEVEL_BUILDERS[level_id]()


def create_env(level_id: int = 1) -> GameEnvironment:
    """Build a GameEnvironment configured with the chosen level."""
    cfg_level = get_level(level_id)
    return GameEnvironment(
        obstacles=cfg_level.obstacles,
        pits=cfg_level.pits,
        goal=cfg_level.goal,
        ghosts=cfg_level.ghosts,
        items=cfg_level.items,
        floor_y=cfg_level.floor_y,
        max_steps=cfg_level.max_steps,
        level_name=cfg_level.name,
    )
