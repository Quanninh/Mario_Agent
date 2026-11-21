from dataclasses import dataclass, field
from typing import Callable, Dict, List

from env.entities import Ghost, GoalFlag, Obstacle, Pit, WarmthItem
from env.game_env import GameEnvironment

DEFAULT_FLOOR_Y: float = 320.0


@dataclass
class LevelConfig:
    """Data container holding the layout of a single graveyard level."""
    name: str
    obstacles: List[Obstacle]
    pits: List[Pit]
    goal: GoalFlag
    ghosts: List[Ghost] = field(default_factory=list)
    items: List[WarmthItem] = field(default_factory=list)
    floor_y: float = DEFAULT_FLOOR_Y
    max_steps: int = 1000


def _build_level_1() -> LevelConfig:
    """Level 1 (Baseline): Flat ground with 2 tombstones, no pits, no ghosts, no items."""
    return LevelConfig(
        name="Level 1: Quiet Graveyard",
        obstacles=[
            Obstacle(x=250.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),
            Obstacle(x=500.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="buddhist"),
        ],
        pits=[],
        goal=GoalFlag(x=720.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=800,
    )


def _build_level_2() -> LevelConfig:
    """Level 2 (Obstacles & Pits): 3 tombstones + 2 grave pits, no ghosts, no items."""
    return LevelConfig(
        name="Level 2: Open Graves Pits",
        obstacles=[
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=520.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0),
            Obstacle(x=860.0, y=DEFAULT_FLOOR_Y - 68.0, width=32.0, height=68.0),
        ],
        pits=[
            Pit(x_start=340.0, x_end=410.0),  # 70px wide
            Pit(x_start=680.0, x_end=755.0),  # 75px wide
        ],
        goal=GoalFlag(x=1100.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_3() -> LevelConfig:
    """Level 3 (Generalization + Items): Shifted tombstones/pits + warmth items introduced."""
    return LevelConfig(
        name="Level 3: Cold Trail",
        obstacles=[
            Obstacle(x=360.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 44.0, width=32.0, height=44.0),
            Obstacle(x=930.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0),
        ],
        pits=[
            Pit(x_start=200.0, x_end=265.0),  # Pit comes FIRST (65px wide)
            Pit(x_start=750.0, x_end=825.0),  # 75px wide
        ],
        items=[
            WarmthItem(x=225.0, y=DEFAULT_FLOOR_Y - 86.0),   # Mid-air over pit 1
            WarmthItem(x=490.0, y=DEFAULT_FLOOR_Y - 28.0),   # Along the safe path
            WarmthItem(x=939.0, y=DEFAULT_FLOOR_Y - 100.0),  # Above tombstone 3
        ],
        goal=GoalFlag(x=1120.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_4() -> LevelConfig:
    """Level 4 (Uniform Ghosts): 2 ghosts sharing the exact same default movement parameters."""
    return LevelConfig(
        name="Level 4: Whispering Graves",
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


def _build_level_5() -> LevelConfig:
    """Level 5 (Diverse Ghosts): 4 ghosts with distinct speeds, bobbing heights, and wave speeds."""
    return LevelConfig(
        name="Level 5: Midnight Escape",
                obstacles=[
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="buddhist"),
            Obstacle(x=1000.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),
            Obstacle(x=1320.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="buddhist"),
        ],
        pits=[
            Pit(x_start=460.0, x_end=535.0, has_fire=True, fire_offset=0),   # 75px wide (erupts on ticks 80..99)
            Pit(x_start=840.0, x_end=915.0, has_fire=True, fire_offset=50),  # 75px wide (alternates: erupts on ticks 30..49)
        ],
        ghosts=[
            # 1. Slow eerie drifter (low -> jump OVER)
            Ghost(
                x=320.0, y=DEFAULT_FLOOR_Y - 38.0,
                patrol_min_x=280.0, patrol_max_x=435.0,
                speed=1.0, float_amplitude=6.0, float_speed=0.05,
            ),
            # 2. Fast twitchy ghost (high -> walk UNDER)
            Ghost(
                x=720.0, y=DEFAULT_FLOOR_Y - 82.0,
                patrol_min_x=675.0, patrol_max_x=820.0,
                speed=-2.4, float_amplitude=4.0, float_speed=0.14,
            ),
            # 3. Swooping poltergeist (high with wide vertical wave)
            Ghost(
                x=1120.0, y=DEFAULT_FLOOR_Y - 86.0,
                patrol_min_x=1060.0, patrol_max_x=1285.0,
                speed=1.7, float_amplitude=13.0, float_speed=0.10,
                has_laser=True, laser_length=95.0, laser_rotation_speed=0.04,
            ),
            # 4. Brisk final guardian before Hoá's house (low -> jump OVER)
            Ghost(
                x=1400.0, y=DEFAULT_FLOOR_Y - 40.0,
                patrol_min_x=1375.0, patrol_max_x=1495.0,
                speed=-1.9, float_amplitude=9.0, float_speed=0.12,
            ),
        ],
        items=[
            WarmthItem(x=229.0, y=DEFAULT_FLOOR_Y - 86.0),   # Above tombstone 1
            WarmthItem(x=490.0, y=DEFAULT_FLOOR_Y - 88.0),   # Mid-air over pit 1
            WarmthItem(x=740.0, y=DEFAULT_FLOOR_Y - 28.0),   # Under fast high ghost 2
            WarmthItem(x=1329.0, y=DEFAULT_FLOOR_Y - 102.0), # Above tombstone 4
        ],
        goal=GoalFlag(x=1520.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )

def _build_level_6() -> LevelConfig:
    """Level 6 (The Nightmare Catacombs): Extreme gauntlet with crossfire lasers, cascading flame pits, and twitchy ghosts."""
    return LevelConfig(
        name="Level 6: Nightmare Catacombs",
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


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
    4: _build_level_4,
    5: _build_level_5,
    6: _build_level_6,
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
        ghosts=cfg_level.ghosts,
        items=cfg_level.items,
        goal=cfg_level.goal,
        floor_y=cfg_level.floor_y,
        max_steps=cfg_level.max_steps,
        level_name=cfg_level.name,
    )
