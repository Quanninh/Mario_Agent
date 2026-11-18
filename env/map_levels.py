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
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 44.0, width=32.0, height=44.0),
            Obstacle(x=480.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
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
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=560.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
            Obstacle(x=920.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),  # 70px wide
            Pit(x_start=760.0, x_end=830.0),  # 70px wide
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
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0),
            Obstacle(x=1000.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0),
            Obstacle(x=1320.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0),
        ],
        pits=[
            Pit(x_start=460.0, x_end=535.0),  # 75px wide
            Pit(x_start=840.0, x_end=915.0),  # 75px wide
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


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
    4: _build_level_4,
    5: _build_level_5,
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
    )
