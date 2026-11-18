from dataclasses import dataclass
from typing import Callable, Dict, List

from env.entities import GoalFlag, Obstacle, Pit
from env.mario_env import MarioEnvironment

DEFAULT_FLOOR_Y: float = 320.0


@dataclass
class LevelConfig:
    """Data container holding the layout of a single Mario level."""
    name: str
    obstacles: List[Obstacle]
    pits: List[Pit]
    goal: GoalFlag
    floor_y: float = DEFAULT_FLOOR_Y
    max_steps: int = 1000


def _build_level_1() -> LevelConfig:
    """Level 1 (Baseline): Flat ground with 2 simple pipes, no pits."""
    return LevelConfig(
        name="Level 1",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 44.0, width=32.0, height=44.0),
            Obstacle(x=480.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
        ],
        pits=[],
        goal=GoalFlag(x=720.0, y=DEFAULT_FLOOR_Y - 96.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=800,
    )


def _build_level_2() -> LevelConfig:
    """Level 2 (Standard): Multiple pipes + pits with camera scrolling."""
    return LevelConfig(
        name="Level 2",
        obstacles=[
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=520.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0),
            Obstacle(x=860.0, y=DEFAULT_FLOOR_Y - 68.0, width=32.0, height=68.0),
        ],
        pits=[
            Pit(x_start=340.0, x_end=410.0),  # 70px wide
            Pit(x_start=680.0, x_end=755.0),  # 75px wide
        ],
        goal=GoalFlag(x=1100.0, y=DEFAULT_FLOOR_Y - 96.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_3() -> LevelConfig:
    """Level 3 (Generalization Test): Different obstacle/pit order & spacing."""
    return LevelConfig(
        name="Level 3",
        obstacles=[
            Obstacle(x=360.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 44.0, width=32.0, height=44.0),
            Obstacle(x=930.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0),
        ],
        pits=[
            Pit(x_start=200.0, x_end=265.0),  # Pit comes FIRST (65px wide)
            Pit(x_start=750.0, x_end=825.0),  # 75px wide
        ],
        goal=GoalFlag(x=1120.0, y=DEFAULT_FLOOR_Y - 96.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
}


def get_level(level_id: int) -> LevelConfig:
    """Return a fresh LevelConfig for the requested level_id."""
    if level_id not in LEVEL_BUILDERS:
        available = list(LEVEL_BUILDERS.keys())
        raise ValueError(f"Unknown level_id={level_id}. Available levels: {available}")
    return LEVEL_BUILDERS[level_id]()


def create_env(level_id: int = 1) -> MarioEnvironment:
    """Build a MarioEnvironment configured with the chosen level."""
    cfg_level = get_level(level_id)
    return MarioEnvironment(
        obstacles=cfg_level.obstacles,
        pits=cfg_level.pits,
        goal=cfg_level.goal,
        floor_y=cfg_level.floor_y,
        max_steps=cfg_level.max_steps,
    )
