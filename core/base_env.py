from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

class Action:
    """Discrete action space for the Mario agent."""
    STAY = 0
    LEFT = 1
    JUMP = 2
    RIGHT = 3
    RIGHT_JUMP = 4

    NUM_ACTIONS = 5

@dataclass
class RawState:
    """Continuous physical state of the game world."""
    mario_x: float           # Mario's current horizontal position
    mario_y: float           # Mario's current vertical position
    velocity_y: float        # Current vertical velocity (gravity / jump)
    is_grounded: bool        # True if Mario is standing on floor or platform
    
    # Proximity features (what Mario "sees" ahead)
    dist_to_obstacle: float  # Distance to nearest pipe/block in front
    obstacle_height: float   # Height of the obstacle ahead
    dist_to_pit: float       # Distance to nearest pit/hole ahead
    dist_to_goal: float      # Distance remaining to finish line

@dataclass
class GameEvents:
    """Events occurring in the current step (used by RewardStrategy)."""
    dx: float = 0.0              # Horizontal distance traveled this tick
    reached_goal: bool = False   # True if touched the flagpole
    fell_in_pit: bool = False    # True if fell below the floor
    hit_obstacle: bool = False   # True if bumped into a block/pipe side


class BaseEnvironment(ABC):
    """Abstract interface that any Mario environment must implement."""

    @abstractmethod
    def reset(self) -> RawState:
        """Reset the game to initial state and return starting RawState."""
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]: 
        """
        Execute one action tick.
        Returns:
            - next_state (RawState): New physical measurements
            - events (GameEvents): Game occurrences this tick
            - done (bool): True if game over (death or win)
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """Draw the game on screen (Pygame window)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up Pygame window and resources."""
        pass