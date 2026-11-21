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
    CROUCH = 5

    NUM_ACTIONS = 6

@dataclass
class RawState:
    """Continuous physical state of the game world."""
    # Player Kinematics & Posture
    player_x: float          # Hoá's current horizontal position
    player_y: float          # Hoá's current vertical position
    velocity_x: float        # Current horizontal velocity (momentum / slide)
    velocity_y: float        # Current vertical velocity (gravity / jump)
    is_grounded: bool        # True if standing on ground or tombstone
    can_double_jump: bool    # True ONLY when in mid-air with second jump available
    is_crouching: bool       # True when Hoá is ducking or sliding with low hitbox

    # Proximity features (what Hoá "sees" ahead)
    dist_to_obstacle: float  # Distance to nearest tombstone ahead (-1 if none)
    obstacle_height: float   # Height of the nearest tombstone ahead
    dist_to_pit: float       # Distance to nearest grave pit ahead (-1 if none)
    dist_to_ghost: float     # Distance to nearest hovering ghost ahead (-1 if none)
    ghost_y: float           # Y-position of that ghost (0 if none; low vs high)
    dist_to_item: float      # Distance to nearest warmth item ahead (-1 if none)
    dist_to_goal: float      # Distance remaining to Hoá's home

    

@dataclass
class GameEvents:
    """Events occurring in the current step (used by RewardStrategy)."""
    dx: float = 0.0               # Horizontal distance traveled this tick
    reached_goal: bool = False    # True if Hoá reached home [WIN]
    fell_in_pit: bool = False     # True if Hoá fell into a grave pit [DIE]
    hit_obstacle: bool = False    # True if Hoá bumped into a tombstone side
    hit_ghost: bool = False       # True if Hoá touched a floating ghost [DIE]
    collected_item: bool = False  # True if Hoá collected a warmth item [BONUS]

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
