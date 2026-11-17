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
    velocity_x: float        # Current horizontal velocity
    velocity_y: float        # Current vertical velocity (gravity / jump)
    is_grounded: bool        # True if Mario is standing on floor or platform
    
    # Proximity features (what Mario "sees" ahead)
    dist_to_obstacle: float  # Distance to nearest pipe/block in front
    obstacle_height: float   # Height of the obstacle ahead
    dist_to_pit: float       # Distance to nearest pit/hole ahead
    dist_to_goal: float      # Distance remaining to finish line

