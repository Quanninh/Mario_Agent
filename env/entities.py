from dataclasses import dataclass
from typing import Tuple

@dataclass
class Mario:
    """Represents the player character."""
    x: float                    # Horizontal coordinates on the 2D grid
    y: float                    # Vertical coordinates on the 2D grid
    width: float = 24.0         # Hitbox (24 pixels) used for collision detection 
    height: float = 32.0        # Hitbox (32 pixels) used for collision detection 
    vy: float = 0.0             # Vertical Velocity: handle airborne physics
    is_grounded: bool = True    # True only when Mario is resting on solid ground or a platform
    is_alive: bool = True       # Set to False if Mario falls into a pit or hits a lethal hazard

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset Mario to starting position."""
        self.x = start_x
        self.y = start_y
        self.vy = 0.0
        self.is_grounded = True
        self.is_alive = True

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        """Returns bounding box (x, y, width, height) for collision detection."""
        return (self.x, self.y, self.width, self.height)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

@dataclass
class Obstacle:
    """Represents a solid pipe or block on the ground."""
    x: float
    y: float
    width: float = 32.0
    height: float = 48.0

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)
    @property
    def right(self) -> float:
        return self.x + self.width
    @property
    def bottom(self) -> float:
        return self.y + self.height

@dataclass
class Pit:
    x_start: float
    x_end: float

    def contains(self, x: float, width: float) -> bool:
        """Checks if an entity is over the pit (meaning no floor beneath it)."""
        # Fully or mostly stepped into the pit
        center_x = x + (width / 2.0)
        return self.x_start < center_x < self.x_end

@dataclass
class GoalFlag:
    """Represents the finish line."""
    x: float
    y: float
    width: float = 16.0
    height: float = 96.0

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)
    
    def is_reached(self, mario: Mario) -> bool:
        """True if Mario has reached or crossed the goal line."""
        return mario.right >= self.x