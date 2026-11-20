from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class Player:
    """Represents Hoá (the player character)."""
    x: float                    # Horizontal coordinates on the 2D grid
    y: float                    # Vertical coordinates on the 2D grid
    width: float = 24.0         # Hitbox (24 pixels) used for collision detection 
    height: float = 32.0        # Hitbox (32 pixels) used for collision detection 
    vy: float = 0.0             # Vertical Velocity: handle airborne physics
    is_grounded: bool = True    # True only when Mario is resting on solid ground or a platform
    is_alive: bool = True       # Set to False if Mario falls into a pit or hits a lethal hazard
    facing_right: bool = True   # Direction Hoá is facing (for rendering eyes)

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset Hoá to starting position."""
        self.x = start_x
        self.y = start_y
        self.vy = 0.0
        self.is_grounded = True
        self.is_alive = True
        self.facing_right = True

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
    """Represents a tall tombstone blocking Hoá's path."""
    x: float
    y: float
    width: float = 32.0
    height: float = 48.0
    style: str = "buddhist"  # "buddhist" (classic arch + chữ Vạn) or "cross" (cross-shaped tombstone)

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
    """Represents an open grave pit between x_start and x_end."""
    x_start: float
    x_end: float

    def contains(self, x: float, width: float) -> bool:
        """Checks if an entity is over the pit (meaning no floor beneath it)."""
        # Fully or mostly stepped into the pit
        center_x = x + (width / 2.0)
        return self.x_start < center_x < self.x_end

@dataclass
class Ghost:
    """Represents a hovering ghost patrolling back and forth in mid-air."""
    x: float                    # Current horizontal position
    y: float                    # Current vertical position (updates with floating wave)
    patrol_min_x: float         # Left boundary of the ghost's patrol zone
    patrol_max_x: float         # Right boundary of the ghost's patrol zone
    speed: float = 1.5          # Horizontal patrol speed (positive = right, negative = left)
    width: float = 24.0         # Hitbox width (pixels)
    height: float = 28.0        # Hitbox height (pixels)
    float_amplitude: float = 8.0   # Max vertical distance (pixels) the ghost bobs up and down
    float_speed: float = 0.08      # Speed of the sinusoidal floating wave (radians per tick)

    def __post_init__(self) -> None:
        self._start_x = self.x
        self._base_y = self.y
        self._start_speed = self.speed
        self._phase = 0.0

    def update(self) -> None:
        """Patrol horizontally and hover up and down smoothly."""
        self.x += self.speed
        if self.x <= self.patrol_min_x:
            self.x = self.patrol_min_x
            self.speed = abs(self.speed)
        elif self.right >= self.patrol_max_x:
            self.x = self.patrol_max_x - self.width
            self.speed = -abs(self.speed)

        self._phase = (self._phase + self.float_speed) % (2 * math.pi)
        self.y = self._base_y + self.float_amplitude * math.sin(self._phase)

    def reset(self) -> None:
        """Restore ghost to its starting position and patrol direction."""
        self.x = self._start_x
        self.y = self._base_y
        self.speed = self._start_speed
        self._phase = 0.0

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
class WarmthItem:
    """Represents a glowing warmth collectible along the path home."""
    x: float
    y: float
    width: float = 14.0
    height: float = 14.0
    is_collected: bool = False

    def reset(self) -> None:
        """Make the item available again for the next episode."""
        self.is_collected = False

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
class GoalFlag:
    """Represents the finish line."""
    x: float
    y: float
    width: float = 64.0
    height: float = 88.0

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)
    
    def is_reached(self, player: Player) -> bool:
        """True if Mario has reached or crossed the goal line."""
        return player.right >= self.x