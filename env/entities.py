from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple, List
import math

@dataclass
class Player:
    """Represents Hoá (the player character)."""
    x: float                        # Horizontal coordinates on the 2D grid
    y: float                        # Vertical coordinates on the 2D grid
    width: float = 24.0             # Hitbox (24 pixels) used for collision detection 
    height: float = 32.0            # Hitbox (32 pixels) used for collision detection 
    vx: float = 0.0                 # Horizontal Velocity: for momentum sliding
    vy: float = 0.0                 # Vertical Velocity: handle airborne physics
    is_grounded: bool = True        # True only when Mario is resting on solid ground or a platform
    can_double_jump: bool = False   # False on ground; becomes True only while airborne
    is_crouching: bool = False
    is_alive: bool = True           # Set to False if Mario falls into a pit or hits a lethal hazard
    facing_right: bool = True       # Direction Hoá is facing (for rendering eyes)

    def reset(self, start_x: float, start_y: float) -> None:
        """Reset Hoá to starting position."""
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.height = 32.0
        self.is_crouching = False
        self.is_grounded = True
        self.can_double_jump = False
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
    has_fire: bool = False      # True if this pit shoots periodic flame jets
    fire_offset: int = 0        # Tick offset for staggered timing with other pits
    fire_height: float = 65.0

    # Flame timing constants
    FIRE_CYCLE: ClassVar[float] = 100.0
    FIRE_WARN_START: ClassVar[float] = 65.0
    FIRE_ACTIVE_START: ClassVar[float] = 80.0
    FIRE_HEIGHT: ClassVar[float] = 65.0

    def contains(self, x: float, width: float) -> bool:
        """Checks if an entity is over the pit (meaning no floor beneath it)."""
        # Fully or mostly stepped into the pit
        center_x = x + (width / 2.0)
        return self.x_start < center_x < self.x_end

    def get_fire_phase(self, step_count: int) -> int:
        """Returns 0: Safe/Off, 1: Warning embers, 2: Active lethal flame."""
        if not self.has_fire:
            return 0
        phase = (step_count + self.fire_offset) % int(self.FIRE_CYCLE)
        if phase < self.FIRE_WARN_START:
            return 0
        elif phase < self.FIRE_ACTIVE_START:
            return 1
        return 2

    def get_flame_rect(self, step_count: int, floor_y: float, canvas_height: float = 400.0) -> Optional[Tuple[float, float, float, float]]:
        """Returns (x, y, w, h) bounding box of active flame spanning pit depth, or None if safe."""
        if self.get_fire_phase(step_count) == 2:
            return (
                self.x_start,
                floor_y - self.fire_height,
                self.x_end - self.x_start,
                canvas_height - (floor_y - self.fire_height),
            )
        return None
@dataclass
class WillOWisp:
    """A spirit-flame projectile traveling in a parabolic arc that rolls on ground before fizzling."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 6.0
    gravity: float = 0.28            # Light gravity for an ethereal, floaty arc
    roll_ticks: int = 35             # Duration it rolls on the grass before dissolving
    ground_friction: float = 0.92
    is_grounded: bool = False
    is_alive: bool = True

    def update(self, pits: list, floor_y: float = 320.0, canvas_height: float = 400.0) -> None:
        """Update projectile physics: airborne parabola -> ground roll -> despawn."""
        if not self.is_grounded:
            self.x += self.vx
            self.y += self.vy
            self.vy += self.gravity

            # Check if projectile reached ground level
            if self.y + self.radius >= floor_y:
                # Check if it fell into an open pit
                in_pit = any(p.x_start < self.x < p.x_end for p in pits)
                if in_pit:
                    # Let it fall completely past the bottom of the screen before despawning
                    if self.y > canvas_height + self.radius:
                        self.is_alive = False
                else:
                    # Lands on solid ground -> transition to rolling state
                    self.y = floor_y - self.radius
                    self.vy = 0.0
                    self.is_grounded = True
        else:
            # Rolling along the grass with friction
            self.x += self.vx
            self.vx *= self.ground_friction
            self.roll_ticks -= 1
            if self.roll_ticks <= 0:
                self.is_alive = False

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        """AABB bounding box for collision detection."""
        return (self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

# -------------
# For the Ghost
# -------------

@dataclass
class Ghost:
    """Base ghost with horizontal patrol and sinusoidal vertical floating."""
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
class LaserGhost(Ghost):
    # --- Continuous 360° Rotating Laser ---
    laser_length: float = 90.0          # Beam reach in pixels
    laser_rotation_speed: float = 0.04  # Rotation speed in rad/tick (positive = clockwise)
    laser_initial_angle: float = 0.0    # Starting angle offset in radians

    def get_laser_segment(self, step_count: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns ((start_x, start_y), (end_x, end_y)) of the continuous 360-degree rotating laser beam."""
        # Originates from the center of the ghost's eyes
        eye_cx = self.x + self.width / 2.0
        eye_cy = self.y + self.height * 0.4

        # Continuously rotates 360 degrees based on step count
        angle = (self.laser_initial_angle + step_count * self.laser_rotation_speed) % (2.0 * math.pi)
        end_x = eye_cx + math.cos(angle) * self.laser_length
        end_y = eye_cy + math.sin(angle) * self.laser_length
        return ((eye_cx, eye_cy), (end_x, end_y))
    
@dataclass
class WispGhost(Ghost):
    """Ghost that casts Will-o'-the-Wisp projectiles on a periodic cycle with alternating parabolic heights."""
    wisp_cycle: int = 140            # Ticks per casting cycle (~2.3 seconds)
    wisp_offset: int = 0             # Phase shift timing between multiple wisp ghosts
    wisp_mode: str = "alternating"   # "alternating" (high/low), "high", or "low"

    def is_charging_wisp(self, step_count: int) -> bool:
        """Telegraphing: Shows a charging spirit-flame in hand 25 ticks prior to release."""
        phase = (step_count + self.wisp_offset) % self.wisp_cycle
        return phase >= (self.wisp_cycle - 25)
    
    def can_cast_wisp(self, step_count: int) -> bool:
        """Triggers exactly at cycle completion."""
        return ((step_count + self.wisp_offset) % self.wisp_cycle) == 0

    def get_wisp_launch_params(self, step_count: int) -> Tuple[float, float, float, float]:
        """Returns (hand_x, hand_y, vx, vy) with varied parabolic velocities."""
        facing_dir = 1.0 if self.speed >= 0 else -1.0
        hand_x = self.x + (self.width if facing_dir > 0 else 0.0)
        hand_y = self.y + self.height * 0.55

        shot_number = (step_count + self.wisp_offset) // self.wisp_cycle

        if self.wisp_mode == "high" or (self.wisp_mode == "alternating" and shot_number % 2 == 0):
            # High Parabolic Arc: Floats high into the sky (player can crouch/slide beneath)
            vx = facing_dir * 1.8
            vy = -6.4
        else:
            # Low Parabolic Arc: Fast skimming shot (player must jump/double-jump over)
            vx = facing_dir * 3.4
            vy = -3.2

        return hand_x, hand_y, vx, vy

# ------------------ 
# For Undead Monster
# ------------------ 
class GraveMonsterState:
    DORMANT = 0   # Sleeping inside the tomb/ground
    EMERGING = 1  # Rising upward out of the grave
    ACTIVE = 2    # Fully emerged and pursuing player

@dataclass
class GraveMonster:
    """Base class for monsters ambushing Hoá from graves (ground pursuit)."""
    x: float
    y: float
    trigger_dist: float = 170.0   # Proximity to Hoá that triggers emergence
    emerge_speed: float = 1.6     # Vertical rising speed during emergence
    width: float = 24.0
    height: float = 32.0
    state: int = GraveMonsterState.DORMANT
    vx: float = 0.0
    vy: float = 0.0
    gravity: float = 0.4          # Gravitational acceleration when airborne
    leap_force_y: float = -7.5    # Upward impulse when leaping over pits
    is_alive: bool = True

    def __post_init__(self) -> None:
        self._start_x = self.x
        self._start_y = self.y
        self._start_state = self.state

    def reset(self) -> None:
        """Restore monster to its initial dormant position."""
        self.x = self._start_x
        self.y = self._start_y
        self.state = self._start_state
        self.vx = 0.0
        self.vy = 0.0
        self.is_alive = True

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)
    @property
    def right(self) -> float:
        return self.x + self.width
    @property
    def bottom(self) -> float:
        return self.y + self.height

    def _get_upcoming_pit(self, pits: List[Pit], lookahead: float) -> Optional[Pit]:
        """Finds the nearest pit ahead to the left within the lookahead distance."""
        for pit in pits:
            # Monster is to the right of pit.x_end and approaching it
            if 0.0 <= self.x - pit.x_end <= lookahead:
                return pit
        return None

    def _vault_pit(self, pit: Pit) -> None:
        """Calculates and applies exact horizontal & vertical velocity to clear the pit."""
        target_x = pit.x_start - self.width - 15.0
        leap_dist = self.x - target_x
        air_time = 2.0 * abs(self.leap_force_y) / self.gravity
        self.vx = -(leap_dist / air_time)
        self.vy = self.leap_force_y

    def update(self, player_x: float, floor_y: float, pits: Optional[List[Pit]] = None) -> None:
        if not self.is_alive:
            return 

        if pits is None:
            pits = []
        
        target_stand_y = floor_y - self.height

        if self.state == GraveMonsterState.DORMANT:
            if abs(player_x - self.x) <= self.trigger_dist:
                self.state = GraveMonsterState.EMERGING

        elif self.state == GraveMonsterState.EMERGING:
            # Rise upward out of the grave
            self.y -= self.emerge_speed
            if self.y <= target_stand_y:
                self.y = target_stand_y
                self.state = GraveMonsterState.ACTIVE

        elif self.state == GraveMonsterState.ACTIVE:
            self.update_pursuit(player_x, floor_y, pits)

        # Despawn once the entire sprite has crossed off the left edge of the screen
        if self.right <= 0.0:
            self.is_alive = False

    def update_pursuit(self, player_x: float, floor_y: float, pits: List[Pit]) -> None:
        """Hook method overridden by specialized subclasses."""
        pass

@dataclass
class QuyNhapTrang(GraveMonster):
    """Corpse Demon: Initially walks slowly, then accelerates aggressively toward player."""
    base_speed: float = 1.4
    max_speed: float = 5.2        # Faster than Hoá's walk speed (4.0)!
    acceleration: float = 0.1    # Speed increase per tick

    def __post_init__(self) -> None:
        super().__post_init__()
        self._current_speed = self.base_speed
        self._is_jumping: bool = False

    def reset(self) -> None:
        super().reset()
        self._current_speed = self.base_speed
        self._is_jumping = False

    def update_pursuit(self, player_x: float, floor_y: float, pits: List[Pit]) -> None:
        stand_y = floor_y - self.height

        if self._is_jumping:
            # Airborne trajectory vaulting over the pit
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy

            # Land safely past the pit
            if self.y >= stand_y:
                self.y = stand_y
                self.vy = 0.0
                self._is_jumping = False
        else:
            dynamic_lookahead = 15.0 + self._current_speed * 5.0
            upcoming_pit = self._get_upcoming_pit(pits, lookahead=dynamic_lookahead)
            if upcoming_pit is not None:
                self._vault_pit(upcoming_pit)
                self._is_jumping = True
            else:
                # Normal accelerating ground sprint
                self._current_speed = min(self.max_speed, self._current_speed + self.acceleration)
                self.vx = -self._current_speed
                self.x += self.vx
                self.y = stand_y

@dataclass
class CuongThi(GraveMonster):
    """Jiangshi (Hopping Vampire): Rises from grave and performs rhythmic hops strictly towards the left."""
    hop_speed_x: float = 2.4      # Horizontal speed during regular hop
    hop_force_y: float = -5.5     # Upward leap impulse for regular hops
    pause_duration: int = 18      # Ticks paused on ground between hops

    def __post_init__(self) -> None:
        super().__post_init__()
        self._pause_timer: int = 0
        self._is_grounded: bool = True

    def reset(self) -> None:
        super().reset()
        self._pause_timer = 0
        self._is_grounded = True

    def update_pursuit(self, player_x: float, floor_y: float, pits: List[Pit]) -> None:
        stand_y = floor_y - self.height

        if self._is_grounded:
            self.vx = 0.0
            self.vy = 0.0
            self.y = stand_y
            self._pause_timer += 1

            if self._pause_timer >= self.pause_duration:
                self._pause_timer = 0
                self._is_grounded = False

                # Check if a pit is directly ahead within hopping distance
                upcoming_pit = self._get_upcoming_pit(pits, lookahead=40.0)
                if upcoming_pit is not None:
                    # Supernatural Jiangshi vault over the pit gap
                    self._vault_pit(upcoming_pit)
                else:
                    # Normal hop strictly towards the left
                    self.vx = -self.hop_speed_x
                    self.vy = self.hop_force_y
        else:
            # Airborne hop physics using inherited gravity
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy

            # Land back on the ground
            if self.y >= stand_y:
                self.y = stand_y
                self.vy = 0.0
                self.vx = 0.0
                self._is_grounded = True

# ---------------
# For the Garlics
# ---------------
    
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