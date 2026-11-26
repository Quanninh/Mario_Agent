from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

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

    # 1. Pit & Flame Hazards
    dist_to_pit: float        # Distance to nearest grave pit ahead (-1 if none)
    pit_width: float          # Width of that pit (px) (0 if none; helps choose single vs double jump)
    pit_fire_state: int       # 0: Safe/Off, 1: Warning embers, 2: Lethal flame
    pit_fire_cycle: float     # Progress in 100-tick fire cycle: [0.0 -> 1.0] (macro cycle)
    pit_fire_countdown: float # Exact ticks remaining until the next phase change (micro countdown)

    # 2. Ghost Patrol & Wave Kinematics
    dist_to_ghost: float     # Distance to nearest hovering ghost ahead (-1 if none)
    ghost_y: float           # Y-position of that ghost (0 if none; low vs high)
    ghost_vx: float          # Ghost's horizontal velocity (negative = flying toward Hoá, positive = away)
    ghost_vy: float          # Ghost's vertical sinusoidal velocity (negative = ascending, positive = descending)

    # 3. Rotating Laser Specter Hazard
    dist_to_laser: float     # Distance to nearest active laser beam ahead (-1 if none)
    laser_angle: float       # Current beam angle in radians [0, 2*pi] (0 if no laser)
    laser_is_blocking: bool  # True if laser beam is currently sweeping across ground/jumping path

    # 4. Will-o'-the-Wisp Projectiles & WispGhost Casting
    dist_to_wisp: float      # Distance to nearest active will-o'-the-wisp projectile ahead
    wisp_dir: int            # 1: Ahead (in front), -1: Behind (chasing from left), 0: None
    wisp_vy: float           # Wisp's vertical velocity (negative = rising, positive = falling, 0 if rolling)
    wisp_is_grounded: bool   # True if wisp is rolling along the grass floor (requires jump)
    wisp_ghost_charging: bool# True if nearby WispGhost is winding up / telegraphing a shot

    # 5. Grave Monster Ambush (Cương Thi & Quỷ Nhập Tràng)
    dist_to_monster: float   # Distance to active/emerging monster ahead (-1 if none)
    monster_y: float         # Y-position of that monster (ground sprint vs airborne hop)
    monster_state: int       # 0: DORMANT, 1: EMERGING, 2: ACTIVE
    monster_vx: float        # Monster's current sprint / hop velocity
    monster_is_grounded: bool# True if monster is on the ground, False if leaping over pit/airborne
    dist_to_monster_trigger: float # Distance to sleeping monster's grave trigger boundary (< 0 if none/already triggered)

    # 6. Items & Destination
    dist_to_item: float      # Distance to nearest warmth item ahead (-1 if none)
    dist_to_goal: float      # Distance remaining to Hoá's home

    # 7. Extended Continuous & Cyclic Kinematics (DQN-ready, backward-compatible)
    obstacle_dy: float = 0.0            # nearest_obs.y - player.y (0 if none)
    obstacle_width: float = 0.0         # Width of nearest obstacle ahead (0 if none)
    
    pit_fire_sin: float = 0.0           # sin(2*pi * pit_fire_cycle)
    pit_fire_cos: float = 1.0           # cos(2*pi * pit_fire_cycle)

    ghost_dy: float = 0.0               # nearest_ghost.y - player.y (0 if none)
    ghost_float_amplitude: float = 0.0  # Ghost sinusoidal bobbing amplitude
    ghost_float_speed: float = 0.0      # Ghost sinusoidal angular speed (rad/tick)
    ghost_phase_sin: float = 0.0        # sin(ghost._phase)
    ghost_phase_cos: float = 1.0        # cos(ghost._phase)
    ghost_dist_to_left_bound: float = 0.0   # ghost.x - ghost.patrol_min_x
    ghost_dist_to_right_bound: float = 0.0  # ghost.patrol_max_x - ghost.right

    laser_ghost_dx: float = 0.0         # eye_cx - player.right for nearest upcoming LaserGhost
    laser_ghost_dy: float = 0.0         # eye_cy - player.y for nearest upcoming LaserGhost
    laser_angle_sin: float = 0.0        # sin(cur_angle)
    laser_angle_cos: float = 1.0        # cos(cur_angle)
    laser_angular_vel: float = 0.0      # Instantaneous d(angle)/dt (rad/tick)
    laser_length: float = 0.0           # Beam reach in pixels
    laser_sweep_amplitude: float = 0.0  # 0.0 for 360-deg rotation, >0 for searchlight sweep

    wisp_ghost_dx: float = 0.0          # nearest_wg.x - player.right (0 if none)
    wisp_ghost_dy: float = 0.0          # nearest_wg.y - player.y (0 if none)
    wisp_cast_progress: float = 0.0     # phase / wisp_cycle in [0, 1)
    wisp_ticks_to_cast: float = 0.0     # wisp_cycle - phase
    wisp_next_is_high: bool = False     # True if next cast is High Arc, False if Low Fast Arc
    wisp_dy: float = 0.0                # nearest_wisp.y - player.y (0 if none)
    wisp_vx: float = 0.0                # nearest_wisp.vx (0 if none)
    wisp_roll_ticks_left: int = 0       # Remaining ground roll ticks if grounded, else 0

    monster_type: int = 0               # 0: none, 1: QuyNhapTrang, 2: CuongThi
    monster_is_sentry: bool = False     # True if patrolling an elevated platform
    monster_dy: float = 0.0             # nearest_monster.y - player.y (0 if none)
    monster_vy: float = 0.0             # nearest_monster.vy (0 if none)
    monster_pause_progress: float = 0.0 # CuongThi _pause_timer / pause_duration when grounded
    monster_speed_ratio: float = 0.0    # QuyNhapTrang (_current_speed - base) / (max - base)
    monster_dist_to_pit: float = -1.0   # Distance from monster.x to nearest pit to its left (-1 if none)

    

@dataclass
class GameEvents:
    """Events occurring in the current step (used by RewardStrategy)."""
    dx: float = 0.0               # Horizontal distance traveled this tick
    reached_goal: bool = False    # True if Hoá reached home [WIN]
    fell_in_pit: bool = False     # True if Hoá fell into a grave pit [DIE]
    hit_obstacle: bool = False    # True if Hoá bumped into a tombstone side
    hit_ghost: bool = False       # True if Hoá touched a floating ghost [DIE]
    hit_fire: bool = False        # True if Hoá touched an active flame jet [DIE]
    hit_laser: bool = False       # True if Hoá touched a ghost's eye laser [DIE]
    hit_wisp: bool = False        # True if Hoá touched a will-o'-the-wisp projectile [DIE]
    hit_monster: bool = False     # True if Hoá touched an emerging or active grave monster [DIE]
    monster_type: Optional[str] = None  # Specific monster type: "cuong_thi" or "quy_nhap_trang"
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
