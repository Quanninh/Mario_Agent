from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Union

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
    pit_fire_state: int      # 0: Safe/Off, 1: Warning embers, 2: Lethal flame
    dist_to_ghost: float     # Distance to nearest hovering ghost ahead (-1 if none)
    ghost_y: float           # Y-position of that ghost (0 if none; low vs high)
    dist_to_item: float      # Distance to nearest warmth item ahead (-1 if none)
    dist_to_laser: float     # Distance to nearest active laser beam ahead (-1 if none)
    dist_to_wisp: float      # Distance to nearest active will-o'-the-wisp projectile ahead
    wisp_dir: int            # 1: Ahead (in front), -1: Behind (chasing from left), 0: None
    dist_to_monster: float   # Distance to active/emerging monster ahead (-1 if none)
    monster_y: float         # Y-position of that monster (ground sprint vs airborne hop)
    dist_to_goal: float      # Distance remaining to Hoá's home

    # Extended Pit, Kinematics, Laser, Wisp & Monster State (Levels 4-8 / game_frontend_v4)
    pit_width: float = 0.0                  # Width of nearest pit (px) (0 if none)
    pit_fire_cycle: float = 0.0             # Progress in 100-tick fire cycle: [0.0 -> 1.0]
    pit_fire_countdown: float = 0.0         # Exact ticks remaining until next phase change
    ghost_vx: float = 0.0                   # Ghost horizontal velocity (<0 = advancing, >0 = retreating)
    ghost_vy: float = 0.0                   # Ghost vertical sinusoidal velocity (<0 = ascending)
    laser_angle: float = -1.0               # Current angle of rotating laser [0, 2pi), -1 if none
    laser_rot_speed: float = 0.0            # Laser rotation speed & direction (rad/tick)
    laser_is_blocking: bool = False         # True if laser beam is sweeping across ground/jumping path
    wisp_vx: float = 0.0                    # Horizontal velocity of nearest threatening wisp
    wisp_vy: float = 0.0                    # Vertical velocity of nearest threatening wisp
    wisp_y: float = 0.0                     # Y-position of nearest threatening wisp (0 if none)
    wisp_is_grounded: bool = False          # True if wisp is rolling along the grass floor
    wisp_charging: bool = False             # True if an upcoming WispGhost is charging a cast
    wisp_ghost_charging: bool = False       # Alias for nearby WispGhost charging state
    monster_state: int = 0                  # 0: DORMANT, 1: EMERGING, 2: ACTIVE
    monster_vx: float = 0.0                 # Horizontal velocity of nearest monster (<0 = charging)
    monster_vy: float = 0.0                 # Vertical velocity of nearest monster (<0 = hopping/airborne)
    monster_is_grounded: bool = True        # True if monster is on ground, False if airborne
    dist_to_monster_trigger: float = -1.0   # Distance to sleeping monster's grave trigger (<0 if none)
    monster_type: Union[int, str, None] = 0 # 0/None: none, 1/"quy_nhap_trang": QuyNhapTrang, 2/"cuong_thi": CuongThi

    # Extended Continuous & Cyclic Kinematics (DQN-ready, backward-compatible)
    obstacle_dy: float = 0.0                # nearest_obs.y - player.y (0 if none)
    obstacle_width: float = 0.0             # Width of nearest obstacle ahead (0 if none)

    pit_fire_sin: float = 0.0               # sin(2*pi * pit_fire_cycle)
    pit_fire_cos: float = 1.0               # cos(2*pi * pit_fire_cycle)

    ghost_dy: float = 0.0                   # nearest_ghost.y - player.y (0 if none)
    ghost_float_amplitude: float = 0.0      # Ghost sinusoidal bobbing amplitude
    ghost_float_speed: float = 0.0          # Ghost sinusoidal angular speed (rad/tick)
    ghost_phase_sin: float = 0.0            # sin(ghost._phase)
    ghost_phase_cos: float = 1.0            # cos(ghost._phase)
    ghost_dist_to_left_bound: float = 0.0   # ghost.x - ghost.patrol_min_x
    ghost_dist_to_right_bound: float = 0.0  # ghost.patrol_max_x - ghost.right

    laser_ghost_dx: float = 0.0             # eye_cx - player.right for nearest upcoming LaserGhost
    laser_ghost_dy: float = 0.0             # eye_cy - player.y for nearest upcoming LaserGhost
    laser_angle_sin: float = 0.0            # sin(cur_angle)
    laser_angle_cos: float = 1.0            # cos(cur_angle)
    laser_angular_vel: float = 0.0          # Instantaneous d(angle)/dt (rad/tick)
    laser_length: float = 0.0               # Beam reach in pixels
    laser_sweep_amplitude: float = 0.0      # 0.0 for 360-deg rotation, >0 for searchlight sweep

    wisp_ghost_dx: float = 0.0              # nearest_wg.x - player.right (0 if none)
    wisp_ghost_dy: float = 0.0              # nearest_wg.y - player.y (0 if none)
    wisp_cast_progress: float = 0.0         # phase / wisp_cycle in [0, 1)
    wisp_ticks_to_cast: float = 0.0         # wisp_cycle - phase
    wisp_next_is_high: bool = False         # True if next cast is High Arc, False if Low Fast Arc
    wisp_dy: float = 0.0                    # nearest_wisp.y - player.y (0 if none)
    wisp_roll_ticks_left: int = 0           # Remaining ground roll ticks if grounded, else 0

    monster_is_sentry: bool = False         # True if patrolling an elevated platform
    monster_dy: float = 0.0                 # nearest_monster.y - player.y (0 if none)
    monster_pause_progress: float = 0.0     # CuongThi _pause_timer / pause_duration when grounded
    monster_speed_ratio: float = 0.0        # QuyNhapTrang (_current_speed - base) / (max - base)
    monster_dist_to_pit: float = -1.0       # Distance from monster.x to nearest pit to its left (-1 if none)

    

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
    died: bool = False            # True if Hoá died by any lethal hazard [DIE]

    @property
    def is_death(self) -> bool:
        return (
            self.died
            or self.fell_in_pit
            or self.hit_ghost
            or self.hit_fire
            or self.hit_laser
            or self.hit_wisp
            or self.hit_monster
        )

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
