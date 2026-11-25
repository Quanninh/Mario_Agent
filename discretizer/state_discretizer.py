import math
from typing import Optional
from core.base_env import RawState
from core.interfaces import BaseDiscretizer, DiscreteState


class MarioStateDiscretizer(BaseDiscretizer):
    """
    Transforms continuous RawState into a compact hashable 15-tuple:
        (grounded_bin, vy_bin, obs_dist_bin, obs_height_bin, pit_dist_bin,
         ghost_dist_bin, ghost_alt_bin, laser_bin, wisp_bin, crouch_bin,
         double_jump_bin, monster_bin, pit_width_bin, ghost_vx_bin, vx_bin)

    Cardinality:
        grounded_bin:        2 values (0: airborne, 1: on ground)
        vy_bin:              3 values (-1: falling, 0: flat/stationary, 1: ascending)
        obs_dist_bin:        4 values (0: None >80px or cleared, 1: Far 45-80px, 2: Close 15-45px, 3: Immediate <15px)
        obs_height_bin:      2 values (0: Low <=50px or cleared, 1: High >50px)
        pit_dist_bin:        4 values (0: None >100px or <0px, 1: Far, 2: Immediate Safe, 3: Immediate/Imminent Fire)
        ghost_dist_bin:      4 values (0: None >80px or <0px, 1: Far 35-80px, 2: Close 0<dist<=35px, 3: Overlap dist==0.0)
        ghost_alt_bin:       2 values (0: Low/None - jump over, 1: High - walk/crouch under)
        laser_bin:           4 values (0: Safe/None, 1: Overhead Safe Window, 2: Ground Sweep Danger, 3: Closing Window)
        wisp_bin:            5 values (0: Safe/None, 1: Incoming Low, 2: Chasing Behind, 3: Incoming High, 4: Charging)
        crouch_bin:          2 values (0: Standing, 1: Crouching/Sliding)
        double_jump_bin:     2 values (0: Spent/Unavailable, 1: Armed/Ready)
        monster_bin:         4 values (0: Safe/None, 1: Airborne Hop, 2: Paused/Emerging, 3: Ground Sprint)
        pit_width_bin:       2 values (0: None/Narrow <=70px, 1: Wide >70px)
        ghost_vx_bin:        3 values (0: None/Stationary, 1: Advancing <0, 2: Retreating >0)
        vx_bin:              3 values (-1: Moving Left, 0: Stationary, 1: Moving Right)
    """

    def __init__(
        self,
        velocity_threshold: float = 0.1,
        high_obstacle_threshold: float = 50.0,
        floor_y: float = 320.0,
        player_height: float = 32.0,
        mario_height: Optional[float] = None,
        ghost_high_threshold: float = 60.0,
        obs_close_threshold: float = 45.0,
        ghost_height: float = 28.0,
        ghost_clearance_margin: float = 4.0,
        laser_threat_distance: float = 75.0,
        laser_lookahead_ticks: int = 8,
        wisp_threat_distance: float = 65.0,
        wisp_radius: float = 6.0,
        monster_threat_distance: float = 75.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.high_obstacle_threshold = high_obstacle_threshold
        self.floor_y = floor_y
        self.player_height = mario_height if mario_height is not None else player_height
        self.ghost_high_threshold = ghost_high_threshold
        self.obs_close_threshold = obs_close_threshold
        self.ghost_height = ghost_height
        self.ghost_clearance_margin = ghost_clearance_margin
        self.laser_threat_distance = laser_threat_distance
        self.laser_lookahead_ticks = laser_lookahead_ticks
        self.wisp_threat_distance = wisp_threat_distance
        self.wisp_radius = wisp_radius
        self.monster_threat_distance = monster_threat_distance

    def discretize(self, raw_state: RawState) -> DiscreteState:
        player_y = getattr(raw_state, "player_y", getattr(raw_state, "mario_y", 0.0))
        player_feet_y = player_y + self.player_height

        # 1. Ground Contact Flag (0 or 1)
        grounded_bin = 1 if raw_state.is_grounded else 0

        # 2. Vertical Velocity Direction (-1: falling, 0: flat, 1: ascending)
        if raw_state.velocity_y < -self.velocity_threshold:
            vy_bin = 1   # Ascending (jumping up, vy is negative in Pygame coordinates)
        elif raw_state.velocity_y > self.velocity_threshold:
            vy_bin = -1  # Falling (falling down, vy is positive in Pygame coordinates)
        else:
            vy_bin = 0   # Flat / stationary

        # 3. Obstacle Proximity Bin
        dist_obs = raw_state.dist_to_obstacle
        if dist_obs < 0.0 or dist_obs > 80.0:
            obs_dist_bin = 0  # NONE
        elif dist_obs >= self.obs_close_threshold:
            obs_dist_bin = 1  # FAR
        elif dist_obs >= 15.0:
            obs_dist_bin = 2  # CLOSE
        else:
            obs_dist_bin = 3  # IMMEDIATE

        # Check if obstacle is already cleared vertically:
        obstacle_top_y = self.floor_y - raw_state.obstacle_height
        if obs_dist_bin != 0 and player_feet_y <= (obstacle_top_y + 2.0):
            obs_dist_bin = 0  # Treat obstacle as already cleared

        # 4. Obstacle Height Bin
        obs_height_bin = (
            1 if (obs_dist_bin != 0 and raw_state.obstacle_height > self.high_obstacle_threshold) else 0
        )

        # 5. Fire- & Width-Aware Pit Proximity Bin (0: None, 1: Far, 2: Immediate Safe, 3: Immediate/Imminent Fire)
        dist_pit = getattr(raw_state, "dist_to_pit", -1.0)
        pit_width = getattr(raw_state, "pit_width", 0.0)
        pit_fire = getattr(raw_state, "pit_fire_state", 0)
        pit_countdown = getattr(raw_state, "pit_fire_countdown", 0.0)
        takeoff_dist = 42.0 if pit_width > 70.0 else 32.0
        imminent_fire = (pit_fire in (1, 2)) or (
            pit_fire == 0 and 0.0 < pit_countdown <= 20.0 and pit_width > 70.0
        )
        if dist_pit < 0.0 or dist_pit > 100.0:
            pit_dist_bin = 0  # NONE
        elif dist_pit > takeoff_dist:
            pit_dist_bin = 1  # FAR (early warning approach zone)
        else:
            pit_dist_bin = 3 if imminent_fire else 2  # Immediate danger zone

        # 6. Ghost Proximity Bin (4 bins: NONE, FAR, CLOSE, OVERLAP)
        dist_ghost = getattr(raw_state, "dist_to_ghost", -1.0)
        ghost_y = getattr(raw_state, "ghost_y", 0.0)

        if dist_ghost < 0.0 or dist_ghost > 80.0:
            ghost_dist_bin = 0  # NONE
        elif dist_ghost > 35.0:
            ghost_dist_bin = 1  # FAR
        elif dist_ghost > 0.0:
            ghost_dist_bin = 2  # CLOSE (safe jump take-off zone for low ghosts)
        else:
            ghost_dist_bin = 3  # OVERLAP (dist_ghost == 0.0, horizontally above/underneath)

        # 7. Ghost Altitude Bin (0: Low/None - jump over, 1: High - walk/crouch under)
        # Evaluates player-relative overhead clearance: ghost_y + ghost_height <= player_y - margin
        if (
            ghost_dist_bin != 0
            and ghost_y > 0.0
            and (ghost_y + self.ghost_height) <= (player_y - self.ghost_clearance_margin)
        ):
            ghost_alt_bin = 1  # HIGH (hovering safely overhead -> walk under)
        else:
            ghost_alt_bin = 0  # LOW / NONE (hovering low / eye-level -> jump over)

        # 8. Rotating Laser Hazard & Rotation Bin
        # 0: Safe/None, 1: Overhead Safe Window, 2: Ground Sweep Danger, 3: Closing Window
        dist_laser = getattr(raw_state, "dist_to_laser", -1.0)
        laser_angle = getattr(raw_state, "laser_angle", -1.0)
        laser_rot_speed = getattr(raw_state, "laser_rot_speed", 0.0)
        laser_blocking = getattr(raw_state, "laser_is_blocking", False)
        if dist_laser < 0.0 or dist_laser > self.laser_threat_distance:
            laser_bin = 0  # Safe / None
        elif laser_blocking:
            laser_bin = 2  # Actively blocking ground/jump path
        else:
            if laser_angle >= 0.0:
                norm_angle = laser_angle % (2.0 * math.pi)
                # Angle in [0, pi): sweeps down into ground/player plane (lethal danger)
                # Angle in [pi, 2*pi): points up toward clouds/sky (safe clearance window)
                if norm_angle < math.pi:
                    laser_bin = 2
                else:
                    next_angle = (norm_angle + laser_rot_speed * self.laser_lookahead_ticks) % (2.0 * math.pi)
                    laser_bin = 3 if (laser_rot_speed != 0.0 and next_angle < math.pi) else 1
            else:
                laser_bin = 1  # Threat flag when angle is unspecified

        # 9. Will-o'-the-Wisp Hazard Bin
        # 0: Safe/None, 1: Incoming Low/Grounded, 2: Chasing Behind, 3: Incoming High, 4: Charging
        dist_wisp = getattr(raw_state, "dist_to_wisp", -1.0)
        wisp_dir = getattr(raw_state, "wisp_dir", 0)
        wisp_y = getattr(raw_state, "wisp_y", 0.0)
        wisp_grounded = getattr(raw_state, "wisp_is_grounded", False)
        wisp_charging = getattr(raw_state, "wisp_charging", False) or getattr(raw_state, "wisp_ghost_charging", False)
        if 0.0 <= dist_wisp <= self.wisp_threat_distance:
            if wisp_dir < 0:
                wisp_bin = 2  # CHASING_BEHIND
            elif wisp_grounded:
                wisp_bin = 1  # INCOMING_LOW (rolling along floor -> jump over)
            elif wisp_y > 0.0 and (wisp_y + self.wisp_radius) <= (player_y - self.ghost_clearance_margin):
                wisp_bin = 3  # INCOMING_HIGH (overhead arc -> walk/crouch under)
            else:
                wisp_bin = 1  # INCOMING_LOW (skimming/rolling -> jump over)
        elif wisp_charging:
            wisp_bin = 4  # CHARGING (WispGhost telegraphing cast ahead)
        else:
            wisp_bin = 0  # SAFE / NONE

        # 10. Crouch Posture Bin (0: Standing, 1: Crouching)
        crouch_bin = 1 if getattr(raw_state, "is_crouching", False) else 0

        # 11. Mid-Air Double Jump Bin (0: Unavailable/Spent, 1: Ready)
        double_jump_bin = 1 if getattr(raw_state, "can_double_jump", False) else 0

        # 12. Grave Monster Proximity, State & Velocity Bin
        # 0: Safe/None
        # 1: Generic threat / Cuong Thi airborne hop (vy < -1.0 or !grounded)
        # 2: Cuong Thi paused on ground (grounded, vy >= -1.0)
        # 3: Quy Nhap Trang high-speed ground sprint (vx < -1.0)
        dist_monster = getattr(raw_state, "dist_to_monster", -1.0)
        monster_state = getattr(raw_state, "monster_state", 2)
        monster_grounded = getattr(raw_state, "monster_is_grounded", True)
        monster_vx = getattr(raw_state, "monster_vx", 0.0)
        monster_vy = getattr(raw_state, "monster_vy", 0.0)
        monster_type = getattr(raw_state, "monster_type", None)
        if dist_monster < 0.0 or dist_monster > self.monster_threat_distance:
            monster_bin = 0
        else:
            if monster_type == "quy_nhap_trang" or monster_vx < -2.0:
                monster_bin = 3  # Sprint demon rush
            elif not monster_grounded or monster_vy < -1.0:
                monster_bin = 1  # Airborne hop
            elif monster_type == "cuong_thi" or monster_vy != 0.0:
                monster_bin = 2  # Paused / emerging on ground
            else:
                monster_bin = 1

        # 13. Pit Width Bin (0: None/Narrow <=70px, 1: Wide >70px)
        pit_width_bin = 1 if (pit_dist_bin != 0 and pit_width > 70.0) else 0

        # 14. Ghost Horizontal Velocity Bin (0: None/Stationary, 1: Advancing <0, 2: Retreating >0)
        ghost_vx = getattr(raw_state, "ghost_vx", 0.0)
        if ghost_dist_bin == 0 or abs(ghost_vx) <= 0.05:
            ghost_vx_bin = 0
        elif ghost_vx < 0.0:
            ghost_vx_bin = 1
        else:
            ghost_vx_bin = 2

        # 15. Player Horizontal Momentum Bin (-1: Left, 0: Stationary, 1: Right)
        vx = getattr(raw_state, "velocity_x", 0.0)
        if vx < -0.5:
            vx_bin = -1
        elif vx > 0.5:
            vx_bin = 1
        else:
            vx_bin = 0

        return (
            grounded_bin,
            vy_bin,
            obs_dist_bin,
            obs_height_bin,
            pit_dist_bin,
            ghost_dist_bin,
            ghost_alt_bin,
            laser_bin,
            wisp_bin,
            crouch_bin,
            double_jump_bin,
            monster_bin,
            pit_width_bin,
            ghost_vx_bin,
            vx_bin,
        )


# Backward-compatible alias
PlayerStateDiscretizer = MarioStateDiscretizer
