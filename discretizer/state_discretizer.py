from typing import Optional
from core.base_env import RawState
from core.interfaces import BaseDiscretizer, DiscreteState


class MarioStateDiscretizer(BaseDiscretizer):
    """
    Transforms continuous RawState into a compact hashable 12-tuple:
        (grounded_bin, vy_bin, obs_dist_bin, obs_height_bin, pit_dist_bin,
         ghost_dist_bin, ghost_alt_bin, laser_bin, wisp_bin, crouch_bin,
         double_jump_bin, monster_bin)

    Cardinality:
        55,296 states when enable_wisp_sensor=False (default),
        110,592 states when enable_wisp_sensor=True.
        grounded_bin:        2 values (0: airborne, 1: on ground)
        vy_bin:              3 values (-1: falling vy>2.0, 0: flat/apex -3.0<=vy<=2.0 or grounded, 1: fast ascending vy<-3.0)
        obs_dist_bin:        4 values (0: None >80px or cleared, 1: Far 45-80px, 2: Close 15-45px, 3: Immediate <15px)
        obs_height_bin:      2 values (0: Low <=50px or cleared, 1: High >50px)
        pit_dist_bin:        4 values (0: None >100px or <0px; grounded: 1: Far Safe 24-100px, 2: Immediate Safe <=24px, 3: Fire Danger/Pre-warn; airborne: 1: Approaching 0<d<=100px, 2: Mid-pit dist==0.0, 3: Fire Danger/Pre-warn)
        ghost_dist_bin:      4 values (0: None >80px or <0px, 1: Far 35-80px, 2: Close 0<dist<=35px, 3: Overlap dist==0.0)
        ghost_alt_bin:       2 values (0: Low/None - jump over, 1: High - walk/crouch under)
        laser_bin:           3 values (0: Safe >85px or <0px, 1: Closing/Imminent <=28px or glitch/closing/first tick, 2: Valid Receding 0.5<=delta<=15.0px)
        wisp_bin:            3 bins (suppressed to 0 when enable_wisp_sensor=False)
        crouch_bin:          2 values (0: Standing, 1: Crouching/Sliding)
        double_jump_bin:     2 values (0: Spent/Unavailable, 1: Armed/Ready)
        monster_bin:         2 values (0: Safe/None, 1: Threat/Approaching <=95px)
    """

    def __init__(
        self,
        velocity_threshold: float = 0.1,
        fast_ascend_threshold: float = -3.0,
        fall_threshold: float = 2.0,
        high_obstacle_threshold: float = 50.0,
        floor_y: float = 320.0,
        player_height: float = 32.0,
        mario_height: Optional[float] = None,
        ghost_high_threshold: float = 60.0,
        obs_close_threshold: float = 45.0,
        ghost_height: float = 28.0,
        ghost_clearance_margin: float = 4.0,
        pit_immediate: float = 24.0,
        pit_fire_danger_distance: float = 65.0,
        fire_prewarn_ticks: int = 48,
        fire_prewarn_distance: float = 50.0,
        laser_threat_distance: float = 85.0,
        laser_imminent_distance: float = 28.0,
        laser_max_valid_delta: float = 15.0,
        laser_receding_min_delta: float = 0.5,
        wisp_threat_distance: float = 65.0,
        enable_wisp_sensor: bool = False,
        monster_threat_distance: float = 95.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.fast_ascend_threshold = fast_ascend_threshold
        self.fall_threshold = fall_threshold
        self.high_obstacle_threshold = high_obstacle_threshold
        self.floor_y = floor_y
        self.player_height = mario_height if mario_height is not None else player_height
        self.ghost_high_threshold = ghost_high_threshold
        self.obs_close_threshold = obs_close_threshold
        self.ghost_height = ghost_height
        self.ghost_clearance_margin = ghost_clearance_margin
        self.pit_immediate = pit_immediate
        self.pit_fire_danger_distance = pit_fire_danger_distance
        self.fire_prewarn_ticks = int(fire_prewarn_ticks)
        self.fire_prewarn_distance = float(fire_prewarn_distance)
        self.laser_threat_distance = laser_threat_distance
        self.laser_imminent_distance = laser_imminent_distance
        self.laser_max_valid_delta = float(laser_max_valid_delta)
        self.laser_receding_min_delta = float(laser_receding_min_delta)
        self.wisp_threat_distance = wisp_threat_distance
        self.enable_wisp_sensor = bool(enable_wisp_sensor)
        self.monster_threat_distance = monster_threat_distance
        self._prev_laser_world_x: Optional[float] = None
        self._prev_pit_fire_state: int = 0
        self._fire_phase_known: bool = False
        self._fire_safe_ticks: int = 0
        self._prev_pit_world_x: Optional[float] = None

    def reset(self) -> None:
        """Clears per-episode world-frame laser and hellfire-cycle tracking history."""
        self._prev_laser_world_x = None
        self._prev_pit_fire_state = 0
        self._fire_phase_known = False
        self._fire_safe_ticks = 0
        self._prev_pit_world_x = None

    @property
    def state_space_size(self) -> int:
        """Nominal discrete state space capacity (55,296 when wisp suppressed; 110,592 when enabled)."""
        wisp_cardinality = 2 if self.enable_wisp_sensor else 1
        return 55296 * wisp_cardinality

    def discretize(self, raw_state: RawState) -> DiscreteState:
        player_y = getattr(raw_state, "player_y", getattr(raw_state, "mario_y", 0.0))
        player_feet_y = player_y + self.player_height

        # 1. Ground Contact Flag (0 or 1)
        grounded_bin = 1 if raw_state.is_grounded else 0

        # 2. Double-Jump-Safe Vertical Velocity Bin (-1: falling, 0: flat/apex/grounded, 1: fast ascending)
        if raw_state.is_grounded:
            vy_bin = 0
        elif raw_state.velocity_y < self.fast_ascend_threshold:
            vy_bin = 1   # Fast ascend: double jumping here wastes height
        elif raw_state.velocity_y > self.fall_threshold:
            vy_bin = -1  # Falling
        else:
            vy_bin = 0   # Apex / safe double-jump extension window

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

        # 5. Airborne-Aware & Temporal Fire-Cycle Guard Pit Proximity Bin
        dist_pit = getattr(raw_state, "dist_to_pit", -1.0)
        pit_fire = getattr(raw_state, "pit_fire_state", 0)
        player_x = getattr(raw_state, "player_x", getattr(raw_state, "mario_x", 0.0))

        # Detect switching to a different upcoming pit (world x jumped by > 40 px)
        if dist_pit >= 0.0:
            pit_world_x = player_x + dist_pit
            if self._prev_pit_world_x is not None and abs(pit_world_x - self._prev_pit_world_x) > 40.0:
                self._fire_phase_known = False
                self._fire_safe_ticks = 0
                self._prev_pit_fire_state = pit_fire
            self._prev_pit_world_x = pit_world_x
        else:
            self._prev_pit_world_x = None
            self._fire_phase_known = False
            self._fire_safe_ticks = 0

        if pit_fire == 0:
            if self._prev_pit_fire_state == 2:
                # Exact cycle start (phase == 0) observed right after active flame ended
                self._fire_phase_known = True
                self._fire_safe_ticks = 0
            elif self._fire_phase_known:
                self._fire_safe_ticks += 1
                # Safety cap: if non-fire pit or missed transition, expire lock after 68 ticks
                if self._fire_safe_ticks > 68:
                    self._fire_phase_known = False
                    self._fire_safe_ticks = 0
        else:
            # In warning (1) or active (2) phase; wait for 2 -> 0 edge
            self._fire_phase_known = False
            self._fire_safe_ticks = 0

        self._prev_pit_fire_state = pit_fire

        fire_guard_active = (
            self.fire_prewarn_ticks > 0
            and self._fire_phase_known
            and self._fire_safe_ticks >= self.fire_prewarn_ticks
            and 0.0 <= dist_pit <= self.fire_prewarn_distance
        )
        fire_threatening = (pit_fire in (1, 2)) or fire_guard_active

        if dist_pit < 0.0 or dist_pit > 100.0:
            pit_dist_bin = 0  # NONE
        elif raw_state.is_grounded:
            if fire_threatening and dist_pit <= self.pit_fire_danger_distance:
                pit_dist_bin = 3  # FIRE_DANGER (active/warning or pre-warned eruption)
            elif dist_pit <= self.pit_immediate:
                pit_dist_bin = 2  # IMMEDIATE_SAFE (<=24px takeoff horizon)
            else:
                pit_dist_bin = 1  # FAR_SAFE (24..100px)
        else:
            if dist_pit == 0.0:
                pit_dist_bin = 3 if fire_threatening else 2  # MID_PIT_AIRBORNE or FIRE_DANGER
            elif fire_threatening and dist_pit <= self.pit_fire_danger_distance:
                pit_dist_bin = 3  # FIRE_DANGER
            else:
                pit_dist_bin = 1  # FAR_SAFE (airborne approaching pit)

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
        # While airborne, evaluate overhead clearance against standing head height so jumping over
        # pits never flips an overhead LaserGhost from HIGH to LOW and triggers suicidal double-jumps.
        standing_head_y = self.floor_y - self.player_height
        ref_head_y = min(standing_head_y, player_y) if raw_state.is_grounded else standing_head_y
        if (
            ghost_dist_bin != 0
            and ghost_y > 0.0
            and (ghost_y + self.ghost_height) <= (ref_head_y - self.ghost_clearance_margin)
        ):
            ghost_alt_bin = 1  # HIGH (hovering safely overhead -> walk/crouch under)
        else:
            ghost_alt_bin = 0  # LOW / NONE (eye-level -> jump over)

        # 8. Glitch-Filtered World-Frame Motion-Aware 3-Bin Laser Bin
        dist_laser = getattr(raw_state, "dist_to_laser", -1.0)
        if dist_laser < 0.0 or dist_laser > self.laser_threat_distance:
            laser_bin = 0
            self._prev_laser_world_x = None
        else:
            laser_world_x = player_x + dist_laser
            if dist_laser <= self.laser_imminent_distance:
                laser_bin = 1  # IMMINENT / CLOSING THREAT
            elif self._prev_laser_world_x is None:
                laser_bin = 1  # First sighting -> treat conservatively as threat
            else:
                delta = laser_world_x - self._prev_laser_world_x
                if abs(delta) > self.laser_max_valid_delta:
                    # Discontinuous beam switch between dual lasers -> conservative threat
                    laser_bin = 1
                elif self.laser_receding_min_delta <= delta <= self.laser_max_valid_delta:
                    laser_bin = 2  # VALID RECEDING BEAM
                else:
                    laser_bin = 1  # CLOSING OR STATIONARY THREAT
            self._prev_laser_world_x = laser_world_x

        # 9. Will-o'-the-Wisp Hazard Bin (suppressed to 0 by default to prevent state fragmentation)
        if not self.enable_wisp_sensor:
            wisp_bin = 0
        else:
            dist_wisp = getattr(raw_state, "dist_to_wisp", -1.0)
            wisp_dir = getattr(raw_state, "wisp_dir", 0)
            if dist_wisp < 0.0 or dist_wisp > self.wisp_threat_distance:
                wisp_bin = 0  # SAFE / NONE
            elif wisp_dir >= 0:
                wisp_bin = 1  # INCOMING_AHEAD
            else:
                wisp_bin = 2  # CHASING_BEHIND

        # 10. Crouch Posture Bin (0: Standing, 1: Crouching)
        crouch_bin = 1 if getattr(raw_state, "is_crouching", False) else 0

        # 11. Mid-Air Double Jump Bin (0: Unavailable/Spent, 1: Ready)
        double_jump_bin = 1 if getattr(raw_state, "can_double_jump", False) else 0

        # 12. Grave Monster Proximity Bin (0: Safe/None, 1: Threat/Approaching <= 95px)
        dist_monster = getattr(raw_state, "dist_to_monster", -1.0)
        monster_bin = 1 if (0.0 <= dist_monster <= self.monster_threat_distance) else 0

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
        )


# Backward-compatible alias
PlayerStateDiscretizer = MarioStateDiscretizer