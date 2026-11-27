import math
from typing import Optional
import numpy as np
from core.base_env import RawState


class ContinuousStateExtractor:
    """
    Extracts a normalized, stationary continuous feature vector from RawState.
    
    Transforms the continuous physical and sensory measurements into a fixed-length
    1D numpy array suitable for Deep Q-Networks (DQN).
    
    Synchronized with the latest game_frontend_v4 & Section 7 DQN-ready dynamics:
      - Player kinematics & posture (6)
      - Tombstone obstacles: dist, height, dy, width (4)
      - Pit & periodic hellfire: dist, width, state, cycle, countdown, sin, cos (7)
      - 2D Ghost patrol & wave kinematics: dist, rel_y, vx, vy, amp, speed, sin, cos, bounds (10)
      - Rotating Laser hazard: dist, sin, cos, blocking, dx, dy, angular_vel, length, sweep_amp (9)
      - Will-o'-the-wisp & WispGhost casting: dist, dir, grounded, charging, dx, dy, progress, ticks, next_high, wisp_dy, vx, vy, roll_ticks (13)
      - Grave monsters (Quỷ Nhập Tràng & Cương Thi): dist, rel_y, vx, grounded, sprinting, vy, state, trigger, sentry, pause_progress, speed_ratio, dist_to_pit (12)
      - Warmth item & goal distance (2)

    Feature vector dimension: 63
    """

    STATE_DIM: int = 63

    def __init__(
        self,
        canvas_height: float = 400.0,
        max_threat_distance: float = 200.0,
        max_level_length: float = 8000.0,
    ) -> None:
        self.canvas_height = canvas_height
        self.max_threat_distance = max_threat_distance
        self.max_level_length = max_level_length

    def extract(self, raw: RawState) -> np.ndarray:
        player_y = getattr(raw, "player_y", getattr(raw, "mario_y", 288.0))
        vy = getattr(raw, "velocity_y", 0.0)
        vx = getattr(raw, "velocity_x", 0.0)

        # 1. Player Kinematics & Posture (6 features)
        norm_y = np.clip(player_y / self.canvas_height, 0.0, 1.0)
        norm_vx = np.clip(vx / 7.0, -1.0, 1.0)
        norm_vy = np.clip(vy / 15.0, -1.0, 1.0)
        grounded = 1.0 if getattr(raw, "is_grounded", False) else 0.0
        double_jump = 1.0 if getattr(raw, "can_double_jump", False) else 0.0
        crouch = 1.0 if getattr(raw, "is_crouching", False) else 0.0

        # 2. Tombstone Obstacles (4 features)
        dist_obs = getattr(raw, "dist_to_obstacle", -1.0)
        norm_dist_obs = (
            np.clip(dist_obs / self.max_threat_distance, 0.0, 1.0)
            if dist_obs >= 0.0
            else 1.0
        )
        norm_height_obs = np.clip(getattr(raw, "obstacle_height", 0.0) / 100.0, 0.0, 1.0)
        norm_obs_dy = (
            np.clip(getattr(raw, "obstacle_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
            if dist_obs >= 0.0
            else 0.0
        )
        norm_obs_width = np.clip(getattr(raw, "obstacle_width", 0.0) / 100.0, 0.0, 1.0)

        # 3. Grave Pits & Periodic Hellfire (7 features)
        dist_pit = getattr(raw, "dist_to_pit", -1.0)
        norm_dist_pit = (
            np.clip(dist_pit / self.max_threat_distance, 0.0, 1.0)
            if dist_pit >= 0.0
            else 1.0
        )
        pit_width = getattr(raw, "pit_width", 0.0)
        norm_pit_width = np.clip(pit_width / 150.0, 0.0, 1.0)
        norm_fire_state = np.clip(getattr(raw, "pit_fire_state", 0) / 2.0, 0.0, 1.0)
        pit_fire_cycle = np.clip(getattr(raw, "pit_fire_cycle", 0.0), 0.0, 1.0)
        norm_fire_countdown = np.clip(getattr(raw, "pit_fire_countdown", 0.0) / 100.0, 0.0, 1.0)
        pit_fire_sin = float(np.clip(getattr(raw, "pit_fire_sin", 0.0), -1.0, 1.0))
        pit_fire_cos = float(np.clip(getattr(raw, "pit_fire_cos", 1.0), -1.0, 1.0))

        # 4. Ghost Proximity & Wave Kinematics (10 features)
        dist_ghost = getattr(raw, "dist_to_ghost", -1.0)
        norm_dist_ghost = (
            np.clip(dist_ghost / self.max_threat_distance, 0.0, 1.0)
            if dist_ghost >= 0.0
            else 1.0
        )
        ghost_y = getattr(raw, "ghost_y", 0.0)
        if dist_ghost >= 0.0 and ghost_y > 0.0:
            rel_ghost_y = np.clip((ghost_y - player_y) / self.max_threat_distance, -1.0, 1.0)
        elif dist_ghost >= 0.0:
            rel_ghost_y = np.clip(getattr(raw, "ghost_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
        else:
            rel_ghost_y = 0.0
        norm_ghost_vx = np.clip(getattr(raw, "ghost_vx", 0.0) / 5.0, -1.0, 1.0)
        norm_ghost_vy = np.clip(getattr(raw, "ghost_vy", 0.0) / 5.0, -1.0, 1.0)
        norm_ghost_amp = np.clip(getattr(raw, "ghost_float_amplitude", 0.0) / 100.0, 0.0, 1.0)
        norm_ghost_float_speed = np.clip(getattr(raw, "ghost_float_speed", 0.0) / 0.5, -1.0, 1.0)
        ghost_phase_sin = float(np.clip(getattr(raw, "ghost_phase_sin", 0.0), -1.0, 1.0))
        ghost_phase_cos = float(np.clip(getattr(raw, "ghost_phase_cos", 1.0), -1.0, 1.0))
        norm_ghost_left_bound = np.clip(getattr(raw, "ghost_dist_to_left_bound", 0.0) / 250.0, 0.0, 1.0)
        norm_ghost_right_bound = np.clip(getattr(raw, "ghost_dist_to_right_bound", 0.0) / 250.0, 0.0, 1.0)

        # 5. Rotating Laser Hazard (9 features)
        dist_laser = getattr(raw, "dist_to_laser", -1.0)
        norm_dist_laser = (
            np.clip(dist_laser / self.max_threat_distance, 0.0, 1.0)
            if dist_laser >= 0.0
            else 1.0
        )
        laser_angle = getattr(raw, "laser_angle", -1.0)
        raw_laser_sin = getattr(raw, "laser_angle_sin", None)
        raw_laser_cos = getattr(raw, "laser_angle_cos", None)
        if dist_laser >= 0.0 and raw_laser_sin is not None and raw_laser_cos is not None and (raw_laser_sin != 0.0 or raw_laser_cos != 1.0 or laser_angle == 0.0):
            laser_sin = float(np.clip(raw_laser_sin, -1.0, 1.0))
            laser_cos = float(np.clip(raw_laser_cos, -1.0, 1.0))
        elif dist_laser >= 0.0 and laser_angle >= 0.0:
            laser_sin = float(math.sin(laser_angle))
            laser_cos = float(math.cos(laser_angle))
        else:
            laser_sin = 0.0
            laser_cos = 0.0
        laser_is_blocking = 1.0 if getattr(raw, "laser_is_blocking", False) else 0.0
        norm_laser_ghost_dx = np.clip(getattr(raw, "laser_ghost_dx", 0.0) / self.max_threat_distance, -1.0, 1.0)
        norm_laser_ghost_dy = np.clip(getattr(raw, "laser_ghost_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
        norm_laser_angular_vel = np.clip(
            getattr(raw, "laser_angular_vel", getattr(raw, "laser_rot_speed", 0.0)) / 0.2,
            -1.0,
            1.0,
        )
        norm_laser_length = np.clip(getattr(raw, "laser_length", 0.0) / 200.0, 0.0, 1.0)
        norm_laser_sweep_amp = np.clip(getattr(raw, "laser_sweep_amplitude", 0.0) / math.pi, 0.0, 1.0)

        # 6. Will-o'-the-Wisp Projectile & Casting (13 features)
        dist_wisp = getattr(raw, "dist_to_wisp", -1.0)
        norm_dist_wisp = (
            np.clip(dist_wisp / self.max_threat_distance, 0.0, 1.0)
            if dist_wisp >= 0.0
            else 1.0
        )
        norm_wisp_dir = float(np.clip(getattr(raw, "wisp_dir", 0), -1, 1))
        wisp_is_grounded = 1.0 if getattr(raw, "wisp_is_grounded", False) else 0.0
        wisp_charging = 1.0 if (
            getattr(raw, "wisp_charging", False) or getattr(raw, "wisp_ghost_charging", False)
        ) else 0.0
        norm_wisp_ghost_dx = np.clip(getattr(raw, "wisp_ghost_dx", 0.0) / self.max_threat_distance, -1.0, 1.0)
        norm_wisp_ghost_dy = np.clip(getattr(raw, "wisp_ghost_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
        wisp_cast_progress = float(np.clip(getattr(raw, "wisp_cast_progress", 0.0), 0.0, 1.0))
        norm_wisp_ticks_to_cast = np.clip(getattr(raw, "wisp_ticks_to_cast", 0.0) / 200.0, 0.0, 1.0)
        wisp_next_is_high = 1.0 if getattr(raw, "wisp_next_is_high", False) else 0.0
        norm_wisp_dy = np.clip(getattr(raw, "wisp_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
        norm_wisp_vx = np.clip(getattr(raw, "wisp_vx", 0.0) / 8.0, -1.0, 1.0)
        norm_wisp_vy = np.clip(getattr(raw, "wisp_vy", 0.0) / 8.0, -1.0, 1.0)
        norm_wisp_roll_ticks = np.clip(getattr(raw, "wisp_roll_ticks_left", 0) / 100.0, 0.0, 1.0)

        # 7. Grave Monsters (Quỷ Nhập Tràng & Cương Thi) (12 features)
        dist_monster = getattr(raw, "dist_to_monster", -1.0)
        norm_dist_monster = (
            np.clip(dist_monster / self.max_threat_distance, 0.0, 1.0)
            if dist_monster >= 0.0
            else 1.0
        )
        monster_y = getattr(raw, "monster_y", 0.0)
        if dist_monster >= 0.0 and monster_y > 0.0:
            rel_monster_y = np.clip((monster_y - player_y) / self.max_threat_distance, -1.0, 1.0)
        elif dist_monster >= 0.0:
            rel_monster_y = np.clip(getattr(raw, "monster_dy", 0.0) / self.max_threat_distance, -1.0, 1.0)
        else:
            rel_monster_y = 0.0
        monster_vx = getattr(raw, "monster_vx", 0.0)
        norm_monster_vx = np.clip(monster_vx / 5.0, -1.0, 1.0)
        norm_monster_vy = np.clip(getattr(raw, "monster_vy", 0.0) / 15.0, -1.0, 1.0)
        monster_is_grounded = 1.0 if getattr(raw, "monster_is_grounded", True) else 0.0
        monster_type = getattr(raw, "monster_type", None)
        monster_is_sprinting = 1.0 if (
            monster_type in ("quy_nhap_trang", 1) or monster_vx < -2.0
        ) else 0.0
        norm_monster_state = np.clip(getattr(raw, "monster_state", 0) / 2.0, 0.0, 1.0)
        dist_trig = getattr(raw, "dist_to_monster_trigger", -1.0)
        norm_monster_trigger = (
            np.clip(dist_trig / self.max_threat_distance, 0.0, 1.0)
            if dist_trig >= 0.0
            else 1.0
        )
        monster_is_sentry = 1.0 if getattr(raw, "monster_is_sentry", False) else 0.0
        monster_pause_progress = float(np.clip(getattr(raw, "monster_pause_progress", 0.0), 0.0, 1.0))
        monster_speed_ratio = float(np.clip(getattr(raw, "monster_speed_ratio", 0.0), 0.0, 1.0))
        m_pit = getattr(raw, "monster_dist_to_pit", -1.0)
        norm_monster_dist_to_pit = (
            np.clip(m_pit / self.max_threat_distance, 0.0, 1.0)
            if m_pit >= 0.0
            else 1.0
        )

        # 8. Warmth Garlic Item & Level Goal (2 features)
        dist_item = getattr(raw, "dist_to_item", -1.0)
        norm_dist_item = (
            np.clip(dist_item / self.max_threat_distance, 0.0, 1.0)
            if dist_item >= 0.0
            else 1.0
        )
        dist_goal = getattr(raw, "dist_to_goal", self.max_level_length)
        norm_dist_goal = np.clip(dist_goal / self.max_level_length, 0.0, 1.0)

        return np.array(
            [
                # 1. Kinematics (6)
                norm_y,
                norm_vx,
                norm_vy,
                grounded,
                double_jump,
                crouch,
                # 2. Obstacles (4)
                norm_dist_obs,
                norm_height_obs,
                norm_obs_dy,
                norm_obs_width,
                # 3. Pits (7)
                norm_dist_pit,
                norm_pit_width,
                norm_fire_state,
                pit_fire_cycle,
                norm_fire_countdown,
                pit_fire_sin,
                pit_fire_cos,
                # 4. Ghosts (10)
                norm_dist_ghost,
                rel_ghost_y,
                norm_ghost_vx,
                norm_ghost_vy,
                norm_ghost_amp,
                norm_ghost_float_speed,
                ghost_phase_sin,
                ghost_phase_cos,
                norm_ghost_left_bound,
                norm_ghost_right_bound,
                # 5. Lasers (9)
                norm_dist_laser,
                laser_sin,
                laser_cos,
                laser_is_blocking,
                norm_laser_ghost_dx,
                norm_laser_ghost_dy,
                norm_laser_angular_vel,
                norm_laser_length,
                norm_laser_sweep_amp,
                # 6. Wisps (13)
                norm_dist_wisp,
                norm_wisp_dir,
                wisp_is_grounded,
                wisp_charging,
                norm_wisp_ghost_dx,
                norm_wisp_ghost_dy,
                wisp_cast_progress,
                norm_wisp_ticks_to_cast,
                wisp_next_is_high,
                norm_wisp_dy,
                norm_wisp_vx,
                norm_wisp_vy,
                norm_wisp_roll_ticks,
                # 7. Monsters (12)
                norm_dist_monster,
                rel_monster_y,
                norm_monster_vx,
                monster_is_grounded,
                monster_is_sprinting,
                norm_monster_vy,
                norm_monster_state,
                norm_monster_trigger,
                monster_is_sentry,
                monster_pause_progress,
                monster_speed_ratio,
                norm_monster_dist_to_pit,
                # 8. Items & Goal (2)
                norm_dist_item,
                norm_dist_goal,
            ],
            dtype=np.float32,
        )
