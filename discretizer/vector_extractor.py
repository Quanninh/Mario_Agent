import math
from typing import Optional
import numpy as np
from core.base_env import RawState


class ContinuousStateExtractor:
    """
    Extracts a normalized, stationary continuous feature vector from RawState.
    
    Transforms the continuous physical and sensory measurements into a fixed-length
    1D numpy array suitable for Deep Q-Networks (DQN).
    
    Synchronized with the latest game_frontend_v4 dynamics and BalancedRewardStrategy:
      - Pit width & fire countdown / macro cycle
      - 2D Ghost horizontal velocity & vertical sinusoidal wave kinematics
      - Continuous trigonometric (sin, cos) laser angle & ground blocking flag
      - Will-o'-the-wisp ground-rolling status & projectile velocity
      - Grave monster sprint/hop kinematics & type identification
      - Warmth item & goal distance

    Feature vector dimension: 32
    """

    STATE_DIM: int = 32

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

        # 2. Tombstone Obstacles (2 features)
        dist_obs = getattr(raw, "dist_to_obstacle", -1.0)
        norm_dist_obs = (
            np.clip(dist_obs / self.max_threat_distance, 0.0, 1.0)
            if dist_obs >= 0.0
            else 1.0
        )
        norm_height_obs = np.clip(getattr(raw, "obstacle_height", 0.0) / 100.0, 0.0, 1.0)

        # 3. Grave Pits & Periodic Hellfire (5 features)
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

        # 4. Ghost Proximity & Kinematics (4 features)
        dist_ghost = getattr(raw, "dist_to_ghost", -1.0)
        norm_dist_ghost = (
            np.clip(dist_ghost / self.max_threat_distance, 0.0, 1.0)
            if dist_ghost >= 0.0
            else 1.0
        )
        ghost_y = getattr(raw, "ghost_y", 0.0)
        rel_ghost_y = (
            np.clip((ghost_y - player_y) / self.max_threat_distance, -1.0, 1.0)
            if dist_ghost >= 0.0 and ghost_y > 0.0
            else 0.0
        )
        norm_ghost_vx = np.clip(getattr(raw, "ghost_vx", 0.0) / 5.0, -1.0, 1.0)
        norm_ghost_vy = np.clip(getattr(raw, "ghost_vy", 0.0) / 5.0, -1.0, 1.0)

        # 5. Rotating Laser Hazard (4 features: continuous sin/cos angle, blocking & distance)
        dist_laser = getattr(raw, "dist_to_laser", -1.0)
        norm_dist_laser = (
            np.clip(dist_laser / self.max_threat_distance, 0.0, 1.0)
            if dist_laser >= 0.0
            else 1.0
        )
        laser_angle = getattr(raw, "laser_angle", -1.0)
        if dist_laser >= 0.0 and laser_angle >= 0.0:
            laser_sin = float(math.sin(laser_angle))
            laser_cos = float(math.cos(laser_angle))
        else:
            laser_sin = 0.0
            laser_cos = 0.0
        laser_is_blocking = 1.0 if getattr(raw, "laser_is_blocking", False) else 0.0

        # 6. Will-o'-the-Wisp Projectile & Casting (4 features)
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

        # 7. Grave Monsters (Quỷ Nhập Tràng & Cương Thi) (5 features)
        dist_monster = getattr(raw, "dist_to_monster", -1.0)
        norm_dist_monster = (
            np.clip(dist_monster / self.max_threat_distance, 0.0, 1.0)
            if dist_monster >= 0.0
            else 1.0
        )
        monster_y = getattr(raw, "monster_y", 0.0)
        rel_monster_y = (
            np.clip((monster_y - player_y) / self.max_threat_distance, -1.0, 1.0)
            if dist_monster >= 0.0 and monster_y > 0.0
            else 0.0
        )
        norm_monster_vx = np.clip(getattr(raw, "monster_vx", 0.0) / 5.0, -1.0, 1.0)
        monster_is_grounded = 1.0 if getattr(raw, "monster_is_grounded", True) else 0.0
        monster_type = getattr(raw, "monster_type", None)
        monster_vx = getattr(raw, "monster_vx", 0.0)
        monster_is_sprinting = 1.0 if (monster_type == "quy_nhap_trang" or monster_vx < -2.0) else 0.0

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
                # 2. Obstacles (2)
                norm_dist_obs,
                norm_height_obs,
                # 3. Pits (5)
                norm_dist_pit,
                norm_pit_width,
                norm_fire_state,
                pit_fire_cycle,
                norm_fire_countdown,
                # 4. Ghosts (4)
                norm_dist_ghost,
                rel_ghost_y,
                norm_ghost_vx,
                norm_ghost_vy,
                # 5. Lasers (4)
                norm_dist_laser,
                laser_sin,
                laser_cos,
                laser_is_blocking,
                # 6. Wisps (4)
                norm_dist_wisp,
                norm_wisp_dir,
                wisp_is_grounded,
                wisp_charging,
                # 7. Monsters (5)
                norm_dist_monster,
                rel_monster_y,
                norm_monster_vx,
                monster_is_grounded,
                monster_is_sprinting,
                # 8. Items & Goal (2)
                norm_dist_item,
                norm_dist_goal,
            ],
            dtype=np.float32,
        )
