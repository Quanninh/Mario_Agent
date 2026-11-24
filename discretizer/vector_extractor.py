from typing import Optional
import numpy as np
from core.base_env import RawState


class ContinuousStateExtractor:
    """
    Extracts a normalized, stationary continuous feature vector from RawState.
    
    Transforms the continuous physical and sensory measurements into a fixed-length
    1D numpy array suitable for Deep Q-Networks (DQN). All distance and kinematic
    features are normalized into [-1.0, 1.0] or [0.0, 1.0] ranges to facilitate
    stable neural network gradient descent.

    Feature vector dimension: 22
    """

    STATE_DIM: int = 22

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
        player_y = getattr(raw, "player_y", 288.0)
        vy = getattr(raw, "velocity_y", 0.0)
        vx = getattr(raw, "velocity_x", 0.0)

        # 1. Kinematics & Posture (6 features)
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

        # 3. Grave Pits & Fire Hazard (2 features)
        dist_pit = getattr(raw, "dist_to_pit", -1.0)
        norm_dist_pit = (
            np.clip(dist_pit / self.max_threat_distance, 0.0, 1.0)
            if dist_pit >= 0.0
            else 1.0
        )
        norm_fire_state = np.clip(getattr(raw, "pit_fire_state", 0) / 2.0, 0.0, 1.0)

        # 4. Ghost Proximity & Kinematics (3 features)
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

        # 5. Rotating Laser Threat (2 features)
        dist_laser = getattr(raw, "dist_to_laser", -1.0)
        norm_dist_laser = (
            np.clip(dist_laser / self.max_threat_distance, 0.0, 1.0)
            if dist_laser >= 0.0
            else 1.0
        )
        norm_laser_state = np.clip(getattr(raw, "laser_state", 0) / 2.0, 0.0, 1.0)

        # 6. Will-o'-the-Wisp Projectile (2 features)
        dist_wisp = getattr(raw, "dist_to_wisp", -1.0)
        norm_dist_wisp = (
            np.clip(dist_wisp / self.max_threat_distance, 0.0, 1.0)
            if dist_wisp >= 0.0
            else 1.0
        )
        norm_wisp_dir = float(np.clip(getattr(raw, "wisp_dir", 0), -1, 1))

        # 7. Grave Monsters (Quỷ Nhập Tràng & Cương Thi) (3 features)
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
        monster_active = 1.0 if dist_monster >= 0.0 else 0.0

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
                norm_y,
                norm_vx,
                norm_vy,
                grounded,
                double_jump,
                crouch,
                norm_dist_obs,
                norm_height_obs,
                norm_dist_pit,
                norm_fire_state,
                norm_dist_ghost,
                rel_ghost_y,
                norm_ghost_vx,
                norm_dist_laser,
                norm_laser_state,
                norm_dist_wisp,
                norm_wisp_dir,
                norm_dist_monster,
                rel_monster_y,
                monster_active,
                norm_dist_item,
                norm_dist_goal,
            ],
            dtype=np.float32,
        )
