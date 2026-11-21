from typing import Optional
from core.base_env import RawState
from core.interfaces import BaseDiscretizer, DiscreteState


class MarioStateDiscretizer(BaseDiscretizer):
    """
    Transforms continuous RawState into a compact hashable 7-tuple:
        (grounded_bin, vy_bin, obs_dist_bin, obs_height_bin, pit_dist_bin, ghost_dist_bin, ghost_alt_bin)

    Cardinality:
        grounded_bin:        2 values (0: airborne, 1: on ground)
        vy_bin:              3 values (-1: falling, 0: flat/stationary, 1: ascending)
        obs_dist_bin:        4 values (0: None >80px or cleared, 1: Far 45-80px, 2: Close 15-45px, 3: Immediate <15px)
        obs_height_bin:      2 values (0: Low <=50px or cleared, 1: High >50px)
        pit_dist_bin:        3 values (0: None >100px or <0px, 1: Far 32-100px, 2: Immediate <=32px)
        ghost_dist_bin:      4 values (0: None >80px or <0px, 1: Far 35-80px, 2: Close 0<dist<=35px, 3: Overlap dist==0.0)
        ghost_alt_bin:       2 values (0: Low/None - jump over, 1: High - walk under)

    State Cardinality:
        - Baseline levels (1..3) without ghosts: 2 * 3 * 4 * 2 * 3 * 1 * 1 = 144 effective states.
        - Ghost levels (4..5): 144 * 4 * 2 = 1,152 max states.
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
    ):
        self.velocity_threshold = velocity_threshold
        self.high_obstacle_threshold = high_obstacle_threshold
        self.floor_y = floor_y
        self.player_height = mario_height if mario_height is not None else player_height
        self.ghost_high_threshold = ghost_high_threshold
        self.obs_close_threshold = obs_close_threshold

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

        # 5. Pit Proximity Bin (safe take-off zone <= 32px, approach zone 32..100px, none > 100px)
        dist_pit = raw_state.dist_to_pit
        if dist_pit < 0.0 or dist_pit > 100.0:
            pit_dist_bin = 0  # NONE
        elif dist_pit > 32.0:
            pit_dist_bin = 1  # FAR (early warning approach zone 32..100px)
        else:
            pit_dist_bin = 2  # IMMEDIATE (safe take-off zone <= 32px)

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

        # 7. Ghost Altitude Bin (0: Low/None - jump over, 1: High - walk under)
        if ghost_dist_bin != 0 and ghost_y > 0.0 and ghost_y < (self.floor_y - self.ghost_high_threshold):
            ghost_alt_bin = 1  # HIGH (hovering high above ground -> walk under)
        else:
            ghost_alt_bin = 0  # LOW / NONE (hovering low -> jump over)

        return (
            grounded_bin,
            vy_bin,
            obs_dist_bin,
            obs_height_bin,
            pit_dist_bin,
            ghost_dist_bin,
            ghost_alt_bin,
        )


PlayerStateDiscretizer = MarioStateDiscretizer
