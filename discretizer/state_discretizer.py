from typing import Optional
from core.base_env import RawState
from core.interfaces import BaseDiscretizer, DiscreteState


class MarioStateDiscretizer(BaseDiscretizer):
    """
    Transforms continuous RawState into a compact hashable discrete state tuple:
        (is_grounded, vy_sign, obstacle_dist_bin, obstacle_height_bin, pit_dist_bin, ghost_dist_bin, ghost_alt_bin)

    Cardinality:
        is_grounded:          2 values (0: airborne, 1: on ground)
        vy_sign:              3 values (-1: falling, 0: stationary/flat, 1: ascending)
        obstacle_dist_bin:    4 values (0: None >80px or cleared, 1: Far 40-80px, 2: Close 15-40px, 3: Immediate <=15px)
        obstacle_height_bin:  2 values (0: Low <=50px, 1: High >50px)
        pit_dist_bin:         3 values (0: None >80px, 1: Far 32-80px, 2: Immediate <=32px)
        ghost_dist_bin:       3 values (0: None >80px or cleared, 1: Far 35-80px, 2: Close <=35px)
        ghost_alt_bin:        2 values (0: Low/None - jump over, 1: High - walk under)

    State Cardinality:
        - Baseline levels without ghosts: 2 * 3 * 4 * 2 * 3 * 1 * 1 = 144 effective states.
        - Ghost levels: 144 * 3 * 2 = 864 max states.
    """

    def __init__(
        self,
        velocity_threshold: float = 0.1,
        high_obstacle_threshold: float = 50.0,
        floor_y: float = 320.0,
        player_height: float = 32.0,
        mario_height: Optional[float] = None,
        ghost_high_threshold: float = 60.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.high_obstacle_threshold = high_obstacle_threshold
        self.floor_y = floor_y
        self.player_height = mario_height if mario_height is not None else player_height
        self.ghost_high_threshold = ghost_high_threshold

    def discretize(self, raw_state: RawState) -> DiscreteState:
        # Player coordinates (handling player_y with fallback to legacy mario_y if passed)
        player_y = getattr(raw_state, "player_y", getattr(raw_state, "mario_y", 0.0))

        # 1. Ground Contact Flag (0 or 1)
        grounded_bin = 1 if raw_state.is_grounded else 0

        # 2. Vertical Velocity Direction (-1: falling, 0: flat, 1: ascending)
        if raw_state.velocity_y < -self.velocity_threshold:
            vy_bin = 1   # Ascending (jumping up, vy is negative in Pygame)
        elif raw_state.velocity_y > self.velocity_threshold:
            vy_bin = -1  # Falling (falling down, vy is positive in Pygame)
        else:
            vy_bin = 0   # Flat / stationary

        # 3. Obstacle Proximity Bin
        dist_obs = raw_state.dist_to_obstacle
        if dist_obs < 0.0 or dist_obs > 80.0:
            obs_dist_bin = 0  # NONE (negative indicates no obstacle ahead)
        elif dist_obs >= 40.0:
            obs_dist_bin = 1  # FAR
        elif dist_obs >= 15.0:
            obs_dist_bin = 2  # CLOSE
        else:
            obs_dist_bin = 3  # IMMEDIATE

        # Check if obstacle is already cleared vertically:
        # When player's feet (player_y + player_height) are at or above obstacle's top (floor_y - obstacle_height + 2.0)
        player_feet_y = player_y + self.player_height
        obstacle_top_y = self.floor_y - raw_state.obstacle_height
        if obs_dist_bin != 0 and player_feet_y <= (obstacle_top_y + 2.0):
            obs_dist_bin = 0  # Treat obstacle as already cleared

        # 4. Obstacle Height Bin
        obs_height_bin = (
            1 if (obs_dist_bin != 0 and raw_state.obstacle_height > self.high_obstacle_threshold) else 0
        )

        # 5. Pit Proximity Bin (safe take-off zone <= 32px, approach zone 32..80px, none > 80px)
        dist_pit = raw_state.dist_to_pit
        if dist_pit < 0.0 or dist_pit > 80.0:
            pit_dist_bin = 0  # NONE (negative indicates no pit ahead)
        elif dist_pit > 32.0:
            pit_dist_bin = 1  # FAR (approach zone)
        else:
            pit_dist_bin = 2  # IMMEDIATE (safe take-off zone <= 32px)

        # 6. Ghost Proximity Bin (Level 4 & 5 hazards)
        dist_ghost = getattr(raw_state, "dist_to_ghost", -1.0)
        ghost_y = getattr(raw_state, "ghost_y", 0.0)

        if dist_ghost < 0.0 or dist_ghost > 80.0:
            ghost_dist_bin = 0  # NONE
        elif dist_ghost >= 35.0:
            ghost_dist_bin = 1  # FAR
        else:
            ghost_dist_bin = 2  # CLOSE

        # 7. Ghost Altitude Bin (0: low/none - jump over, 1: high - walk under)
        # High ghosts hover well above floor (ghost_y < floor_y - ghost_high_threshold)
        if ghost_dist_bin != 0 and ghost_y > 0.0:
            is_high_ghost = ghost_y < (self.floor_y - self.ghost_high_threshold)
            ghost_alt_bin = 1 if is_high_ghost else 0

            # If low ghost and player is jumping safely above its top, treat as cleared
            if not is_high_ghost and player_feet_y <= (ghost_y + 2.0):
                ghost_dist_bin = 0
                ghost_alt_bin = 0
        else:
            ghost_alt_bin = 0

        return (
            grounded_bin,
            vy_bin,
            obs_dist_bin,
            obs_height_bin,
            pit_dist_bin,
            ghost_dist_bin,
            ghost_alt_bin,
        )

