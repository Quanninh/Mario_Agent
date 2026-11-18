from core.base_env import RawState
from core.interfaces import BaseDiscretizer, DiscreteState


class MarioStateDiscretizer(BaseDiscretizer):
    """
    Transforms continuous RawState into:
        (is_grounded, vy_sign, obstacle_dist_bin, obstacle_height_bin, pit_dist_bin)

    Cardinality:
        is_grounded:          2 values (0: airborne, 1: on ground)
        vy_sign:              3 values (-1: falling, 0: stationary/flat, 1: ascending)
        obstacle_dist_bin:    4 values (0: None >80px, 1: Far 40-80px, 2: Close 15-40px, 3: Immediate <=15px)
        obstacle_height_bin:  2 values (0: None/Low <=32px, 1: High >32px)
        pit_dist_bin:         3 values (0: None >60px, 1: Far 25-60px, 2: Immediate <=25px)

    Total States: 2 * 3 * 4 * 2 * 3 = 144 states.
    """

    def __init__(self, velocity_threshold: float = 0.1, high_obstacle_threshold: float = 32.0):
        self.velocity_threshold = velocity_threshold
        self.high_obstacle_threshold = high_obstacle_threshold

    def discretize(self, raw_state: RawState) -> DiscreteState:
        # 1. Ground Contact Flag (0 or 1)
        grounded_bin = 1 if raw_state.is_grounded else 0

        # 2. Vertical Velocity Direction (-1: falling, 0: flat, 1: ascending)
        if raw_state.velocity_y < -self.velocity_threshold:
            vy_bin = 1   # Ascending (jumping up)
        elif raw_state.velocity_y > self.velocity_threshold:
            vy_bin = -1  # Falling (falling down)
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

        # 4. Obstacle Height Bin
        obs_height_bin = (
            1 if (obs_dist_bin != 0 and raw_state.obstacle_height > self.high_obstacle_threshold) else 0
        )

        # 5. Pit Proximity Bin
        dist_pit = raw_state.dist_to_pit
        if dist_pit < 0.0 or dist_pit > 60.0:
            pit_dist_bin = 0  # NONE (negative indicates no pit ahead)
        elif dist_pit >= 25.0:
            pit_dist_bin = 1  # FAR
        else:
            pit_dist_bin = 2  # IMMEDIATE

        return (grounded_bin, vy_bin, obs_dist_bin, obs_height_bin, pit_dist_bin)