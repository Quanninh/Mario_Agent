from core.base_env import RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class BalancedRewardStrategy(BaseRewardStrategy):
    """
    Production reward strategy for Hoá's Midnight Escape.
    Formulation:
        r = r_progress(dx) + r_goal * I(goal) + r_pit * I(pit) + r_ghost * I(ghost)
            + r_obstacle * I(hit) + r_item * I(item) + r_step
    """

    def __init__(
        self,
        weight_progress: float = 1.0,
        weight_regress: float = 1.5,
        goal_bonus: float = 200.0,
        pit_penalty: float = -75.0,
        ghost_penalty: float = -75.0,
        obstacle_penalty: float = -3.0,
        item_bonus: float = 25.0,
        step_penalty: float = -0.1,
    ):
        self.weight_progress = weight_progress
        self.weight_regress = weight_regress
        self.goal_bonus = goal_bonus
        self.pit_penalty = pit_penalty
        self.ghost_penalty = ghost_penalty
        self.obstacle_penalty = obstacle_penalty
        self.item_bonus = item_bonus
        self.step_penalty = step_penalty

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        # 1. Asymmetric horizontal progress shaping
        if events.dx > 0.0:
            r_progress = self.weight_progress * events.dx
        else:
            r_progress = self.weight_regress * events.dx

        # 2. Terminal Goal Bonus (reaching Hoá's home)
        r_goal = self.goal_bonus if events.reached_goal else 0.0

        # 3. Pit Fall Death Penalty
        r_pit = self.pit_penalty if events.fell_in_pit else 0.0

        # 4. Ghost Collision Death Penalty
        r_ghost = self.ghost_penalty if events.hit_ghost else 0.0

        # 5. Tombstone Obstacle Collision Penalty
        r_obstacle = self.obstacle_penalty if events.hit_obstacle else 0.0

        # 6. Warmth Item Collection Bonus
        r_item = self.item_bonus if events.collected_item else 0.0

        # 7. Constant living / time cost
        r_step = self.step_penalty

        return float(r_progress + r_goal + r_pit + r_ghost + r_obstacle + r_item + r_step)
