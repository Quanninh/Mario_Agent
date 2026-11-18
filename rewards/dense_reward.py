from core.base_env import RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class DenseRewardStrategy(BaseRewardStrategy):
    """Shapes displacement, closing distance to goal, and penalizes stationary ticks."""

    def __init__(
        self,
        progress_mult: float = 1.2,
        goal_bonus: float = 150.0,
        death_penalty: float = -60.0,
        idle_penalty: float = -0.5,
    ):
        self.progress_mult = progress_mult
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self.idle_penalty = idle_penalty

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        # Distance delta toward finish line
        goal_dist_reduction = prev_state.dist_to_goal - next_state.dist_to_goal
        reward = goal_dist_reduction * self.progress_mult

        if events.reached_goal:
            reward += self.goal_bonus
        if events.fell_in_pit:
            reward += self.death_penalty
        if events.hit_obstacle:
            reward -= 2.0
        if abs(events.dx) < 0.05:
            reward += self.idle_penalty

        return float(reward)