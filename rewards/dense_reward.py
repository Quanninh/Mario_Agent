from core.base_env import RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class DenseRewardStrategy(BaseRewardStrategy):
    """
    Dense reward baseline: shapes displacement, distance-to-home delta,
    penalizes deaths (pit/ghost), and rewards warmth items.
    """

    def __init__(
        self,
        progress_mult: float = 1.2,
        goal_bonus: float = 150.0,
        death_penalty: float = -60.0,
        obstacle_penalty: float = -2.0,
        item_bonus: float = 20.0,
        idle_penalty: float = -0.5,
        timeout_penalty: float = -40.0,
    ):
        self.progress_mult = progress_mult
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self.obstacle_penalty = obstacle_penalty
        self.item_bonus = item_bonus
        self.idle_penalty = idle_penalty
        self.timeout_penalty = timeout_penalty

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        # Distance delta toward finish home
        goal_dist_reduction = prev_state.dist_to_goal - next_state.dist_to_goal
        reward = goal_dist_reduction * self.progress_mult

        if events.reached_goal:
            reward += self.goal_bonus
        if events.fell_in_pit or events.hit_ghost:
            reward += self.death_penalty
        if events.hit_obstacle:
            reward += self.obstacle_penalty
        if events.collected_item:
            reward += self.item_bonus
        if abs(events.dx) < 0.05:
            reward += self.idle_penalty
        if (
            done
            and not events.reached_goal
            and not events.fell_in_pit
            and not events.hit_ghost
        ):
            reward += self.timeout_penalty

        return float(reward)
