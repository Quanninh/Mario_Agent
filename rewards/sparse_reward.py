from core.base_env import RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class SparseRewardStrategy(BaseRewardStrategy):
    """Only fires terminal rewards (+100 on goal, -50 on pit, 0 otherwise)."""

    def __init__(self, goal_bonus: float = 100.0, pit_penalty: float = -50.0):
        self.goal_bonus = goal_bonus
        self.pit_penalty = pit_penalty

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        if events.reached_goal:
            return self.goal_bonus
        if events.fell_in_pit:
            return self.pit_penalty
        return 0.0