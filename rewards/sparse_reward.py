from typing import Optional
from core.base_env import RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class SparseRewardStrategy(BaseRewardStrategy):
    """
    Sparse reward baseline: only fires on terminal outcomes (+100 goal, -50 death by pit/ghost)
    and milestone warmth item pickups (+10).
    """

    def __init__(
        self,
        goal_bonus: float = 100.0,
        death_penalty: float = -50.0,
        pit_penalty: Optional[float] = None,
        ghost_penalty: Optional[float] = None,
        fire_penalty: Optional[float] = None,
        laser_penalty: Optional[float] = None,
        wisp_penalty: Optional[float] = None,
        item_bonus: float = 30.0,
    ):
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self.pit_penalty = pit_penalty if pit_penalty is not None else death_penalty
        self.ghost_penalty = ghost_penalty if ghost_penalty is not None else death_penalty
        self.fire_penalty = fire_penalty if fire_penalty is not None else death_penalty
        self.laser_penalty = laser_penalty if laser_penalty is not None else death_penalty
        self.wisp_penalty = wisp_penalty if wisp_penalty is not None else death_penalty
        self.item_bonus = item_bonus

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
        if events.hit_ghost:
            return self.ghost_penalty
        if getattr(events, "hit_fire", False):
            return self.fire_penalty
        if getattr(events, "hit_laser", False):
            return self.laser_penalty
        if getattr(events, "hit_wisp", False):
            return self.wisp_penalty
        if events.collected_item:
            return self.item_bonus
        return 0.0
