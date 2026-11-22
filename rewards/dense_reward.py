from typing import Optional
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
        pit_penalty: Optional[float] = None,
        ghost_penalty: Optional[float] = None,
        fire_penalty: Optional[float] = None,
        laser_penalty: Optional[float] = None,
        obstacle_penalty: float = -2.0,
        item_bonus: float = 30.0,
        idle_penalty: float = 0.0,
        timeout_penalty: float = -40.0,
    ):
        self.progress_mult = progress_mult
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self.pit_penalty = pit_penalty if pit_penalty is not None else death_penalty
        self.ghost_penalty = ghost_penalty if ghost_penalty is not None else death_penalty
        self.fire_penalty = fire_penalty if fire_penalty is not None else death_penalty
        self.laser_penalty = laser_penalty if laser_penalty is not None else death_penalty
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
        if events.fell_in_pit:
            reward += self.pit_penalty
        if events.hit_ghost:
            reward += self.ghost_penalty
        if getattr(events, "hit_fire", False):
            reward += self.fire_penalty
        if getattr(events, "hit_laser", False):
            reward += self.laser_penalty
        if events.hit_obstacle:
            reward += self.obstacle_penalty
        if events.collected_item:
            reward += self.item_bonus
        if abs(events.dx) < 0.05:
            reward += self.idle_penalty
        is_hazard_death = (
            events.fell_in_pit
            or events.hit_ghost
            or getattr(events, "hit_fire", False)
            or getattr(events, "hit_laser", False)
        )
        if done and not events.reached_goal and not is_hazard_death:
            reward += self.timeout_penalty

        return float(reward)
