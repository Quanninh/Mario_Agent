from core.base_env import Action, RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class BalancedRewardStrategy(BaseRewardStrategy):
    """
    Production reward strategy for Hoá's Midnight Escape.
    Formulation:
        r = r_progress(dx) + r_goal * I(goal) + r_pit * I(pit) + r_ghost * I(ghost)
            + r_obstacle * I(hit) + r_item * I(item) + r_step
            + r_unnecessary_jump + r_stall + r_timeout
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
        unnecessary_jump_penalty: float = -1.5,
        stall_penalty: float = -0.5,
        timeout_penalty: float = -50.0,
        vision_range: float = 80.0,
    ):
        self.weight_progress = weight_progress
        self.weight_regress = weight_regress
        self.goal_bonus = goal_bonus
        self.pit_penalty = pit_penalty
        self.ghost_penalty = ghost_penalty
        self.obstacle_penalty = obstacle_penalty
        self.item_bonus = item_bonus
        self.step_penalty = step_penalty
        self.unnecessary_jump_penalty = unnecessary_jump_penalty
        self.stall_penalty = stall_penalty
        self.timeout_penalty = timeout_penalty
        self.vision_range = vision_range

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
            reward = self.weight_progress * events.dx
        else:
            reward = self.weight_regress * events.dx

        # 2. Terminal Goal Bonus (reaching Hoá's home)
        if events.reached_goal:
            reward += self.goal_bonus

        # 3. Pit Fall Death Penalty
        if events.fell_in_pit:
            reward += self.pit_penalty

        # 4. Ghost Collision Death Penalty
        if events.hit_ghost:
            reward += self.ghost_penalty

        # 5. Tombstone Obstacle Collision Penalty
        if events.hit_obstacle:
            reward += self.obstacle_penalty

        # 6. Warmth Item Collection Bonus
        if events.collected_item:
            reward += self.item_bonus

        # 7. Constant living / step cost
        reward += self.step_penalty

        # 8. Unnecessary Jump Penalty
        # Fires ONLY when grounded, jumping, and no obstacle, pit, or ghost within vision_range
        if prev_state.is_grounded and action in (Action.JUMP, Action.RIGHT_JUMP):
            obs_far = (
                prev_state.dist_to_obstacle < 0.0
                or prev_state.dist_to_obstacle > self.vision_range
            )
            pit_far = (
                prev_state.dist_to_pit < 0.0
                or prev_state.dist_to_pit > self.vision_range
            )
            ghost_dist = getattr(prev_state, "dist_to_ghost", -1.0)
            ghost_far = ghost_dist < 0.0 or ghost_dist > self.vision_range

            if obs_far and pit_far and ghost_far:
                reward += self.unnecessary_jump_penalty

        # 9. Stall Penalty (standing still or backtracking without hitting an obstacle)
        if abs(events.dx) < 1e-6 and not events.hit_obstacle:
            reward += self.stall_penalty

        # 10. Timeout Penalty
        if (
            done
            and not events.reached_goal
            and not events.fell_in_pit
            and not events.hit_ghost
        ):
            reward += self.timeout_penalty

        return float(reward)
