from typing import Optional
from core.base_env import Action, RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class BalancedRewardStrategy(BaseRewardStrategy):
    """
    Production reward strategy for Hoá's Midnight Escape.
    Formulation:
        r = r_progress(dx) + r_goal * I(goal) + r_pit * I(pit) + r_ghost * I(ghost)
            + r_obstacle * I(hit) + r_item * I(item) + r_step
            + r_unnecessary_jump + r_timeout
    """

    def __init__(
        self,
        weight_progress: float = 1.0,
        weight_regress: float = 1.0,
        goal_bonus: float = 200.0,
        pit_penalty: float = -75.0,
        ghost_penalty: float = -75.0,
        fire_penalty: float = -75.0,
        laser_penalty: float = -75.0,
        wisp_penalty: float = -75.0,
        monster_penalty: float = -75.0,
        obstacle_penalty: float = -3.0,
        item_bonus: float = 30.0,
        step_penalty: float = -0.1,
        unnecessary_jump_penalty: float = -15.0,
        timeout_penalty: float = -50.0,
        obs_threat_distance: float = 50.0,
        pit_threat_distance: float = 45.0,
        ghost_threat_distance: float = 80.0,
        laser_threat_distance: float = 55.0,
        wisp_threat_distance: float = 65.0,
        monster_threat_distance: float = 75.0,
        ghost_height: float = 28.0,
        ghost_clearance_margin: float = 4.0,
    ):
        self.weight_progress = weight_progress
        self.weight_regress = weight_regress
        self.goal_bonus = goal_bonus
        self.pit_penalty = pit_penalty
        self.ghost_penalty = ghost_penalty
        self.fire_penalty = fire_penalty
        self.laser_penalty = laser_penalty
        self.wisp_penalty = wisp_penalty
        self.monster_penalty = monster_penalty
        self.obstacle_penalty = obstacle_penalty
        self.item_bonus = item_bonus
        self.step_penalty = step_penalty
        self.unnecessary_jump_penalty = unnecessary_jump_penalty
        self.timeout_penalty = timeout_penalty
        self.obs_threat_distance = obs_threat_distance
        self.pit_threat_distance = pit_threat_distance
        self.ghost_threat_distance = ghost_threat_distance
        self.laser_threat_distance = laser_threat_distance
        self.wisp_threat_distance = wisp_threat_distance
        self.monster_threat_distance = monster_threat_distance
        self.ghost_height = ghost_height
        self.ghost_clearance_margin = ghost_clearance_margin

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        # 1. Symmetric horizontal progress / regression shaping
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

        # 5. Pit Fire Death Penalty
        if getattr(events, "hit_fire", False):
            reward += self.fire_penalty

        # 6. Eye-Laser Death Penalty
        if getattr(events, "hit_laser", False):
            reward += self.laser_penalty

        # 6b. Will-o'-the-Wisp Death Penalty
        if getattr(events, "hit_wisp", False):
            reward += self.wisp_penalty

        # 6c. Grave Monster Death Penalty (Quỷ Nhập Tràng / Cương Thi)
        if getattr(events, "hit_monster", False):
            reward += self.monster_penalty

        # 7. Tombstone Obstacle Collision Penalty
        if events.hit_obstacle:
            reward += self.obstacle_penalty

        # 8. Warmth Item Collection Bonus
        if events.collected_item:
            reward += self.item_bonus

        # 9. Constant living / step cost
        reward += self.step_penalty

        # 10. Unnecessary Jump Penalty
        # Fires ONLY when grounded, taking off, and NO hazard justifies a jump.
        # Aligned with Discretizer threat horizons to avoid penalizing legitimate evasive maneuvers.
        # A ghost with overhead clearance (safe to walk under) does NOT exempt jumping.
        if prev_state.is_grounded and action in (Action.JUMP, Action.RIGHT_JUMP):
            obs_far = (
                prev_state.dist_to_obstacle < 0.0
                or prev_state.dist_to_obstacle > self.obs_threat_distance
            )
            pit_far = (
                prev_state.dist_to_pit < 0.0
                or prev_state.dist_to_pit > self.pit_threat_distance
            )
            dist_ghost = getattr(prev_state, "dist_to_ghost", -1.0)
            ghost_y = getattr(prev_state, "ghost_y", 0.0)
            player_y = getattr(prev_state, "player_y", getattr(prev_state, "mario_y", 288.0))

            ghost_overhead_cleared = (
                ghost_y > 0.0
                and (ghost_y + self.ghost_height) <= (player_y - self.ghost_clearance_margin)
            )
            low_ghost_near = (
                0.0 <= dist_ghost <= self.ghost_threat_distance
                and not ghost_overhead_cleared
            )
            laser_near = 0.0 <= getattr(prev_state, "dist_to_laser", -1.0) <= self.laser_threat_distance
            wisp_near = 0.0 <= getattr(prev_state, "dist_to_wisp", -1.0) <= self.wisp_threat_distance
            monster_near = 0.0 <= getattr(prev_state, "dist_to_monster", -1.0) <= self.monster_threat_distance
            if obs_far and pit_far and not low_ghost_near and not laser_near and not wisp_near and not monster_near:
                reward += self.unnecessary_jump_penalty

        # 11. Timeout Penalty (Ensure deaths are never penalized as timeouts)
        is_hazard_death = (
            events.fell_in_pit
            or events.hit_ghost
            or getattr(events, "hit_fire", False)
            or getattr(events, "hit_laser", False)
            or getattr(events, "hit_wisp", False)
            or getattr(events, "hit_monster", False)
        )
        if done and not events.reached_goal and not is_hazard_death:
            reward += self.timeout_penalty

        return float(reward)
