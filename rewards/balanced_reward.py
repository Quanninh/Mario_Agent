import math
from typing import Optional
from core.base_env import Action, RawState, GameEvents
from core.interfaces import BaseRewardStrategy


class BalancedRewardStrategy(BaseRewardStrategy):
    """
    Production balanced reward strategy for Hoá's Midnight Escape.
    Formulation:
        r = r_progress(frontier / dx) + r_goal * I(goal) + r_death * I(died)
            + r_obstacle * I(hit) + r_item * I(item) + r_step
            + r_unnecessary_jump + r_timeout

    Key Architectural Principles:
    1. Unified death event check (events.died / is_death) -> single death_penalty.
    2. Non-punitive backward dodging (weight_regress = 0.0 / frontier max_x).
    3. Takeoff-only jump penalty: Jump penalty is only evaluated on takeoff
       (prev_state.is_grounded and action in (JUMP, RIGHT_JUMP)). Mid-air flight
       frames are NEVER penalized. Jumps near pits, obstacles, or low ghosts
       are fully exempted. Jumps underneath overhead ghosts or on flat ground
       are penalized to prevent bunny-hopping and fatal collisions.
    4. Strongly positive progress gradient: weight_progress = 1.0, goal_bonus = 200.0,
       death_penalty = -75.0 so jumping across gaps strongly dominates walking into them.
    """

    def __init__(
        self,
        weight_progress: float = 1.0,
        weight_regress: float = 1.0,
        goal_bonus: float = 200.0,
        death_penalty: float = -75.0,
        pit_penalty: Optional[float] = None,
        ghost_penalty: Optional[float] = None,
        fire_penalty: Optional[float] = None,
        laser_penalty: Optional[float] = None,
        wisp_penalty: Optional[float] = None,
        monster_penalty: Optional[float] = None,
        obstacle_penalty: float = -3.0,
        item_bonus: float = 30.0,
        step_penalty: float = -0.1,
        unnecessary_jump_penalty: float = -1.5,
        jump_cost: Optional[float] = None,
        timeout_penalty: float = -100.0,
        use_max_x: bool = False,
        obs_threat_distance: float = 50.0,
        pit_threat_distance: float = 45.0,
        runway_pit_distance: float = 85.0,
        ghost_threat_distance: float = 80.0,
        laser_threat_distance: float = 75.0,
        wisp_threat_distance: float = 65.0,
        monster_threat_distance: float = 75.0,
        ghost_height: float = 28.0,
        ghost_clearance_margin: float = 4.0,
        laser_danger_penalty: float = -3.0,
        laser_lookahead_ticks: int = 8,
        ghost_advance_penalty: float = -3.0,
        monster_ground_rush_penalty: float = -3.0,
        monster_airborne_jump_penalty: float = -3.0,
        wisp_danger_penalty: float = -3.0,
        wisp_radius: float = 6.0,
        evasion_bonus: float = 1.5,
    ):
        self.weight_progress = weight_progress
        self.weight_regress = weight_regress
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self.pit_penalty = pit_penalty if pit_penalty is not None else death_penalty
        self.ghost_penalty = ghost_penalty if ghost_penalty is not None else death_penalty
        self.fire_penalty = fire_penalty if fire_penalty is not None else death_penalty
        self.laser_penalty = laser_penalty if laser_penalty is not None else death_penalty
        self.wisp_penalty = wisp_penalty if wisp_penalty is not None else death_penalty
        self.monster_penalty = monster_penalty if monster_penalty is not None else death_penalty
        self.obstacle_penalty = obstacle_penalty
        self.item_bonus = item_bonus
        self.step_penalty = step_penalty

        # Jump penalty configuration (supporting both unnecessary_jump_penalty and jump_cost alias)
        if jump_cost is not None:
            self.unnecessary_jump_penalty = jump_cost
        else:
            self.unnecessary_jump_penalty = unnecessary_jump_penalty
        self.jump_penalty = self.unnecessary_jump_penalty
        self.jump_cost = self.unnecessary_jump_penalty

        self.timeout_penalty = timeout_penalty
        self.use_max_x = use_max_x
        self._max_x_reached: Optional[float] = None

        self.obs_threat_distance = obs_threat_distance
        self.pit_threat_distance = pit_threat_distance
        self.runway_pit_distance = runway_pit_distance
        self.ghost_threat_distance = ghost_threat_distance
        self.laser_threat_distance = laser_threat_distance
        self.wisp_threat_distance = wisp_threat_distance
        self.monster_threat_distance = monster_threat_distance
        self.ghost_height = ghost_height
        self.ghost_clearance_margin = ghost_clearance_margin

        # Timing guidance penalties & bonuses (Levels 4-8)
        self.laser_danger_penalty = laser_danger_penalty
        self.laser_lookahead_ticks = laser_lookahead_ticks
        self.ghost_advance_penalty = ghost_advance_penalty
        self.monster_ground_rush_penalty = monster_ground_rush_penalty
        self.monster_airborne_jump_penalty = monster_airborne_jump_penalty
        self.wisp_danger_penalty = wisp_danger_penalty
        self.wisp_radius = wisp_radius
        self.evasion_bonus = evasion_bonus

    def reset(self) -> None:
        """Reset internal episode tracking (frontier max_x)."""
        self._max_x_reached = None

    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        # Initialize or re-sync frontier max_x on new episode or coordinate leap
        if self._max_x_reached is None or prev_state.player_x < (self._max_x_reached - 500.0):
            self._max_x_reached = prev_state.player_x

        # 1. Horizontal progress / regression shaping
        if self.use_max_x:
            frontier_delta = max(0.0, next_state.player_x - self._max_x_reached)
            reward = self.weight_progress * frontier_delta
            if next_state.player_x > self._max_x_reached:
                self._max_x_reached = next_state.player_x
            elif events.dx < 0.0:
                # Regressing backward is penalized to prevent infinite pacing/stalling loops
                reward += self.weight_regress * events.dx
        else:
            if events.dx > 0.0:
                reward = self.weight_progress * events.dx
            else:
                reward = self.weight_regress * events.dx

        # 2. Terminal Goal Bonus
        if events.reached_goal:
            reward += self.goal_bonus

        # 3. Unified Death Check & Penalty
        is_death = (
            getattr(events, "died", False)
            or getattr(events, "is_death", False)
            or events.fell_in_pit
            or events.hit_ghost
            or getattr(events, "hit_fire", False)
            or getattr(events, "hit_laser", False)
            or getattr(events, "hit_wisp", False)
            or getattr(events, "hit_monster", False)
        )
        if is_death:
            if events.fell_in_pit and self.pit_penalty != self.death_penalty:
                reward += self.pit_penalty
            elif events.hit_ghost and self.ghost_penalty != self.death_penalty:
                reward += self.ghost_penalty
            elif getattr(events, "hit_fire", False) and self.fire_penalty != self.death_penalty:
                reward += self.fire_penalty
            elif getattr(events, "hit_laser", False) and self.laser_penalty != self.death_penalty:
                reward += self.laser_penalty
            elif getattr(events, "hit_wisp", False) and self.wisp_penalty != self.death_penalty:
                reward += self.wisp_penalty
            elif getattr(events, "hit_monster", False) and self.monster_penalty != self.death_penalty:
                reward += self.monster_penalty
            else:
                reward += self.death_penalty

        # 4. Obstacle Collision Penalty (punish running into walls on ground; do not penalize upward leaps clearing the top)
        if events.hit_obstacle and (prev_state.is_grounded or prev_state.velocity_y >= 0.0):
            reward += self.obstacle_penalty

        # 5. Item Collection Bonus
        if events.collected_item:
            reward += self.item_bonus

        # if not prev_state.is_grounded and getattr(prev_state, "can_double_jump", False):
        #     dist_pit = getattr(prev_state, "dist_to_pit", -1.0)
        #     if dist_pit == 0.0:
        #         if action in (Action.JUMP, Action.RIGHT_JUMP):
        #             reward += self.double_jump_bonus

        # 6. Constant living step cost
        reward += self.step_penalty

        # 7. Takeoff-only Jump Penalty (prevents bunny-hopping & jumping into overhead ghosts)
        # ONLY evaluated when grounded and taking off (never in mid-air physics trajectory!)
        if prev_state.is_grounded and action in (Action.JUMP, Action.RIGHT_JUMP):
            obs_dy = getattr(prev_state, "obstacle_dy", 0.0)
            obs_height = getattr(prev_state, "obstacle_height", 0.0)
            obs_overhead_cleared = (
                obs_dy < 0.0
                and (obs_dy + obs_height) <= -self.ghost_clearance_margin
            )
            obs_far = (
                prev_state.dist_to_obstacle < 0.0
                or prev_state.dist_to_obstacle > self.obs_threat_distance
                or obs_overhead_cleared
            )
            pit_far = (
                prev_state.dist_to_pit < 0.0
                or prev_state.dist_to_pit > self.pit_threat_distance
            )
            dist_ghost = getattr(prev_state, "dist_to_ghost", -1.0)
            ghost_y = getattr(prev_state, "ghost_y", 0.0)
            ghost_dy = getattr(prev_state, "ghost_dy", 0.0)
            player_y = getattr(prev_state, "player_y", getattr(prev_state, "mario_y", 288.0))

            if ghost_y > 0.0:
                ghost_overhead_cleared = (
                    (ghost_y + self.ghost_height) <= (player_y - self.ghost_clearance_margin)
                )
            else:
                ghost_overhead_cleared = (
                    ghost_dy < 0.0
                    and (ghost_dy + self.ghost_height) <= -self.ghost_clearance_margin
                )
            low_ghost_near = (
                0.0 <= dist_ghost <= self.ghost_threat_distance
                and not ghost_overhead_cleared
            )
            laser_near = 0.0 <= getattr(prev_state, "dist_to_laser", -1.0) <= self.laser_threat_distance
            dist_wisp = getattr(prev_state, "dist_to_wisp", -1.0)
            wisp_y = getattr(prev_state, "wisp_y", 0.0)
            wisp_dy = getattr(prev_state, "wisp_dy", 0.0)
            if wisp_y > 0.0:
                wisp_overhead_cleared = (
                    (wisp_y + self.wisp_radius) <= (player_y - self.ghost_clearance_margin)
                )
            else:
                wisp_overhead_cleared = (
                    wisp_dy < 0.0
                    and (wisp_dy + self.wisp_radius) <= -self.ghost_clearance_margin
                )
            wisp_near = 0.0 <= dist_wisp <= self.wisp_threat_distance and not wisp_overhead_cleared
            monster_near = 0.0 <= getattr(prev_state, "dist_to_monster", -1.0) <= self.monster_threat_distance

            # If no hazard justifies a jump (or ghost is hovering overhead), penalize takeoff
            if obs_far and pit_far and not low_ghost_near and not laser_near and not wisp_near and not monster_near:
                reward += self.unnecessary_jump_penalty

        # 8. Dynamic Timing & Kinematics Guidance (Levels 4-8):
        if not is_death:
            # A. Rotating Laser Timing Guidance:
            # Penalize rushing forward when laser is actively sweeping downward through the ground plane,
            # blocking the path, or about to close the skyward window (including continuous tracking when dist_to_laser == -1.0).
            dist_laser = getattr(prev_state, "dist_to_laser", -1.0)
            laser_angle = getattr(prev_state, "laser_angle", -1.0)
            laser_rot_speed = getattr(prev_state, "laser_angular_vel", 0.0)
            if laser_rot_speed == 0.0:
                laser_rot_speed = getattr(prev_state, "laser_rot_speed", 0.0)
            laser_blocking = getattr(prev_state, "laser_is_blocking", False)
            laser_ghost_dx = getattr(prev_state, "laser_ghost_dx", 0.0)
            laser_length = getattr(prev_state, "laser_length", 0.0)
            laser_angle_sin = getattr(prev_state, "laser_angle_sin", 0.0)
            laser_angle_cos = getattr(prev_state, "laser_angle_cos", 1.0)

            laser_in_range = (0.0 <= dist_laser <= self.laser_threat_distance) or (
                laser_length > 0.0 and 0.0 <= laser_ghost_dx <= self.laser_threat_distance
            )
            if laser_in_range:
                laser_sweeping_floor = laser_blocking
                laser_closing_window = False
                has_sin_cos = (laser_angle_sin != 0.0) or (laser_angle_cos != 1.0) or (laser_angle == 0.0)
                if has_sin_cos and (dist_laser >= 0.0 or laser_length > 0.0):
                    norm_angle = math.atan2(laser_angle_sin, laser_angle_cos) % (2.0 * math.pi)
                elif laser_angle >= 0.0:
                    norm_angle = laser_angle % (2.0 * math.pi)
                else:
                    norm_angle = -1.0

                if norm_angle >= 0.0:
                    two_pi = 2.0 * math.pi
                    pi = math.pi
                    if dist_laser >= 0.0:
                        laser_sweeping_floor = laser_sweeping_floor or (norm_angle < pi)
                    next_angle = (norm_angle + laser_rot_speed * self.laser_lookahead_ticks) % two_pi
                    laser_closing_window = (norm_angle >= pi) and (laser_rot_speed != 0.0) and (next_angle < pi)
                if laser_sweeping_floor or laser_closing_window:
                    if action in (Action.RIGHT, Action.RIGHT_JUMP):
                        reward += self.laser_danger_penalty

            # B. Patrolling Ghost over Pit Guidance:
            # Targets the compound hazard where a ghost patrols on the takeoff runway of an upcoming pit:
            # Penalize leaping over pit into an oncoming ghost advancing directly towards Hoá;
            # Reward waiting on the runway until the ghost turns around, then leaping across cleanly.
            dist_ghost = getattr(prev_state, "dist_to_ghost", -1.0)
            ghost_vx = getattr(prev_state, "ghost_vx", 0.0)
            dist_pit = getattr(prev_state, "dist_to_pit", -1.0)
            if 0.0 <= dist_ghost <= self.ghost_threat_distance and 0.0 <= dist_pit <= self.runway_pit_distance:
                if ghost_vx < 0.0:
                    # Ghost advancing directly towards Hoá
                    if action in (Action.JUMP, Action.RIGHT_JUMP):
                        reward += self.ghost_advance_penalty
                elif ghost_vx > 0.0:
                    # Ghost retreating away from Hoá -> runway clear to leap!
                    if action in (Action.JUMP, Action.RIGHT_JUMP):
                        reward += self.evasion_bonus

            # C. Grave Monster Velocity & Type Guidance (Quy Nhập Tràng & Cương Thi):
            dist_monster = getattr(prev_state, "dist_to_monster", -1.0)
            monster_grounded = getattr(prev_state, "monster_is_grounded", True)
            monster_vx = getattr(prev_state, "monster_vx", 0.0)
            monster_vy = getattr(prev_state, "monster_vy", 0.0)
            monster_type = getattr(prev_state, "monster_type", None)
            if 0.0 <= dist_monster <= self.monster_threat_distance:
                # 1. Quy Nhập Tràng (fast ground sprint demon):
                # Staying flat on ground or walking into sprint is fatal; leaping over is required!
                if monster_type in ("quy_nhap_trang", 1) or (monster_vx < -1.0 and monster_vy >= 0.0):
                    if prev_state.is_grounded and action in (Action.STAY, Action.CROUCH, Action.RIGHT):
                        reward += self.monster_ground_rush_penalty
                    elif action in (Action.JUMP, Action.RIGHT_JUMP):
                        reward += self.evasion_bonus

                # 2. Cương Thi (Jiangshi / hopping vampire):
                elif monster_type in ("cuong_thi", 2) or (not monster_grounded) or (monster_vy < -1.0):
                    if (not monster_grounded) or (monster_vy < -1.0):
                        # Airborne hop: jumping up into it collides mid-air; wait/stay!
                        if action in (Action.JUMP, Action.RIGHT_JUMP):
                            reward += self.monster_airborne_jump_penalty
                    else:
                        # Paused on ground: jumping over is safe
                        if prev_state.is_grounded and action in (Action.JUMP, Action.RIGHT_JUMP):
                            reward += self.evasion_bonus

            # D. Fire Pit & Wide-Pit Timing Guidance:
            # Penalize jumping or walking into an active, warning, or imminent flame;
            # Encourage grounded jump takeoff for wide pits (>70px).
            dist_pit = getattr(prev_state, "dist_to_pit", -1.0)
            pit_width = getattr(prev_state, "pit_width", 0.0)
            pit_fire_state = getattr(prev_state, "pit_fire_state", 0)
            pit_countdown = getattr(prev_state, "pit_fire_countdown", 0.0)
            if 0.0 <= dist_pit <= self.pit_threat_distance:
                fire_imminent = (pit_fire_state in (1, 2)) or (
                    pit_fire_state == 0 and 0.0 < pit_countdown <= 15.0 and pit_width > 70.0
                )
                if fire_imminent:
                    if action in (Action.RIGHT, Action.RIGHT_JUMP, Action.JUMP):
                        reward += self.laser_danger_penalty
                elif pit_width > 70.0 and prev_state.is_grounded and dist_pit <= 35.0:
                    if action == Action.RIGHT_JUMP:
                        reward += self.evasion_bonus
                    elif action == Action.RIGHT and dist_pit <= 12.0:
                        reward += self.laser_danger_penalty

            # E. Will-o'-the-Wisp Velocity, Altitude & Casting Phase Guidance:
            dist_wisp = getattr(prev_state, "dist_to_wisp", -1.0)
            wisp_dir = getattr(prev_state, "wisp_dir", 0)
            wisp_vx = getattr(prev_state, "wisp_vx", 0.0)
            wisp_y = getattr(prev_state, "wisp_y", 0.0)
            wisp_dy = getattr(prev_state, "wisp_dy", 0.0)
            wisp_grounded = getattr(prev_state, "wisp_is_grounded", False)
            wisp_ghost_dx = getattr(prev_state, "wisp_ghost_dx", 0.0)
            wisp_charging = getattr(prev_state, "wisp_charging", False) or (
                getattr(prev_state, "wisp_ghost_charging", False) and 0.0 <= wisp_ghost_dx <= 150.0
            )
            player_y = getattr(prev_state, "player_y", getattr(prev_state, "mario_y", 288.0))
            if wisp_y > 0.0:
                wisp_overhead = (wisp_y + self.wisp_radius) <= (player_y - self.ghost_clearance_margin)
            else:
                wisp_overhead = (wisp_dy < 0.0) and ((wisp_dy + self.wisp_radius) <= -self.ghost_clearance_margin)

            if 0.0 <= dist_wisp <= self.wisp_threat_distance:
                if wisp_dir < 0 and wisp_vx > 0.0:
                    # Chasing from behind: standing still risks getting hit from the left
                    if prev_state.is_grounded and action in (Action.STAY, Action.CROUCH):
                        reward += self.wisp_danger_penalty
                    elif action in (Action.RIGHT, Action.RIGHT_JUMP):
                        reward += self.evasion_bonus
                elif (not wisp_grounded) and wisp_overhead:
                    # High parabolic arc overhead: jumping into it is fatal
                    if action in (Action.JUMP, Action.RIGHT_JUMP):
                        reward += self.wisp_danger_penalty
                elif wisp_grounded or wisp_vx < 0.0:
                    # Low/ground-rolling wisp ahead: leap over instead of walking/standing into it
                    if prev_state.is_grounded and action in (Action.STAY, Action.CROUCH, Action.RIGHT):
                        reward += self.wisp_danger_penalty
                    elif action in (Action.JUMP, Action.RIGHT_JUMP):
                        reward += self.evasion_bonus
            elif wisp_charging and action == Action.RIGHT_JUMP:
                # WispGhost telegraphing an imminent cast ahead: avoid leaping blindly into the spawn
                reward += 0.5 * self.wisp_danger_penalty

        # 9. Timeout Penalty
        if done and not events.reached_goal and not is_death:
            reward += self.timeout_penalty

        if done:
            self._max_x_reached = None

        return float(reward)
