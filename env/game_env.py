import math
from typing import List, Optional, Tuple
import pygame

from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from env.entities import (
    Ghost, LaserGhost, WispGhost, WillOWisp, GoalFlag, Obstacle, Pit, Player, WarmthItem,
    GraveMonster, GraveMonsterState, QuyNhapTrang, CuongThi
)
from .game_renderer import GameRenderer

class GameEnvironment(BaseEnvironment):
    """
    2D Midnight Graveyard platformer simulation engine.
    Pure physics, collision detection, and MDP state extraction.
    """
    # Physics constants
    GRAVITY: float = 0.8
    WALK_SPEED: float = 4.0
    JUMP_FORCE: float = -13.0
    DOUBLE_JUMP_RATIO: float = 0.75
    SLIDE_INITIAL_SPEED: float = 6.8
    SLIDE_FRICTION: float = 0.90
    SLIDE_MIN_CRAWL_SPEED: float = 2.5
    STANDING_HEIGHT: float = 32.0
    CROUCH_HEIGHT: float = 18.0

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        pits: Optional[List[Pit]] = None,
        ghosts: Optional[List[Ghost]] = None,
        monsters: Optional[List[GraveMonster]] = None,
        items: Optional[List[WarmthItem]] = None,
        goal: Optional[GoalFlag] = None,
        floor_y: float = 320.0,
        max_steps: int = 1000,
        level_name: str = "Graveyard",
        level_id: int = 1,
        canvas_width: int = 800,
        canvas_height: int = 400,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.floor_y = floor_y
        self.max_steps = max_steps
        self.level_name = level_name
        self.level_id = level_id

        # Level entities
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else []
        self.pits: List[Pit] = pits if pits is not None else []
        self.ghosts: List[Ghost] = ghosts if ghosts is not None else []
        self.monsters: List[GraveMonster] = monsters if monsters is not None else []
        self.items: List[WarmthItem] = items if items is not None else []
        self.wisps: List[WillOWisp] = []
        self.goal: GoalFlag = goal if goal is not None else GoalFlag(
            x=720.0, y=floor_y - 88.0
        )


        # Player entity (Hoá)
        self.start_x: float = 50.0
        self.start_y: float = floor_y - 32.0  # floor_y - mario.height
        self.player = Player(x=self.start_x, y=self.start_y)

        # Episode tracking
        self.step_count: int = 0
        self.god_mode: bool = False

        # UI Renderer (Lazy initialized on first render call)
        self._renderer: Optional[GameRenderer] = None

    @property
    def is_running(self) -> bool:
        """Returns True if the Pygame rendering window is active, False otherwise."""
        return self._renderer is not None and self._renderer._screen is not None

    def reset(self) -> RawState:
        """Reset Hoá, ghosts, warmth items, and step counter."""
        self.player.reset(self.start_x, self.start_y)
        for ghost in self.ghosts:
            ghost.reset()
        for monster in self.monsters: # <-- Add here
            monster.reset()
        for item in self.items:
            item.reset()
        self.wisps.clear()
        self.step_count = 0
        return self._get_raw_state()

    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
        """Run one physics tick for the given discrete action."""
        self.step_count += 1
        old_x = self.player.x
        was_grounded = self.player.is_grounded
        events = GameEvents()

        # --------------------------------------------------------
        # Horizontal Movement, Crouch, Slide, and X-Axis Collision
        # --------------------------------------------------------
        # 1. Check whether an overhead obstacle prevents standing up
        standing_test_rect = (
            self.player.x,
            self.player.bottom - self.STANDING_HEIGHT,
            self.player.width,
            self.STANDING_HEIGHT,
        )
        has_low_ceiling = any(self._rects_overlap(standing_test_rect, obs.rect) for obs in self.obstacles)
        # 2. Player must remain crouched if: pressing CROUCH while grounded OR trapped under a low ceiling
        must_crouch = (action == Action.CROUCH and self.player.is_grounded) or (self.player.is_crouching and has_low_ceiling)
        if must_crouch:
            was_moving = abs(self.player.vx) > 0.1
            facing = 1.0 if self.player.facing_right else -1.0

            if not self.player.is_crouching:
                # Transition standing -> crouching: lower hitbox height
                self.player.is_crouching = True
                self.player.y += (self.STANDING_HEIGHT - self.CROUCH_HEIGHT)
                self.player.height = self.CROUCH_HEIGHT
                
                # Only initiate momentum slide if already running; stationary duck keeps vx = 0
                self.player.vx = facing * self.SLIDE_INITIAL_SPEED if was_moving else 0.0
            else:
                if has_low_ceiling:
                    # Inside low tunnel: crawl if directional keys are held, or decay momentum
                    if action in (Action.RIGHT, Action.RIGHT_JUMP):
                        self.player.facing_right = True
                        self.player.vx = max(self.player.vx * self.SLIDE_FRICTION, self.SLIDE_MIN_CRAWL_SPEED)
                    elif action == Action.LEFT:
                        self.player.facing_right = False
                        self.player.vx = min(self.player.vx * self.SLIDE_FRICTION, -self.SLIDE_MIN_CRAWL_SPEED)
                    else:
                        # Released controls: decelerate to a complete stop
                        self.player.vx *= self.SLIDE_FRICTION
                        if abs(self.player.vx) < 0.2:
                            self.player.vx = 0.0
                else:
                    # Outdoors: momentum slide decays to stop
                    self.player.vx *= self.SLIDE_FRICTION
                    if abs(self.player.vx) < 0.2:
                        self.player.vx = 0.0
        else:
            # Outdoors and NOT crouching: stand upright (Slide Cancel)
            if self.player.is_crouching:
                self.player.is_crouching = False
                self.player.y -= (self.STANDING_HEIGHT - self.CROUCH_HEIGHT)
                self.player.height = self.STANDING_HEIGHT
            # Standard outdoor walking / running
            if action in (Action.RIGHT, Action.RIGHT_JUMP):
                self.player.vx = self.WALK_SPEED
                self.player.facing_right = True
            elif action == Action.LEFT:
                self.player.vx = -self.WALK_SPEED
                self.player.facing_right = False
            elif action in (Action.STAY, Action.JUMP):
                self.player.vx = 0.0
            
        
        self.player.x += self.player.vx

        # Prevent walking off the left screen boundary
        if self.player.x < 0.0:
            self.player.x = 0.0
            self.player.vx = 0.0

        # Resolve side collisions with obstacles
        for obs in self.obstacles:
            if self._rects_overlap(self.player.rect, obs.rect):
                events.hit_obstacle = True
                if self.player.vx > 0:  # Moving right -> snap to left edge of obstacle
                    self.player.x = obs.x - self.player.width
                    self.player.vx = 0.0
                elif self.player.vx < 0:  # Moving left -> snap to right edge of obstacle
                    self.player.x = obs.right
                    self.player.vx = 0.0

        # -----------------------------------------------------
        # Vertical Movement (Jump + Gravity) & Y-Axis Collision
        # -----------------------------------------------------
        if action in (Action.JUMP, Action.RIGHT_JUMP) and not self.player.is_crouching:
            # RULE / GUARDRAIL: DO NOT UNCOMMENT OR BOOST JUMP FORCE IN GOD MODE:
            # # jump_impulse = self.JUMP_FORCE * 1.35 if getattr(self, "god_mode", False) else self.JUMP_FORCE
            # God Mode MUST strictly preserve standard JUMP_FORCE so level playability / human reach testing remains 100% fair.

            if self.player.is_grounded:
                # 1. First ground jump: take off and ENABLE double jump for mid-air!
                self.player.vy = self.JUMP_FORCE
                self.player.is_grounded = False
                self.player.can_double_jump = True
            elif self.player.can_double_jump or getattr(self, "god_mode", False):
                # 2. Mid-air double jump (and infinite airborne flutter leaps in God Mode!)
                self.player.vy = self.JUMP_FORCE * self.DOUBLE_JUMP_RATIO
                if not getattr(self, "god_mode", False):
                    self.player.can_double_jump = False
            
        self.player.vy += self.GRAVITY
        old_bottom = self.player.bottom
        self.player.y += self.player.vy
        self.player.is_grounded = False

        # Resolve top/bottom collisions with obstacles
        for obs in self.obstacles:
            if self._rects_overlap(self.player.rect, obs.rect):
                if self.player.vy > 0:  # Falling onto top of obstacle
                    self.player.y = obs.y - self.player.height
                    self.player.vy = 0.0
                    self.player.is_grounded = True
                    self.player.can_double_jump = False
                elif self.player.vy < 0:  # Hitting bottom of obstacle
                    self.player.y = obs.bottom
                    self.player.vy = 0.0

        # Resolve floor collision (only if NOT standing over a pit)
        over_pit = any(pit.contains(self.player.x, self.player.width) for pit in self.pits)
        if not over_pit and old_bottom <= self.floor_y and self.player.bottom >= self.floor_y:
            self.player.y = self.floor_y - self.player.height
            self.player.vy = 0.0
            self.player.is_grounded = True
            self.player.can_double_jump = False     # Landed on ground -> False

        if was_grounded and not self.player.is_grounded:
            self.player.can_double_jump = True

        self._resolve_pit_wall_collisions()

        # -------------------------------------------
        # Update Ghosts and Wisps
        # -------------------------------------------
        for ghost in self.ghosts:
            ghost.update()
            # If ghost is a WispGhost and it's time to cast, spawn a projectile
            if isinstance(ghost, WispGhost) and ghost.can_cast_wisp(self.step_count):
                hx, hy, vx, vy = ghost.get_wisp_launch_params(self.step_count)
                self.wisps.append(WillOWisp(x=hx, y=hy, vx=vx, vy=vy))

        # Update all active will-o'-the-wisps (airborne arc -> ground roll -> despawn)
        for wisp in self.wisps:
            wisp.update(self.pits, self.floor_y, self.canvas_height)
        self.wisps = [w for w in self.wisps if w.is_alive]

        # Update active grave monsters (rising from graves, leaping over pits, pursuing left)
        for monster in self.monsters:
            monster.update(self.player.x, self.floor_y, self.pits)

        # Check Warmth Item Pickups
        for item in self.items:
            if not item.is_collected and self._rects_overlap(self.player.rect, item.rect):
                item.is_collected = True
                events.collected_item = True

        # ----------------------------------------------
        # Record Events and Check Termination Conditions
        # ----------------------------------------------
        events.dx = self.player.x - old_x
        done = False

        # Check death and termination conditions (clean if / elif chain)
        hit_ghost = any(self._rects_overlap(self.player.rect, g.rect) for g in self.ghosts)
        hit_fire = any(
            self._rects_overlap(self.player.rect, flame_rect)
            for pit in self.pits
            if (flame_rect := pit.get_flame_rect(self.step_count, self.floor_y, self.canvas_height)) is not None
        )
        hit_laser = any(
            self._segment_intersects_rect(g.get_laser_segment(self.step_count), self.player.rect)
            for g in self.ghosts
            if isinstance(g, LaserGhost)
        )
        hit_wisp = any(self._rects_overlap(self.player.rect, w.rect) for w in self.wisps)

        collided_monster = next(
            (m for m in self.monsters
             if m.is_alive
             and m.state in (GraveMonsterState.EMERGING, GraveMonsterState.ACTIVE)
             and self._rects_overlap(self.player.rect, m.rect)),
            None
        )
        hit_monster = collided_monster is not None

        if not self.god_mode:
            # Death by hitting ghost
            if hit_ghost:
                self.player.is_alive = False
                events.hit_ghost = True
                done = True

            # Death by hitting fire
            elif hit_fire:
                self.player.is_alive = False
                events.hit_fire = True
                done = True

            # Death by hitting laser of the ghost
            elif hit_laser:
                self.player.is_alive = False
                events.hit_laser = True
                done = True

            # Death by hitting the will-o-wisp
            elif hit_wisp:
                self.player.is_alive = False
                events.hit_wisp = True
                done = True

            # Death by hitting grave monster
            elif hit_monster:
                self.player.is_alive = False
                events.hit_monster = True
                from .entities import CuongThi, QuyNhapTrang
                if isinstance(collided_monster, CuongThi):
                    events.monster_type = "cuong_thi"
                elif isinstance(collided_monster, QuyNhapTrang):
                    events.monster_type = "quy_nhap_trang"
                done = True

            # Pit death
            elif self.player.y > self.canvas_height:
                self.player.is_alive = False
                events.fell_in_pit = True
                done = True
        else:
            # In God Mode: if fall into the pit, the force push the player back
            if self.player.y > self.floor_y + 30.0:
                self.player.y = self.floor_y - self.player.height
                self.player.vy = -10.0      # Force of the pushing back

        # Goal reached
        if self.goal.is_reached(self.player):
            events.reached_goal = True
            done = True

        # Timeout
        elif self.step_count >= self.max_steps:
            done = True

        return self._get_raw_state(), events, done

    def render(self) -> None:
        """Delegates drawing to GameRenderer."""
        if self._renderer is None:
            self._renderer = GameRenderer(
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
            )
        self._renderer.render(self)

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def show_game_over(self, events: GameEvents) -> None:
        """Delegates displaying the victory or game over screen to the renderer if active."""
        if self._renderer is not None:
            self._renderer.show_game_over(self, events)
        
    # --------------
    # Helper Methods
    # --------------
    def _resolve_pit_wall_collisions(self) -> None:
        """Confines Hoá horizontally between the pit walls when falling below ground level."""
        if self.player.bottom > self.floor_y:
            for pit in self.pits:
                if self.player.x < pit.x_end and self.player.right > pit.x_start:
                    if self.player.x < pit.x_start:
                        self.player.x = pit.x_start
                        self.player.vx = 0.0
                    elif self.player.right > pit.x_end:
                        self.player.x = pit.x_end - self.player.width
                        self.player.vx = 0.0

    def _get_raw_state(self) -> RawState:
        """Build local-vision RawState for the agent/discretizer."""
        upcoming_obs = [obs for obs in self.obstacles if obs.right > self.player.x]
        if upcoming_obs:
            nearest_obs = min(upcoming_obs, key=lambda o: o.x)
            dist_to_obstacle = max(0.0, nearest_obs.x - self.player.right)
            obstacle_height = nearest_obs.height
            obstacle_dy = nearest_obs.y - self.player.y
            obstacle_width = nearest_obs.width
        else:
            dist_to_obstacle = -1.0
            obstacle_height = 0.0
            obstacle_dy = 0.0
            obstacle_width = 0.0

        # 1. Nearest upcoming pit & fire timing
        upcoming_pits = [pit for pit in self.pits if pit.x_end > self.player.x]
        if upcoming_pits:
            nearest_pit = min(upcoming_pits, key=lambda p: p.x_start)
            dist_to_pit = max(0.0, nearest_pit.x_start - self.player.right)
            pit_width = nearest_pit.x_end - nearest_pit.x_start
            pit_fire_state = nearest_pit.get_fire_phase(self.step_count)
            # Calculate fire cycle progress [0.0 -> 1.0] and countdown ticks
            if nearest_pit.has_fire:
                phase = (self.step_count + nearest_pit.fire_offset) % int(nearest_pit.FIRE_CYCLE)
                pit_fire_cycle = phase / nearest_pit.FIRE_CYCLE
                if phase < nearest_pit.FIRE_WARN_START:
                    pit_fire_countdown = nearest_pit.FIRE_WARN_START - phase  # Ticks until smoke warning
                elif phase < nearest_pit.FIRE_ACTIVE_START:
                    pit_fire_countdown = nearest_pit.FIRE_ACTIVE_START - phase  # Ticks until flame bursts
                else:
                    pit_fire_countdown = nearest_pit.FIRE_CYCLE - phase  # Ticks until flame turns off
            else:
                pit_fire_cycle = 0.0
                pit_fire_countdown = 0.0
        else:
            dist_to_pit = -1.0
            pit_width = 0.0
            pit_fire_state = 0
            pit_fire_cycle = 0.0
            pit_fire_countdown = 0.0
        pit_fire_sin = math.sin(2.0 * math.pi * pit_fire_cycle)
        pit_fire_cos = math.cos(2.0 * math.pi * pit_fire_cycle)

        # 2. Nearest upcoming ghost & kinematic velocities
        upcoming_ghosts = [g for g in self.ghosts if g.right > self.player.x]
        if upcoming_ghosts:
            nearest_ghost = min(upcoming_ghosts, key=lambda g: g.x)
            dist_to_ghost = max(0.0, nearest_ghost.x - self.player.right)
            ghost_y = nearest_ghost.y
            ghost_vx = nearest_ghost.speed
            # Vy of sinusoidal floating wave: d/dt [A * sin(phase)] = A * float_speed * cos(phase)
            ghost_vy = nearest_ghost.float_amplitude * nearest_ghost.float_speed * math.cos(nearest_ghost._phase)
            ghost_dy = nearest_ghost.y - self.player.y
            ghost_float_amplitude = nearest_ghost.float_amplitude
            ghost_float_speed = nearest_ghost.float_speed
            ghost_phase_sin = math.sin(nearest_ghost._phase)
            ghost_phase_cos = math.cos(nearest_ghost._phase)
            ghost_dist_to_left_bound = nearest_ghost.x - nearest_ghost.patrol_min_x
            ghost_dist_to_right_bound = nearest_ghost.patrol_max_x - nearest_ghost.right
        else:
            dist_to_ghost = -1.0
            ghost_y = 0.0
            ghost_vx = 0.0
            ghost_vy = 0.0
            ghost_dy = 0.0
            ghost_float_amplitude = 0.0
            ghost_float_speed = 0.0
            ghost_phase_sin = 0.0
            ghost_phase_cos = 1.0
            ghost_dist_to_left_bound = 0.0
            ghost_dist_to_right_bound = 0.0

        # 3. Nearest upcoming rotating laser hazard (continuous tracking even when pointed upward)
        dist_to_laser = -1.0
        laser_angle = 0.0
        laser_is_blocking = False
        laser_ghost_dx = 0.0
        laser_ghost_dy = 0.0
        laser_angle_sin = 0.0
        laser_angle_cos = 1.0
        laser_angular_vel = 0.0
        laser_length = 0.0
        laser_sweep_amplitude = 0.0

        upcoming_lasers = [
            g for g in self.ghosts
            if isinstance(g, LaserGhost) and (g.x + g.width / 2.0 + g.laser_length) > self.player.x
        ]
        if upcoming_lasers:
            nearest_laser_ghost = min(upcoming_lasers, key=lambda g: g.x)
            eye_cx = nearest_laser_ghost.x + nearest_laser_ghost.width / 2.0
            eye_cy = nearest_laser_ghost.y + nearest_laser_ghost.height * 0.4
            laser_ghost_dx = eye_cx - self.player.right
            laser_ghost_dy = eye_cy - self.player.y
            laser_length = nearest_laser_ghost.laser_length
            laser_sweep_amplitude = nearest_laser_ghost.laser_sweep_amplitude
            if nearest_laser_ghost.laser_sweep_amplitude > 0.0:
                laser_angle = (
                    nearest_laser_ghost.laser_initial_angle
                    + nearest_laser_ghost.laser_sweep_amplitude
                    * math.sin(self.step_count * nearest_laser_ghost.laser_rotation_speed)
                )
                laser_angular_vel = (
                    nearest_laser_ghost.laser_sweep_amplitude
                    * nearest_laser_ghost.laser_rotation_speed
                    * math.cos(self.step_count * nearest_laser_ghost.laser_rotation_speed)
                )
            else:
                laser_angle = (
                    nearest_laser_ghost.laser_initial_angle
                    + self.step_count * nearest_laser_ghost.laser_rotation_speed
                ) % (2.0 * math.pi)
                laser_angular_vel = nearest_laser_ghost.laser_rotation_speed
            laser_angle_sin = math.sin(laser_angle)
            laser_angle_cos = math.cos(laser_angle)

        for g in upcoming_lasers:
            seg = g.get_laser_segment(self.step_count)
            if seg is None:
                continue

            (sx, sy), (ex, ey) = seg
            laser_min_x = min(sx, ex)
            laser_max_x = max(sx, ex)
            laser_min_y = min(sy, ey)
            laser_max_y = max(sy, ey)

            # Check dynamic 2D threat relative to Hoá's actual current position (standing, jumping, or sliding)
            laser_overlaps_player_y = (laser_max_y >= self.player.y) and (laser_min_y <= self.player.bottom)
            laser_is_ahead_or_overlapping = laser_max_x > self.player.x

            if laser_overlaps_player_y and laser_is_ahead_or_overlapping:
                dist = max(0.0, laser_min_x - self.player.right)
                if dist_to_laser < 0 or dist < dist_to_laser:
                    dist_to_laser = dist
                    if laser_max_y >= (self.floor_y - 48.0):
                        laser_is_blocking = True

        # 4. Nearest upcoming wisp & WispGhost casting
        threatening_wisps = []
        for w in self.wisps:
            if not w.is_alive:
                continue
            # 1. Wisp is ahead of Hoá (in front)
            if (w.x + w.radius) >= self.player.x:
                dist = max(0.0, (w.x - w.radius) - self.player.right)
                threatening_wisps.append((dist, 1, w))
            elif w.vx > 0:
                dist = max(0.0, self.player.x - (w.x + w.radius))
                threatening_wisps.append((dist, -1, w))
        if threatening_wisps:
            dist_to_wisp, wisp_dir, nearest_wisp = min(threatening_wisps, key=lambda item: item[0])
            wisp_vy = nearest_wisp.vy
            wisp_is_grounded = nearest_wisp.is_grounded
            wisp_dy = nearest_wisp.y - self.player.y
            wisp_vx = nearest_wisp.vx
            wisp_roll_ticks_left = nearest_wisp.roll_ticks if nearest_wisp.is_grounded else 0
        else:
            dist_to_wisp = -1.0
            wisp_dir = 0
            wisp_vy = 0.0
            wisp_is_grounded = False
            wisp_dy = 0.0
            wisp_vx = 0.0
            wisp_roll_ticks_left = 0

        # Nearest upcoming WispGhost ahead (telegraphing & cycle phase)
        upcoming_wisp_ghosts = [
            g for g in self.ghosts
            if isinstance(g, WispGhost) and g.right > self.player.x
        ]
        wisp_ghost_charging = any(g.is_charging_wisp(self.step_count) for g in upcoming_wisp_ghosts)
        if upcoming_wisp_ghosts:
            nearest_wg = min(upcoming_wisp_ghosts, key=lambda g: g.x)
            wg_phase = (self.step_count + nearest_wg.wisp_offset) % nearest_wg.wisp_cycle
            shot_number = (self.step_count + nearest_wg.wisp_offset) // nearest_wg.wisp_cycle
            next_shot_number = shot_number if wg_phase == 0 else shot_number + 1
            wisp_ghost_dx = nearest_wg.x - self.player.right
            wisp_ghost_dy = nearest_wg.y - self.player.y
            wisp_cast_progress = wg_phase / nearest_wg.wisp_cycle
            wisp_ticks_to_cast = float(nearest_wg.wisp_cycle - wg_phase)
            wisp_next_is_high = (
                nearest_wg.wisp_mode == "high"
                or (nearest_wg.wisp_mode == "alternating" and next_shot_number % 2 == 0)
            )
        else:
            wisp_ghost_dx = 0.0
            wisp_ghost_dy = 0.0
            wisp_cast_progress = 0.0
            wisp_ticks_to_cast = 0.0
            wisp_next_is_high = False

        # 5. Grave Monsters (Active, Emerging & Dormant Triggers)
        upcoming_active_monsters = [
            m for m in self.monsters
            if m.is_alive
            and m.state in (GraveMonsterState.EMERGING, GraveMonsterState.ACTIVE)
            and m.right > self.player.x
        ]
        if upcoming_active_monsters:
            nearest_monster = min(upcoming_active_monsters, key=lambda m: m.x)
            dist_to_monster = max(0.0, nearest_monster.x - self.player.right)
            monster_y = nearest_monster.y
            monster_state = nearest_monster.state
            monster_vx = nearest_monster.vx
            monster_vy = nearest_monster.vy
            monster_dy = nearest_monster.y - self.player.y
            monster_is_sentry = nearest_monster.patrol_min_x is not None
            monster_pause_progress = 0.0
            monster_speed_ratio = 0.0
            if isinstance(nearest_monster, QuyNhapTrang):
                monster_type = 1
                monster_is_grounded = not nearest_monster._is_jumping
                speed_span = nearest_monster.max_speed - nearest_monster.base_speed
                if speed_span > 0.0:
                    monster_speed_ratio = (nearest_monster._current_speed - nearest_monster.base_speed) / speed_span
            elif isinstance(nearest_monster, CuongThi):
                monster_type = 2
                monster_is_grounded = nearest_monster._is_grounded
                if nearest_monster._is_grounded and nearest_monster.pause_duration > 0:
                    monster_pause_progress = nearest_monster._pause_timer / nearest_monster.pause_duration
            else:
                monster_type = 0
                monster_is_grounded = True

            left_pits = [nearest_monster.x - p.x_end for p in self.pits if p.x_end <= nearest_monster.x]
            monster_dist_to_pit = min(left_pits) if left_pits else -1.0
        else:
            dist_to_monster = -1.0
            monster_y = 0.0
            monster_state = GraveMonsterState.DORMANT
            monster_vx = 0.0
            monster_vy = 0.0
            monster_dy = 0.0
            monster_is_grounded = True
            monster_type = 0
            monster_is_sentry = False
            monster_pause_progress = 0.0
            monster_speed_ratio = 0.0
            monster_dist_to_pit = -1.0

        # Check proximity to dormant grave triggers (sleeping monster ambush)
        dormant_monsters = [
            m for m in self.monsters
            if m.is_alive and m.state == GraveMonsterState.DORMANT and m.x > self.player.x
        ]
        if dormant_monsters:
            nearest_dormant = min(dormant_monsters, key=lambda m: m.x)
            dist_to_monster_trigger = max(0.0, (nearest_dormant.x - nearest_dormant.trigger_dist) - self.player.right)
        else:
            dist_to_monster_trigger = -1.0

        # Nearest upcoming uncollected warmth item
        upcoming_items = [it for it in self.items if not it.is_collected and it.right > self.player.x]
        if upcoming_items:
            nearest_item = min(upcoming_items, key=lambda it: it.x)
            dist_to_item = max(0.0, nearest_item.x - self.player.right)
        else:
            dist_to_item = -1.0

        # Distance to Hoá's home
        dist_to_goal = max(0.0, self.goal.x - self.player.right)

        return RawState(
            player_x=self.player.x,
            player_y=self.player.y,
            velocity_x=self.player.vx,
            velocity_y=self.player.vy,
            is_grounded=self.player.is_grounded,
            can_double_jump=self.player.can_double_jump,
            is_crouching=self.player.is_crouching,
            dist_to_obstacle=dist_to_obstacle,
            obstacle_height=obstacle_height,
            dist_to_pit=dist_to_pit,
            pit_width=pit_width,
            pit_fire_state=pit_fire_state,
            pit_fire_cycle=pit_fire_cycle,
            pit_fire_countdown=pit_fire_countdown,
            dist_to_ghost=dist_to_ghost,
            ghost_y=ghost_y,
            ghost_vx=ghost_vx,
            ghost_vy=ghost_vy,
            dist_to_laser=dist_to_laser,
            laser_angle=laser_angle,
            laser_is_blocking=laser_is_blocking,
            dist_to_wisp=dist_to_wisp,
            wisp_dir=wisp_dir,
            wisp_vy=wisp_vy,
            wisp_is_grounded=wisp_is_grounded,
            wisp_ghost_charging=wisp_ghost_charging,
            dist_to_monster=dist_to_monster,
            monster_y=monster_y,
            monster_state=monster_state,
            monster_vx=monster_vx,
            monster_is_grounded=monster_is_grounded,
            dist_to_monster_trigger=dist_to_monster_trigger,
            dist_to_item=dist_to_item,
            dist_to_goal=dist_to_goal,
            obstacle_dy=obstacle_dy,
            obstacle_width=obstacle_width,
            pit_fire_sin=pit_fire_sin,
            pit_fire_cos=pit_fire_cos,
            ghost_dy=ghost_dy,
            ghost_float_amplitude=ghost_float_amplitude,
            ghost_float_speed=ghost_float_speed,
            ghost_phase_sin=ghost_phase_sin,
            ghost_phase_cos=ghost_phase_cos,
            ghost_dist_to_left_bound=ghost_dist_to_left_bound,
            ghost_dist_to_right_bound=ghost_dist_to_right_bound,
            laser_ghost_dx=laser_ghost_dx,
            laser_ghost_dy=laser_ghost_dy,
            laser_angle_sin=laser_angle_sin,
            laser_angle_cos=laser_angle_cos,
            laser_angular_vel=laser_angular_vel,
            laser_length=laser_length,
            laser_sweep_amplitude=laser_sweep_amplitude,
            wisp_ghost_dx=wisp_ghost_dx,
            wisp_ghost_dy=wisp_ghost_dy,
            wisp_cast_progress=wisp_cast_progress,
            wisp_ticks_to_cast=wisp_ticks_to_cast,
            wisp_next_is_high=wisp_next_is_high,
            wisp_dy=wisp_dy,
            wisp_vx=wisp_vx,
            wisp_roll_ticks_left=wisp_roll_ticks_left,
            monster_type=monster_type,
            monster_is_sentry=monster_is_sentry,
            monster_dy=monster_dy,
            monster_vy=monster_vy,
            monster_pause_progress=monster_pause_progress,
            monster_speed_ratio=monster_speed_ratio,
            monster_dist_to_pit=monster_dist_to_pit,
        )

    @staticmethod
    def _rects_overlap(
        r1: Tuple[float, float, float, float],
        r2: Tuple[float, float, float, float],
    ) -> bool:
        """Standard AABB (Axis-Aligned Bounding Box) intersection check."""
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2

        return (x1 < x2 + w2) and (x1 + w1 > x2) and (y1 < y2 + h2) and (y1 + h1 > y2)

    @staticmethod
    def _segment_intersects_rect(
        seg: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
        rect: Tuple[float, float, float, float],
    ) -> bool:
        """Checks if a 2D line segment intersects an AABB rectangle."""
        if seg is None:
            return False
        (x1, y1), (x2, y2) = seg
        rx, ry, rw, rh = rect
        r_right, r_bottom = rx + rw, ry + rh

        # Fast bounding box check 
        if max(x1, x2) < rx or min(x1, x2) > r_right or max(y1, y2) < ry or min(y1, y2) > r_bottom:
            return False

        # Check if either endpoint lies inside the rectangle
        if (rx <= x1 <= r_right and ry <= y1 <= r_bottom) or (rx <= x2 <= r_right and ry <= y2 <= r_bottom):
            return True

        # Check intersection against the 4 bounding box edges
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        p1, p2 = (x1, y1), (x2, y2)

        return (
            intersect(p1, p2, (rx, ry), (r_right, ry)) or
            intersect(p1, p2, (r_right, ry), (r_right, r_bottom)) or
            intersect(p1, p2, (rx, r_bottom), (r_right, r_bottom)) or
            intersect(p1, p2, (rx, ry), (rx, r_bottom))
        )