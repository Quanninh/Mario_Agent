from typing import List, Optional, Tuple
import pygame

from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from env.entities import Ghost, GoalFlag, Obstacle, Pit, Player, WarmthItem
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
    SLIDE_FRICTION: float = 0.90
    STANDING_HEIGHT: float = 32.0
    CROUCH_HEIGHT: float = 18.0

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        pits: Optional[List[Pit]] = None,
        ghosts: Optional[List[Ghost]] = None,
        items: Optional[List[WarmthItem]] = None,
        goal: Optional[GoalFlag] = None,
        floor_y: float = 320.0,
        max_steps: int = 1000,
        level_name: str = "Graveyard",
        canvas_width: int = 800,
        canvas_height: int = 400,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.floor_y = floor_y
        self.max_steps = max_steps
        self.level_name = level_name

        # Level entities
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else [
            Obstacle(x=250.0, y=floor_y - 48.0, width=32.0, height=48.0),
            Obstacle(x=500.0, y=floor_y - 64.0, width=32.0, height=64.0),
        ]
        self.pits: List[Pit] = pits if pits is not None else [
            Pit(x_start=360.0, x_end=420.0),
        ]
        self.ghosts: List[Ghost] = ghosts if ghosts is not None else []
        self.items: List[WarmthItem] = items if items is not None else []
        self.goal: GoalFlag = goal if goal is not None else GoalFlag(
            x=720.0, y=floor_y - 88.0
        )

        # Player entity (Hoá)
        self.start_x: float = 50.0
        self.start_y: float = floor_y - 32.0  # floor_y - mario.height
        self.player = Player(x=self.start_x, y=self.start_y)

        # Episode tracking
        self.step_count: int = 0

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
        for item in self.items:
            item.reset()
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
        if action == Action.CROUCH and self.player.is_grounded:
            if not self.player.is_crouching:
                # Transition standing -> crouching: lower head, keep feet on ground
                self.player.is_crouching = True
                self.player.y += (self.STANDING_HEIGHT - self.CROUCH_HEIGHT)
                self.player.height = self.CROUCH_HEIGHT
            self.player.vx *= self.SLIDE_FRICTION
            if abs(self.player.vx) < 0.3:
                self.player.vx = 0.0
        else:
            if self.player.is_crouching:
                # Transition crouching -> standing: raise head back up
                self.player.is_crouching = False
                self.player.y -= (self.STANDING_HEIGHT - self.CROUCH_HEIGHT)
                self.player.height = self.STANDING_HEIGHT

            # Update velocity for standard walking / jumping
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
        if action in (Action.JUMP, Action.RIGHT_JUMP):
            if self.player.is_grounded:
                # 1. First ground jump: take off and ENABLE double jump for mid-air!
                self.player.vy = self.JUMP_FORCE
                self.player.is_grounded = False
                self.player.can_double_jump = True
            elif self.player.can_double_jump:
                # 2. Mid-air double jump: second impulse and CONSUME double jump!
                self.player.vy = self.JUMP_FORCE * self.DOUBLE_JUMP_RATIO
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

        # -------------------------------------------
        # Update Ghosts and Check Warmth Item Pickups
        # -------------------------------------------
        for ghost in self.ghosts:
            ghost.update()

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

        # Pit death
        elif self.player.y > self.canvas_height:
            self.player.is_alive = False
            events.fell_in_pit = True
            done = True

        # Goal reached
        elif self.goal.is_reached(self.player):
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
    def _get_raw_state(self) -> RawState:
        """Build local-vision RawState for the agent/discretizer."""
        upcoming_obs = [obs for obs in self.obstacles if obs.right > self.player.x]
        if upcoming_obs:
            nearest_obs = min(upcoming_obs, key=lambda o: o.x)
            dist_to_obstacle = max(0.0, nearest_obs.x - self.player.right)
            obstacle_height = nearest_obs.height
        else:
            dist_to_obstacle = -1.0
            obstacle_height = 0.0

        # Nearest upcoming pit
        upcoming_pits = [pit for pit in self.pits if pit.x_end > self.player.x]
        if upcoming_pits:
            nearest_pit = min(upcoming_pits, key=lambda p: p.x_start)
            dist_to_pit = max(0.0, nearest_pit.x_start - self.player.right)
            pit_fire_state = nearest_pit.get_fire_phase(self.step_count)
        else:
            dist_to_pit = -1.0
            pit_fire_state = 0

        # Nearest upcoming ghost
        upcoming_ghosts = [g for g in self.ghosts if g.right > self.player.x]
        if upcoming_ghosts:
            nearest_ghost = min(upcoming_ghosts, key=lambda g: g.x)
            dist_to_ghost = max(0.0, nearest_ghost.x - self.player.right)
            ghost_y = nearest_ghost.y
        else:
            dist_to_ghost = -1.0
            ghost_y = 0.0

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
            velocity_x =self.player.vx,
            velocity_y=self.player.vy,
            is_grounded=self.player.is_grounded,
            can_double_jump=self.player.can_double_jump,
            is_crouching=self.player.is_crouching,
            dist_to_obstacle=dist_to_obstacle,
            obstacle_height=obstacle_height,
            dist_to_pit=dist_to_pit,
            pit_fire_state=pit_fire_state,
            dist_to_ghost=dist_to_ghost,
            ghost_y=ghost_y,
            dist_to_item=dist_to_item,
            dist_to_goal=dist_to_goal,
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
