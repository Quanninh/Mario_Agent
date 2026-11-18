from typing import List, Optional, Tuple
import pygame
from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from env.entities import GoalFlag, Mario, Obstacle, Pit

class MarioEnvironment(BaseEnvironment):
    """2D Mario platformer environment implementing BaseEnvironment."""

    # Physics constants
    GRAVITY: float = 0.8
    WALK_SPEED: float = 4.0
    JUMP_FORCE: float = -13.0

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        pits: Optional[List[Pit]] = None,
        goal: Optional[GoalFlag] = None,
        screen_width: int = 800,
        screen_height: int = 400,
        floor_y: float = 320.0,
        max_steps: int = 1000,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.floor_y = floor_y
        self.max_steps = max_steps

        # Default simple level if none provided (Step 4 will load custom levels)
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else [
            Obstacle(x=250.0, y=floor_y - 48.0, width=32.0, height=48.0),
            Obstacle(x=500.0, y=floor_y - 64.0, width=32.0, height=64.0),
        ]
        self.pits: List[Pit] = pits if pits is not None else [
            Pit(x_start=360.0, x_end=420.0),
        ]
        self.goal: GoalFlag = goal if goal is not None else GoalFlag(
            x=720.0, y=floor_y - 96.0
        )

        # Player entity
        self.start_x: float = 50.0
        self.start_y: float = floor_y - 32.0  # floor_y - mario.height
        self.mario = Mario(x=self.start_x, y=self.start_y)

        # Episode tracking
        self.step_count: int = 0

        # Pygame rendering state (lazy-initialized in render())
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

    def reset(self) -> RawState:
        """Reset Mario and episode counters to initial state."""
        self.mario.reset(self.start_x, self.start_y)
        self.step_count = 0
        return self._get_raw_state()

    def step(self, action: int) -> Tuple[RawState, GameEvents, bool]:
        """Run one physics tick for the given discrete action."""
        self.step_count += 1
        old_x = self.mario.x
        events = GameEvents()

        # --------------------------------------
        # Horizontal Movement & X-Axis Collision
        # --------------------------------------
        vx = 0.0
        if action in (Action.RIGHT, Action.RIGHT_JUMP):
            vx = self.WALK_SPEED
        elif action == Action.LEFT:
            vx = -self.WALK_SPEED

        self.mario.x += vx

        # Prevent walking off the left screen boundary
        if self.mario.x < 0.0:
            self.mario.x = 0.0

        # Resolve side collisions with obstacles
        for obs in self.obstacles:
            if self._rects_overlap(self.mario.rect, obs.rect):
                events.hit_obstacle = True
                if vx > 0:  # Moving right -> snap to left edge of obstacle
                    self.mario.x = obs.x - self.mario.width
                elif vx < 0:  # Moving left -> snap to right edge of obstacle
                    self.mario.x = obs.right

        # -----------------------------------------------------
        # Vertical Movement (Jump + Gravity) & Y-Axis Collision
        # -----------------------------------------------------
        if action in (Action.JUMP, Action.RIGHT_JUMP) and self.mario.is_grounded:
            self.mario.vy = self.JUMP_FORCE
            self.mario.is_grounded = False

        self.mario.vy += self.GRAVITY
        old_bottom = self.mario.bottom
        self.mario.y += self.mario.vy
        self.mario.is_grounded = False

        # Resolve top/bottom collisions with obstacles
        for obs in self.obstacles:
            if self._rects_overlap(self.mario.rect, obs.rect):
                if self.mario.vy > 0:  # Falling onto top of obstacle
                    self.mario.y = obs.y - self.mario.height
                    self.mario.vy = 0.0
                    self.mario.is_grounded = True
                elif self.mario.vy < 0:  # Hitting bottom of obstacle
                    self.mario.y = obs.bottom
                    self.mario.vy = 0.0

        # Resolve floor collision (only if NOT standing over a pit)
        over_pit = any(pit.contains(self.mario.x, self.mario.width) for pit in self.pits)
        if not over_pit and old_bottom <= self.floor_y and self.mario.bottom >= self.floor_y:
            self.mario.y = self.floor_y - self.mario.height
            self.mario.vy = 0.0
            self.mario.is_grounded = True

        # ----------------------------------------------
        # Record Events and Check Termination Conditions
        # ----------------------------------------------
        events.dx = self.mario.x - old_x
        done = False

        # Check pit death
        if self.mario.y > self.screen_height:
            self.mario.is_alive = False
            events.fell_in_pit = True
            done = True

        # Check goal flag reached
        elif self.goal.is_reached(self.mario):
            events.reached_goal = True
            done = True

        # Check episode timeout
        elif self.step_count >= self.max_steps:
            done = True

        return self._get_raw_state(), events, done

    def render(self) -> None:
        """Render the current game frame using Pygame."""
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Mario RL Environment")
            self._clock = pygame.time.Clock()

        # Process window quit events cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Camera offset (keeps Mario visible on wide levels)
        camera_x = max(0.0, self.mario.x - self.screen_width * 0.3)

        # Sky background
        self._screen.fill((135, 206, 235))

        # Draw ground (brown) and carve out pits (sky color)
        pygame.draw.rect(
            self._screen,
            (139, 69, 19),
            (0, self.floor_y, self.screen_width, self.screen_height - self.floor_y),
        )

        for pit in self.pits:
            pit_x = pit.x_start - camera_x
            pit_w = pit.x_end - pit.x_start
            pygame.draw.rect(
                self._screen,
                (135, 206, 235),
                (pit_x, self.floor_y, pit_w, self.screen_height - self.floor_y),
            )

        # Draw obstacles / pipes (dark green)
        for obs in self.obstacles:
            pygame.draw.rect(
                self._screen,
                (0, 100, 0),
                (obs.x - camera_x, obs.y, obs.width, obs.height),
            )

        # Draw goal flag (gold pole + red flag)
        flag_x = self.goal.x - camera_x
        pygame.draw.rect(
            self._screen,
            (255, 215, 0),
            (flag_x, self.goal.y, self.goal.width, self.goal.height),
        )

        # Draw Mario (red rectangle)
        pygame.draw.rect(
            self._screen,
            (220, 20, 60),
            (self.mario.x - camera_x, self.mario.y, self.mario.width, self.mario.height),
        )

        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)

    def close(self) -> None:
        """Close Pygame display cleanly."""
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None

    # --------------
    # Helper Methods
    # --------------
    def _get_raw_state(self) -> RawState:
        """Build local-vision RawState for the agent/discretizer."""
        upcoming_obs = [obs for obs in self.obstacles if obs.right > self.mario.x]
        if upcoming_obs:
            nearest_obs = min(upcoming_obs, key=lambda o: o.x)
            dist_to_obstacle = max(0.0, nearest_obs.x - self.mario.right)
            obstacle_height = nearest_obs.height
        else:
            dist_to_obstacle = -1.0
            obstacle_height = 0.0

        # Nearest upcoming pit
        upcoming_pits = [pit for pit in self.pits if pit.x_end > self.mario.x]
        if upcoming_pits:
            nearest_pit = min(upcoming_pits, key=lambda p: p.x_start)
            dist_to_pit = max(0.0, nearest_pit.x_start - self.mario.right)
        else:
            dist_to_pit = -1.0

        # Distance to goal
        dist_to_goal = max(0.0, self.goal.x - self.mario.right)

        return RawState(
            mario_x=self.mario.x,
            mario_y=self.mario.y,
            velocity_y=self.mario.vy,
            is_grounded=self.mario.is_grounded,
            dist_to_obstacle=dist_to_obstacle,
            obstacle_height=obstacle_height,
            dist_to_pit=dist_to_pit,
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
    
        

        