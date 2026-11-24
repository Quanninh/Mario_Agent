from typing import List, Optional, Tuple
import pygame

from core.base_env import Action, BaseEnvironment, GameEvents, RawState
from env.entities import Ghost, GoalFlag, Obstacle, Pit, Player, WarmthItem

class GameEnvironment(BaseEnvironment):
    """2D platformer environment for Hoá's Midnight Escape."""

    # Physics constants
    GRAVITY: float = 0.8
    WALK_SPEED: float = 4.0
    JUMP_FORCE: float = -13.0

    # Midnight Graveyard color palette 
    COLOR_SKY = (15, 20, 45)
    COLOR_MOON = (245, 240, 200)
    COLOR_STAR = (210, 220, 255)
    COLOR_SOIL = (38, 46, 44)
    COLOR_GRASS = (52, 78, 60)
    COLOR_TOMBSTONE = (115, 122, 136)
    COLOR_TOMBSTONE_DETAIL = (75, 80, 92)
    COLOR_GHOST = (215, 242, 255)
    COLOR_GHOST_EYE = (20, 25, 45)
    COLOR_ITEM_GLOW = (255, 195, 65)
    COLOR_ITEM_CORE = (255, 245, 180)
    COLOR_PLAYER_JACKET = (65, 145, 230)
    COLOR_PLAYER_HEAD = (245, 210, 180)
    COLOR_HOUSE_WALL = (185, 135, 90)
    COLOR_HOUSE_ROOF = (145, 50, 50)
    COLOR_HOUSE_DOOR = (100, 65, 40)
    COLOR_HOUSE_WINDOW = (255, 230, 115)

    # Decorative stars in the night sky (screen-space coordinates)
    STARS = [
        (70, 45), (150, 90), (230, 35), (320, 75), (410, 40),
        (510, 85), (590, 30), (740, 110), (270, 125), (470, 120),
    ]

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        pits: Optional[List[Pit]] = None,
        ghosts: Optional[List[Ghost]] = None,
        items: Optional[List[WarmthItem]] = None,
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

        # Pygame rendering state (lazy-initialized in render())
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

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
        events = GameEvents()

        # --------------------------------------
        # Horizontal Movement & X-Axis Collision
        # --------------------------------------
        vx = 0.0
        if action in (Action.RIGHT, Action.RIGHT_JUMP):
            vx = self.WALK_SPEED
            self.player.facing_right = True
        elif action == Action.LEFT:
            vx = -self.WALK_SPEED
            self.player.facing_right = False

        self.player.x += vx

        # Prevent walking off the left screen boundary
        if self.player.x < 0.0:
            self.player.x = 0.0

        # Resolve side collisions with obstacles
        for obs in self.obstacles:
            if self._rects_overlap(self.player.rect, obs.rect):
                events.hit_obstacle = True
                if vx > 0:  # Moving right -> snap to left edge of obstacle
                    self.player.x = obs.x - self.player.width
                elif vx < 0:  # Moving left -> snap to right edge of obstacle
                    self.player.x = obs.right

        # -----------------------------------------------------
        # Vertical Movement (Jump + Gravity) & Y-Axis Collision
        # -----------------------------------------------------
        if action in (Action.JUMP, Action.RIGHT_JUMP) and self.player.is_grounded:
            self.player.vy = self.JUMP_FORCE
            self.player.is_grounded = False

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
                elif self.player.vy < 0:  # Hitting bottom of obstacle
                    self.player.y = obs.bottom
                    self.player.vy = 0.0

        # Resolve floor collision (only if NOT standing over a pit)
        over_pit = any(pit.contains(self.player.x, self.player.width) for pit in self.pits)
        if not over_pit and old_bottom <= self.floor_y and self.player.bottom >= self.floor_y:
            self.player.y = self.floor_y - self.player.height
            self.player.vy = 0.0
            self.player.is_grounded = True

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

        # Check hit ghost death
        if any(self._rects_overlap(self.player.rect, g.rect) for g in self.ghosts):
            self.player.is_alive = False
            events.hit_ghost = True
            done = True

        # Check pit death
        elif self.player.y > self.screen_height:
            self.player.is_alive = False
            events.fell_in_pit = True
            done = True

        # Check goal flag reached
        elif self.goal.is_reached(self.player):
            events.reached_goal = True
            done = True

        # Check episode timeout
        elif self.step_count >= self.max_steps:
            done = True

        return self._get_raw_state(), events, done

    def render(self) -> None:
        """Render Hoá's Midnight Escape using Pygame."""
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Hoá's Midnight Escape")
            self._clock = pygame.time.Clock()

        # Process window quit events cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        screen = self._screen
        # Camera offset (keeps Mario visible on wide levels)
        camera_x = max(0.0, self.player.x - self.screen_width * 0.3)

        # Drawing entities in the game environment
        self._draw_background(screen, camera_x)

        for obs in self.obstacles:
            self._draw_tombstone(screen, obs, camera_x)

        for ghost in self.ghosts:
            self._draw_ghost(screen, ghost, camera_x)

        for item in self.items:
            if not item.is_collected:
                self._draw_item(screen, item, camera_x)

        self._draw_home(screen, camera_x)

        self._draw_player(screen, camera_x)

        # Set the FPS of the game
        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)

    def close(self) -> None:
        """Close Pygame display cleanly."""
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None

    # ------------------------------------------
    # Drawing Helpers (Midnight Graveyard Theme)
    # ------------------------------------------
    def _draw_background(self, screen: pygame.Surface, camera_x: float) -> None:
        screen.fill(self.COLOR_SKY)

        # Stars & full moon
        for sx, sy in self.STARS:
            pygame.draw.circle(screen, self.COLOR_STAR, (sx, sy), 2)
        pygame.draw.circle(screen, self.COLOR_MOON, (self.screen_width - 90, 65), 28)

        # Graveyard soil & dark grass trim
        pygame.draw.rect(
            screen,
            self.COLOR_SOIL,
            (0, self.floor_y, self.screen_width, self.screen_height - self.floor_y),
        )
        pygame.draw.rect(screen, self.COLOR_GRASS, (0, self.floor_y, self.screen_width, 6))

        # Carve out open grave pits
        for pit in self.pits:
            pit_x = pit.x_start - camera_x
            pit_w = pit.x_end - pit.x_start
            pygame.draw.rect(
                screen,
                self.COLOR_SKY,
                (pit_x, self.floor_y, pit_w, self.screen_height - self.floor_y),
            )

    def _draw_tombstone(self, screen: pygame.Surface, obs: Obstacle, camera_x: float) -> None:
        x = obs.x - camera_x
        y, w, h = obs.y, obs.width, obs.height
        radius = int(w / 2)

        # Rounded arch top + rectangular stone body
        pygame.draw.circle(screen, self.COLOR_TOMBSTONE, (int(x + radius), int(y + radius)), radius)
        pygame.draw.rect(screen, self.COLOR_TOMBSTONE, (x, y + radius, w, h - radius))

        # Engraved cross (+)
        cx = int(x + w / 2)
        cy = int(y + h * 0.45)
        pygame.draw.line(screen, self.COLOR_TOMBSTONE_DETAIL, (cx, cy - 8), (cx, cy + 10), 2)
        pygame.draw.line(screen, self.COLOR_TOMBSTONE_DETAIL, (cx - 6, cy - 2), (cx + 6, cy - 2), 2)

    def _draw_ghost(self, screen: pygame.Surface, ghost: Ghost, camera_x: float) -> None:
        x = ghost.x - camera_x
        y, w, h = ghost.y, ghost.width, ghost.height
        radius = int(w / 2)

        # Ghost head dome + body
        pygame.draw.circle(screen, self.COLOR_GHOST, (int(x + radius), int(y + radius)), radius)
        pygame.draw.rect(screen, self.COLOR_GHOST, (x, y + radius, w, h - radius - 4))

        # Wavy ghost tail (3 small circles along the bottom)
        wave_r = int(w / 6)
        for i in range(3):
            wx = int(x + wave_r + i * (w / 3))
            wy = int(y + h - wave_r)
            pygame.draw.circle(screen, self.COLOR_GHOST, (wx, wy), wave_r)

        # Two dark hollow eyes
        pygame.draw.circle(screen, self.COLOR_GHOST_EYE, (int(x + w * 0.35), int(y + h * 0.4)), 3)
        pygame.draw.circle(screen, self.COLOR_GHOST_EYE, (int(x + w * 0.65), int(y + h * 0.4)), 3)

    def _draw_item(self, screen: pygame.Surface, item: WarmthItem, camera_x: float) -> None:
        cx = int(item.x - camera_x + item.width / 2)
        cy = int(item.y + item.height / 2)

        pygame.draw.circle(screen, self.COLOR_ITEM_GLOW, (cx, cy), 9)
        pygame.draw.circle(screen, self.COLOR_ITEM_CORE, (cx, cy), 4)

    def _draw_home(self, screen: pygame.Surface, camera_x: float) -> None:
        x = self.goal.x - camera_x
        y, w, h = self.goal.y, self.goal.width, self.goal.height
        roof_h = 28

        # House wall
        pygame.draw.rect(screen, self.COLOR_HOUSE_WALL, (x, y + roof_h, w, h - roof_h))

        # Triangular roof
        roof_pts = [(x - 6, y + roof_h), (x + w / 2, y), (x + w + 6, y + roof_h)]
        pygame.draw.polygon(screen, self.COLOR_HOUSE_ROOF, roof_pts)

        # Front door
        pygame.draw.rect(screen, self.COLOR_HOUSE_DOOR, (x + 10, y + h - 30, 16, 30))

        # Glowing warm yellow window
        pygame.draw.rect(screen, self.COLOR_HOUSE_WINDOW, (x + 34, y + roof_h + 14, 18, 18))

    def _draw_player(self, screen: pygame.Surface, camera_x: float) -> None:
        x = self.player.x - camera_x
        y, w, h = self.player.y, self.player.width, self.player.height

        # Winter jacket body
        pygame.draw.rect(screen, self.COLOR_PLAYER_JACKET, (x + 2, y + 12, w - 4, h - 12))

        # Head
        pygame.draw.circle(screen, self.COLOR_PLAYER_HEAD, (int(x + w / 2), int(y + 8)), 8)

        # Eye facing movement direction
        eye_x = int(x + w * 0.65) if self.player.facing_right else int(x + w * 0.35)
        pygame.draw.circle(screen, self.COLOR_GHOST_EYE, (eye_x, int(y + 7)), 2)


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
        else:
            dist_to_pit = -1.0

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
            velocity_y=self.player.vy,
            is_grounded=self.player.is_grounded,
            dist_to_obstacle=dist_to_obstacle,
            obstacle_height=obstacle_height,
            dist_to_pit=dist_to_pit,
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
    
        

        