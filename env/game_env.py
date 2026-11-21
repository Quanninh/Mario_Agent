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
    DOUBLE_JUMP_RATIO: float = 0.88
    SLIDE_FRICTION: float = 0.90
    STANDING_HEIGHT: float = 32.0
    CROUCH_HEIGHT: float = 18.0

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

    # Decorative stars in the night sky (canvas-space coordinates)
    STARS = [
        (70, 75), (150, 90), (230, 35), (320, 75), (410, 40),
        (510, 85), (590, 30), (740, 110), (270, 125), (470, 120),
    ]

    def __init__(
        self,
        obstacles: Optional[List[Obstacle]] = None,
        pits: Optional[List[Pit]] = None,
        ghosts: Optional[List[Ghost]] = None,
        items: Optional[List[WarmthItem]] = None,
        goal: Optional[GoalFlag] = None,
        canvas_width: int = 800,
        canvas_height: int = 400,
        floor_y: float = 320.0,
        max_steps: int = 1000,
        screen_width: int = 1400,   # For larger game window scaling
        screen_height: int = 700,   # For larger game window scaling
        level_name: str = "Graveyard",
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.level_name = level_name
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
        self._canvas: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None

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

        # Check hit ghost death
        if any(self._rects_overlap(self.player.rect, g.rect) for g in self.ghosts):
            self.player.is_alive = False
            events.hit_ghost = True
            done = True

        # Check pit death
        elif self.player.y > self.canvas_height:
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

            # Displaying the game on the laptop's screen (can be scaled)
            self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Core internal logic canvas (800 x 400 pixels)
            self._canvas = pygame.Surface((self.canvas_width, self.canvas_height))

            pygame.display.set_caption("Hoá's Midnight Escape")
            self._clock = pygame.time.Clock()

            # Font style initialization for the player's name label
            pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 11, bold=True)

        # Process window quit events cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        canvas = self._canvas
        # Camera offset (keeps Mario visible on wide levels)
        camera_x = max(0.0, self.player.x - self.canvas_width * 0.3)

        # Drawing entities in the game environment
        self._draw_background(canvas, camera_x)

        for obs in self.obstacles:
            self._draw_tombstone(canvas, obs, camera_x)

        for ghost in self.ghosts:
            self._draw_ghost(canvas, ghost, camera_x)

        for item in self.items:
            if not item.is_collected:
                self._draw_item(canvas, item, camera_x)

        self._draw_home(canvas, camera_x)

        self._draw_player(canvas, camera_x)

        # Draw fixed top HUD (Level name, countdown timer, warmth counter)
        self._draw_hud(canvas)

        # Scale from canvas logic to screen displaying
        scaled = pygame.transform.scale(canvas, (self.screen_width, self.screen_height))
        self._screen.blit(scaled, (0, 0))

        # Set the FPS of the game
        pygame.display.flip()
        if self._clock:
            self._clock.tick(60)

    def close(self) -> None:
        """Close Pygame display cleanly."""
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._canvas = None
            self._clock = None
            self._font = None

    # ------------------------------------------
    # Drawing Helpers (Midnight Graveyard Theme)
    # ------------------------------------------
    def _draw_background(self, canvas: pygame.Surface, camera_x: float) -> None:
        canvas.fill(self.COLOR_SKY)

        # Stars & full moon
        for sx, sy in self.STARS:
            pygame.draw.circle(canvas, self.COLOR_STAR, (sx, sy), 2)
        pygame.draw.circle(canvas, self.COLOR_MOON, (self.canvas_width - 90, 65), 28)

        # Graveyard soil & dark grass trim
        pygame.draw.rect(
            canvas,
            self.COLOR_SOIL,
            (0, self.floor_y, self.canvas_width, self.canvas_height - self.floor_y),
        )
        pygame.draw.rect(canvas, self.COLOR_GRASS, (0, self.floor_y, self.canvas_width, 6))

        # Carve out open grave pits
        for pit in self.pits:
            pit_x = pit.x_start - camera_x
            pit_w = pit.x_end - pit.x_start
            pygame.draw.rect(
                canvas,
                self.COLOR_SKY,
                (pit_x, self.floor_y, pit_w, self.canvas_height - self.floor_y),
            )

    def _draw_tombstone(self, canvas: pygame.Surface, obs: Obstacle, camera_x: float) -> None:
        x = obs.x - camera_x
        y, w, h = obs.y, obs.width, obs.height
        if obs.style == "cross":
            # ----------------------------------------------------
            # Cross-Shaped Tombstone (Latin stone cross monument)
            # ----------------------------------------------------
            pedestal_h = 8

            # Base stone pedestal at the bottom
            pygame.draw.rect(canvas, self.COLOR_TOMBSTONE, (x, y + h - pedestal_h, w, pedestal_h))

            # Vertical stone pillar of the cross
            col_w = int(w * 0.38)
            col_x = int(x + (w - col_w) / 2)
            pygame.draw.rect(canvas, self.COLOR_TOMBSTONE, (col_x, y, col_w, h - pedestal_h))

            # Horizontal crossbar (arms of the cross)
            bar_h = int(w * 0.32)
            bar_y = int(y + (h - pedestal_h) * 0.28)
            pygame.draw.rect(canvas, self.COLOR_TOMBSTONE, (x + 2, bar_y, w - 4, bar_h))

            # Engraved center seam detail for 3D stone texture
            cx = int(x + w / 2)
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx, y + 4), (cx, y + h - pedestal_h - 4), 1)
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (x + 4, int(bar_y + bar_h / 2)), (x + w - 4, int(bar_y + bar_h / 2)), 1)
        else:
            # ----------------------------------------------------
            # Buddhist Tombstone (Classic arch shape with engraved chữ Vạn 卍)
            # ----------------------------------------------------
            radius = int(w / 2)

            # Rounded arch top + rectangular stone body (classic shape)
            pygame.draw.circle(canvas, self.COLOR_TOMBSTONE, (int(x + radius), int(y + radius)), radius)
            pygame.draw.rect(canvas, self.COLOR_TOMBSTONE, (x, y + radius, w, h - radius))

            # Engraved Buddhist Sauwastika symbol (chữ Vạn 卍)
            cx = int(x + w / 2)
            cy = int(y + h * 0.45)
            s = 5  # Symbol half-size

            # Center cross bars
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx, cy - s), (cx, cy + s), 2)
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx - s, cy), (cx + s, cy), 2)

            # 4 bent arms forming the auspicious 卍 (left-facing sauwastika)
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx, cy - s), (cx - s, cy - s), 2)  # Top arm -> left
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx - s, cy), (cx - s, cy + s), 2)  # Left arm -> down
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx, cy + s), (cx + s, cy + s), 2)  # Bottom arm -> right
            pygame.draw.line(canvas, self.COLOR_TOMBSTONE_DETAIL, (cx + s, cy), (cx + s, cy - s), 2)  # Right arm -> up

    def _draw_ghost(self, canvas: pygame.Surface, ghost: Ghost, camera_x: float) -> None:
        x = ghost.x - camera_x
        y, w, h = ghost.y, ghost.width, ghost.height
        radius = int(w / 2)

        # Ghost head dome + body
        pygame.draw.circle(canvas, self.COLOR_GHOST, (int(x + radius), int(y + radius)), radius)
        pygame.draw.rect(canvas, self.COLOR_GHOST, (x, y + radius, w, h - radius - 4))

        # Wavy ghost tail (3 small circles along the bottom)
        wave_r = int(w / 6)
        for i in range(3):
            wx = int(x + wave_r + i * (w / 3))
            wy = int(y + h - wave_r)
            pygame.draw.circle(canvas, self.COLOR_GHOST, (wx, wy), wave_r)

        # Two dark hollow eyes
        pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (int(x + w * 0.35), int(y + h * 0.4)), 3)
        pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (int(x + w * 0.65), int(y + h * 0.4)), 3)

    def _draw_item(self, canvas: pygame.Surface, item: WarmthItem, camera_x: float) -> None:
        cx = int(item.x - camera_x + item.width / 2)
        cy = int(item.y + item.height / 2)

        pygame.draw.circle(canvas, self.COLOR_ITEM_GLOW, (cx, cy), 9)
        pygame.draw.circle(canvas, self.COLOR_ITEM_CORE, (cx, cy), 4)

    def _draw_home(self, canvas: pygame.Surface, camera_x: float) -> None:
        x = self.goal.x - camera_x
        y, w, h = self.goal.y, self.goal.width, self.goal.height
        roof_h = 28

        # House wall
        pygame.draw.rect(canvas, self.COLOR_HOUSE_WALL, (x, y + roof_h, w, h - roof_h))

        # Triangular roof
        roof_pts = [(x - 6, y + roof_h), (x + w / 2, y), (x + w + 6, y + roof_h)]
        pygame.draw.polygon(canvas, self.COLOR_HOUSE_ROOF, roof_pts)

        # Front door
        pygame.draw.rect(canvas, self.COLOR_HOUSE_DOOR, (x + 10, y + h - 30, 16, 30))

        # Glowing warm yellow window
        pygame.draw.rect(canvas, self.COLOR_HOUSE_WINDOW, (x + 34, y + roof_h + 14, 18, 18))

    def _draw_player(self, canvas: pygame.Surface, camera_x: float) -> None:
        x = self.player.x - camera_x
        y, w, h = self.player.y, self.player.width, self.player.height

        # Winter jacket body
        pygame.draw.rect(canvas, self.COLOR_PLAYER_JACKET, (x + 2, y + 12, w - 4, h - 12))

        # Head
        pygame.draw.circle(canvas, self.COLOR_PLAYER_HEAD, (int(x + w / 2), int(y + 8)), 8)

        # Eye facing movement direction
        eye_x = int(x + w * 0.65) if self.player.facing_right else int(x + w * 0.35)
        pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (eye_x, int(y + 7)), 2)

        # "Hoá" label above the player
        if self._font is not None:
            tag_text = self._font.render("Hoá", True, (245, 235, 190))  # Warm pale yellow
            shadow_text = self._font.render("Hoá", True, (10, 10, 20))   # Dark shadow for contrast

            tag_x = int(x + w / 2 - tag_text.get_width() / 2)
            tag_y = int(y - 16)  # 14px above player head

            # Draw drop shadow first, then main text on top
            canvas.blit(shadow_text, (tag_x + 1, tag_y + 1))
            canvas.blit(tag_text, (tag_x, tag_y))

    def _draw_hud(self, canvas: pygame.Surface) -> None:
        """Render fixed HUD stacked vertically in the top-left corner."""
        if self._font is None:
            return

        # Line 1: Level name
        lvl_text = self._font.render(self.level_name, True, (230, 235, 245))
        lvl_shadow = self._font.render(self.level_name, True, (10, 10, 20))
        canvas.blit(lvl_shadow, (17, 15))
        canvas.blit(lvl_text, (16, 14))

        # Line 2: Remaining steps countdown (right below level name)
        steps_left = max(0, self.max_steps - self.step_count)
        timer_str = f"Steps Left: {steps_left}"
        timer_color = (255, 95, 85) if steps_left <= 150 else (210, 220, 240)

        timer_text = self._font.render(timer_str, True, timer_color)
        timer_shadow = self._font.render(timer_str, True, (10, 10, 20))
        canvas.blit(timer_shadow, (17, 30))
        canvas.blit(timer_text, (16, 29))

        # Line 3: Warmth counter (right below steps left)
        if self.items:
            collected = sum(1 for item in self.items if item.is_collected)
            total = len(self.items)
            warmth_str = f"Warmth: {collected}/{total}"

            warmth_text = self._font.render(warmth_str, True, self.COLOR_ITEM_GLOW)
            warmth_shadow = self._font.render(warmth_str, True, (10, 10, 20))
            canvas.blit(warmth_shadow, (17, 45))
            canvas.blit(warmth_text, (16, 44))
        
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
            velocity_x =self.player.vx,
            velocity_y=self.player.vy,
            is_grounded=self.player.is_grounded,
            can_double_jump=self.player.can_double_jump,
            is_crouching=self.player.is_crouching,
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
    
        

        