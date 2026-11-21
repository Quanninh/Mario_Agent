from typing import TYPE_CHECKING, List, Optional, Tuple
import math
import pygame

from .entities import Ghost, GoalFlag, Obstacle, Pit, Player, WarmthItem

if TYPE_CHECKING:
    from core.base_env import GameEvents
    from .game_env import GameEnvironment

class GameRenderer:
    """Handles all Pygame window management, camera scrolling, and 2D rendering."""

    # Flame jet constants (5-layer pixel-art palette matching reference design)
    COLOR_FIRE_WARN = (230, 90, 25)        # Warning amber/orange ember glow
    COLOR_FLAME_BORDER = (120, 15, 20)     # Dark maroon flame border
    COLOR_FLAME_RED = (225, 40, 25)        # Vivid blazing scarlet red
    COLOR_FLAME_ORANGE = (245, 125, 20)    # Warm radiant orange
    COLOR_FLAME_YELLOW = (255, 215, 30)    # Electric golden yellow
    COLOR_FLAME_WHITE = (255, 255, 255)    # White-hot incandescent core
    COLOR_FLAME_EMBER = (255, 180, 50)     # Floating ember sparks

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
        canvas_width: int = 800,
        canvas_height: int = 400,
        screen_width: int = 1400,   # For larger game window scaling
        screen_height: int = 700,   # For larger game window scaling
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Pygame rendering state (lazy-initialized in render())
        self._screen: Optional[pygame.Surface] = None
        self._canvas: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None

    def render(self, env: "GameEnvironment") -> None:
        """Render Hoá's Midnight Escape using Pygame."""
        if self._screen is None:
            pygame.init()

            # Displaying the game on the laptop's screen (can be scaled)
            self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))

            # Core internal logic canvas (800 x 400 pixels)
            self._canvas = pygame.Surface((self.canvas_width, self.canvas_height))

            pygame.display.set_caption("Hoá's Midnight Escape")
            self._clock = pygame.time.Clock()

            # Font style initialization 
            pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 11, bold=True)

        # Process window quit events cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        canvas = self._canvas
        # Camera offset (keeps Mario visible on wide levels)
        camera_x = max(0.0, env.player.x - self.canvas_width * 0.3)

        # Drawing entities in the game environment
        self._draw_background(canvas, env, camera_x)

        for obs in env.obstacles:
            self._draw_tombstone(canvas, obs, camera_x)

        for ghost in env.ghosts:
            self._draw_ghost(canvas, ghost, camera_x)

        for item in env.items:
            if not item.is_collected:
                self._draw_item(canvas, item, camera_x)

        self._draw_home(canvas, env.goal, camera_x)

        self._draw_player(canvas, env.player, camera_x)

        # Draw fixed top HUD (Level name, countdown timer, warmth counter)
        self._draw_hud(canvas, env)

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

    def show_overlay(
        self,
        title: str,
        subtitle: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        duration_ms: int = 1400,
    ) -> None:
        """Render a semi-transparent dark banner overlay and pause briefly before reset."""
        if self._screen is None:
            return

        sw, sh = self.screen_width, self.screen_height

        # Semi-transparent dark overlay over the frozen frame
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._screen.blit(overlay, (0, 0))

        # Fonts for banner
        title_font = pygame.font.SysFont("Arial", 44, bold=True)
        sub_font = pygame.font.SysFont("Arial", 18, bold=True)

        # Title with drop shadow
        t_surf = title_font.render(title, True, color)
        t_rect = t_surf.get_rect(center=(sw // 2, sh // 2 - 22))
        t_shadow = title_font.render(title, True, (10, 10, 10))
        self._screen.blit(t_shadow, (t_rect.x + 2, t_rect.y + 2))
        self._screen.blit(t_surf, t_rect)

        # Subtitle with drop shadow
        s_surf = sub_font.render(subtitle, True, (230, 235, 245))
        s_rect = s_surf.get_rect(center=(sw // 2, sh // 2 + 26))
        s_shadow = sub_font.render(subtitle, True, (10, 10, 10))
        self._screen.blit(s_shadow, (s_rect.x + 1, s_rect.y + 1))
        self._screen.blit(s_surf, s_rect)

        pygame.display.flip()

        # Event-pumping pause so window never freezes during the overlay
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < duration_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        return
                    # Pressing Space, Enter, or R skips the pause immediately
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_r):
                        return
            if self._clock:
                self._clock.tick(60)

    def show_game_over(self, env: "GameEnvironment", events: "GameEvents") -> None:
        """Determines and displays the appropriate victory or death overlay banner."""
        if events.reached_goal:
            collected = sum(1 for item in env.items if item.is_collected)
            total = len(env.items)
            warmth_info = f" • Warmth: {collected}/{total}" if total > 0 else ""
            self.show_overlay(
                title="HOME SAFE!",
                subtitle=f"Cleared {env.level_name} in {env.step_count} steps!{warmth_info}",
                color=(110, 245, 150),  # Radiant emerald green
                duration_ms=5000,
            )
        elif events.hit_fire:
            self.show_overlay(
                title="INCINERATED!",
                subtitle="Burned by the grave's inferno!",
                color=(255, 95, 30),  # Intense fiery orange
                duration_ms=3000,
            )
        elif events.hit_ghost:
            self.show_overlay(
                title="GAME OVER",
                subtitle="Caught by a spectral ghost!",
                color=(255, 80, 80),   # Crimson red
                duration_ms=3000,
            )
        elif events.fell_in_pit:
            self.show_overlay(
                title="GAME OVER",
                subtitle="Fell into an open grave pit!",
                color=(255, 80, 80),
                duration_ms=3000,
            )
        else:
            self.show_overlay(
                title="TIME OUT",
                subtitle="The freezing midnight overcame Hoá...",
                color=(255, 185, 70),  # Cold amber
                duration_ms=3000,
            )

    

    # ------------------------------------------
    # Drawing Helpers (Midnight Graveyard Theme)
    # ------------------------------------------
    def _draw_background(self, canvas: pygame.Surface, env: "GameEnvironment", camera_x: float) -> None:
        canvas.fill(self.COLOR_SKY)

        # Stars & full moon
        for sx, sy in self.STARS:
            pygame.draw.circle(canvas, self.COLOR_STAR, (sx, sy), 2)
        pygame.draw.circle(canvas, self.COLOR_MOON, (self.canvas_width - 90, 65), 28)

        # Graveyard soil & dark grass trim
        pygame.draw.rect(
            canvas,
            self.COLOR_SOIL,
            (0, env.floor_y, self.canvas_width, self.canvas_height - env.floor_y),
        )
        pygame.draw.rect(canvas, self.COLOR_GRASS, (0, env.floor_y, self.canvas_width, 6))

        # Carve out open grave pits and render flame hazards
        for pit in env.pits:
            pit_x = pit.x_start - camera_x
            pit_w = pit.x_end - pit.x_start
            pygame.draw.rect(
                canvas,
                self.COLOR_SKY,
                (pit_x, env.floor_y, pit_w, self.canvas_height - env.floor_y),
            )

            if pit.has_fire:
                phase = pit.get_fire_phase(env.step_count)
                if phase == 1:
                    # Warning phase: glowing pulsing embers at pit bottom
                    self._draw_pit_warning(canvas, pit_x, pit_w, env.floor_y, env.step_count)
                elif phase == 2:
                    # Active phase: blazing multi-spire pixel-art flame eruption!
                    self._draw_pit_flame(canvas, pit, pit_x, pit_w, env.floor_y, env.step_count)

    def _draw_pit_warning(
        self,
        canvas: pygame.Surface,
        pit_x: float,
        pit_w: float,
        floor_y: float,
        step_count: int,
    ) -> None:
        """Draws glowing, pulsing embers at the very bed of the pit (near canvas_height)."""
        pit_bottom = self.canvas_height
        pulse = abs(int(math.sin(step_count * 0.35) * 5))
        ember_h = 16 + pulse
        ember_y = pit_bottom - ember_h

        # Glowing magma bed at the bottom of the pit trench
        pygame.draw.rect(canvas, self.COLOR_FLAME_BORDER, (pit_x + 2, ember_y - 2, pit_w - 4, ember_h + 2))
        pygame.draw.rect(canvas, self.COLOR_FIRE_WARN, (pit_x + 4, ember_y, pit_w - 8, ember_h))
        pygame.draw.rect(canvas, self.COLOR_FLAME_ORANGE, (pit_x + 10, ember_y + 3, pit_w - 20, max(4, ember_h - 3)))
        pygame.draw.rect(canvas, self.COLOR_FLAME_YELLOW, (pit_x + 18, ember_y + 6, pit_w - 36, max(2, ember_h - 6)))
        pygame.draw.rect(canvas, self.COLOR_FLAME_WHITE, (pit_x + 26, ember_y + 10, pit_w - 52, max(2, ember_h - 10)))

        # Rising warning ember sparks bubbling up from the pit depths towards ground level
        pit_depth = self.canvas_height - floor_y
        for i, (pct_x, speed, shift) in enumerate([(0.22, 1.8, 0), (0.45, 2.3, 14), (0.65, 1.6, 28), (0.82, 2.0, 42)]):
            progress = (step_count * speed * 2 + shift) % (pit_depth - 10)
            sy = pit_bottom - 12 - progress
            sx = pit_x + pit_w * pct_x + math.sin(step_count * 0.2 + i * 2) * 3
            if sy > floor_y:
                sz = 3 if i % 2 == 0 else 2
                pygame.draw.rect(canvas, self.COLOR_FLAME_EMBER, (sx, sy, sz, sz))
                pygame.draw.rect(canvas, self.COLOR_FLAME_YELLOW, (sx + 1, sy + 1, max(1, sz - 1), max(1, sz - 1)))

    def _draw_pit_flame(
        self,
        canvas: pygame.Surface,
        pit: Pit,
        pit_x: float,
        pit_w: float,
        floor_y: float,
        step_count: int,
    ) -> None:
        """Draws multi-tiered pixel-art flame eruption matching reference design."""
        h = pit.FIRE_HEIGHT
        top_y = floor_y - h
        bot_y = self.canvas_height
        cx = pit_x + pit_w / 2.0

        # Subtle dynamic flicker
        f1 = int(math.sin(step_count * 0.4) * 3)
        f2 = int(math.cos(step_count * 0.5) * 3)

        # ----------------------------------------------------
        # 1. Inside the pit (Rectangular inferno shaft)
        # ----------------------------------------------------
        pit_depth = bot_y - floor_y
        pygame.draw.rect(canvas, self.COLOR_FLAME_BORDER, (pit_x, floor_y, pit_w, pit_depth))
        pygame.draw.rect(canvas, self.COLOR_FLAME_RED, (pit_x + 3, floor_y, pit_w - 6, pit_depth))
        pygame.draw.rect(canvas, self.COLOR_FLAME_ORANGE, (pit_x + 8, floor_y, pit_w - 16, pit_depth))
        pygame.draw.rect(canvas, self.COLOR_FLAME_YELLOW, (pit_x + 15, floor_y, pit_w - 30, pit_depth))
        pygame.draw.rect(canvas, self.COLOR_FLAME_WHITE, (pit_x + 24, floor_y + 10, pit_w - 48, pit_depth - 10))

        # ----------------------------------------------------
        # 2. Above the ground (Multi-spire flame burst)
        # ----------------------------------------------------
        # Layer 1: Dark Maroon Silhouette (4 jagged spires)
        dark_pts = [
            (pit_x, floor_y + 1),
            (pit_x + 2, floor_y - h * 0.22),
            (pit_x + 6, floor_y - h * 0.45),
            (pit_x + pit_w * 0.18, top_y + h * 0.36 + f2),
            (pit_x + pit_w * 0.28, top_y + h * 0.52),
            (pit_x + pit_w * 0.38, top_y + h * 0.14 + f1),
            (pit_x + pit_w * 0.46, top_y + h * 0.36),
            (cx + 2, top_y + f2),
            (pit_x + pit_w * 0.62, top_y + h * 0.32),
            (pit_x + pit_w * 0.74, top_y + h * 0.18 - f1),
            (pit_x + pit_w * 0.84, top_y + h * 0.50),
            (pit_x + pit_w - 8, top_y + h * 0.40 - f2),
            (pit_x + pit_w - 4, floor_y - h * 0.25),
            (pit_x + pit_w, floor_y + 1),
        ]
        pygame.draw.polygon(canvas, self.COLOR_FLAME_BORDER, dark_pts)

        # Layer 2: Vivid Scarlet Red
        red_pts = [
            (pit_x + 2, floor_y + 1),
            (pit_x + 4, floor_y - h * 0.20),
            (pit_x + 8, floor_y - h * 0.42),
            (pit_x + pit_w * 0.20, top_y + h * 0.40 + f2),
            (pit_x + pit_w * 0.29, top_y + h * 0.55),
            (pit_x + pit_w * 0.38, top_y + h * 0.18 + f1),
            (pit_x + pit_w * 0.46, top_y + h * 0.39),
            (cx + 2, top_y + h * 0.05 + f2),
            (pit_x + pit_w * 0.62, top_y + h * 0.35),
            (pit_x + pit_w * 0.73, top_y + h * 0.22 - f1),
            (pit_x + pit_w * 0.83, top_y + h * 0.53),
            (pit_x + pit_w - 10, top_y + h * 0.44 - f2),
            (pit_x + pit_w - 5, floor_y - h * 0.22),
            (pit_x + pit_w - 2, floor_y + 1),
        ]
        pygame.draw.polygon(canvas, self.COLOR_FLAME_RED, red_pts)

        # Layer 3: Warm Radiant Orange
        orange_pts = [
            (pit_x + 6, floor_y + 1),
            (pit_x + 10, floor_y - h * 0.20),
            (pit_x + pit_w * 0.26, top_y + h * 0.52 + f1),
            (pit_x + pit_w * 0.36, top_y + h * 0.64),
            (pit_x + pit_w * 0.44, top_y + h * 0.32 + f2),
            (cx + 4, top_y + h * 0.22 + f1),
            (pit_x + pit_w * 0.58, top_y + h * 0.48),
            (pit_x + pit_w * 0.68, top_y + h * 0.38 - f2),
            (pit_x + pit_w * 0.78, top_y + h * 0.60),
            (pit_x + pit_w - 12, floor_y - h * 0.20),
            (pit_x + pit_w - 6, floor_y + 1),
        ]
        pygame.draw.polygon(canvas, self.COLOR_FLAME_ORANGE, orange_pts)

        # Layer 4: Electric Golden Yellow
        yellow_pts = [
            (pit_x + 12, floor_y + 1),
            (pit_x + 16, floor_y - h * 0.22),
            (pit_x + pit_w * 0.34, top_y + h * 0.58 + f2),
            (pit_x + pit_w * 0.42, top_y + h * 0.48),
            (cx + 1, top_y + h * 0.38 + f1),
            (pit_x + pit_w * 0.58, top_y + h * 0.60),
            (pit_x + pit_w * 0.66, top_y + h * 0.52 - f2),
            (pit_x + pit_w - 16, floor_y - h * 0.22),
            (pit_x + pit_w - 12, floor_y + 1),
        ]
        pygame.draw.polygon(canvas, self.COLOR_FLAME_YELLOW, yellow_pts)

        # Layer 5: White-Hot Center Core
        white_pts = [
            (cx - 7, floor_y + 1),
            (cx - 7, floor_y - h * 0.18),
            (cx - 4, floor_y - h * 0.30),
            (cx - 2, top_y + h * 0.52 + f1),
            (cx + 1, top_y + h * 0.45 + f2),
            (cx + 3, top_y + h * 0.55),
            (cx + 5, floor_y - h * 0.28),
            (cx + 7, floor_y - h * 0.16),
            (cx + 7, floor_y + 1),
        ]
        pygame.draw.polygon(canvas, self.COLOR_FLAME_WHITE, white_pts)

        # Layer 6: Rising Pixel Ember Sparks
        ember_offsets = [
            (cx - 8, top_y - 8 + (step_count * 2) % 14, 3),
            (cx + 6, top_y - 4 + ((step_count * 3) % 16), 2),
            (pit_x + pit_w * 0.22, top_y + h * 0.15 + ((step_count * 2 + 5) % 12), 3),
            (pit_x + pit_w * 0.76, top_y + h * 0.18 + ((step_count * 2 + 8) % 14), 2),
        ]
        for ex, ey, es in ember_offsets:
            pygame.draw.rect(canvas, self.COLOR_FLAME_EMBER, (ex, ey, es, es))
            pygame.draw.rect(canvas, self.COLOR_FLAME_YELLOW, (ex + 1, ey + 1, max(1, es - 1), max(1, es - 1)))

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

    def _draw_home(self, canvas: pygame.Surface, goal: GoalFlag, camera_x: float) -> None:
        x = goal.x - camera_x
        y, w, h = goal.y, goal.width, goal.height
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

    def _draw_player(self, canvas: pygame.Surface, player: Player, camera_x: float) -> None:
        x = player.x - camera_x
        y, w, h = player.y, player.width, player.height

        # Winter jacket body
        pygame.draw.rect(canvas, self.COLOR_PLAYER_JACKET, (x + 2, y + 12, w - 4, h - 12))

        # Head
        pygame.draw.circle(canvas, self.COLOR_PLAYER_HEAD, (int(x + w / 2), int(y + 8)), 8)

        # Eye facing movement direction
        eye_x = int(x + w * 0.65) if player.facing_right else int(x + w * 0.35)
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

    def _draw_hud(self, canvas: pygame.Surface, env: "GameEnvironment") -> None:
        """Render fixed HUD stacked vertically in the top-left corner."""
        if self._font is None:
            return

        # Line 1: Level name
        lvl_text = self._font.render(env.level_name, True, (230, 235, 245))
        lvl_shadow = self._font.render(env.level_name, True, (10, 10, 20))
        canvas.blit(lvl_shadow, (17, 15))
        canvas.blit(lvl_text, (16, 14))

        # Line 2: Remaining steps countdown (right below level name)
        steps_left = max(0, env.max_steps - env.step_count)
        timer_str = f"Steps Left: {steps_left}"
        timer_color = (255, 95, 85) if steps_left <= 150 else (210, 220, 240)

        timer_text = self._font.render(timer_str, True, timer_color)
        timer_shadow = self._font.render(timer_str, True, (10, 10, 20))
        canvas.blit(timer_shadow, (17, 30))
        canvas.blit(timer_text, (16, 29))

        # Line 3: Warmth counter (right below steps left)
        if env.items:
            collected = sum(1 for item in env.items if item.is_collected)
            total = len(env.items)
            warmth_str = f"Warmth: {collected}/{total}"

            warmth_text = self._font.render(warmth_str, True, self.COLOR_ITEM_GLOW)
            warmth_shadow = self._font.render(warmth_str, True, (10, 10, 20))
            canvas.blit(warmth_shadow, (17, 45))
            canvas.blit(warmth_text, (16, 44))
