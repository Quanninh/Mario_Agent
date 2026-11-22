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

    # Hoá Character palette (Áo Ngũ Thân May Kép Đệm Bông & Khăn Đóng & Lồng Đèn)
    COLOR_PLAYER_ROBE = (45, 172, 240)         # Vibrant cyan-blue robe
    COLOR_PLAYER_ROBE_SHADOW = (22, 115, 188)   # Shaded robe folds
    COLOR_PLAYER_ROBE_LIGHT = (95, 215, 255)    # Quilted diamond highlights
    COLOR_PLAYER_KHAN = (32, 140, 205)          # Matching cyan-blue turban (Khăn Đóng)
    COLOR_PLAYER_KHAN_FOLD = (18, 95, 155)      # Turban pleat shadows
    COLOR_PLAYER_KHAN_LIGHT = (90, 205, 255)    # Turban pleat highlights
    COLOR_PLAYER_HAIR = (22, 22, 28)            # Black hair tuft / sideburn
    COLOR_PLAYER_HEAD = (248, 205, 172)         # Peachy skin tone
    COLOR_PLAYER_COLLAR = (255, 255, 255)       # Traditional white inner collar rim
    COLOR_PLAYER_PANTS = (242, 245, 250)        # White inner trousers
    COLOR_PLAYER_SHOES = (22, 22, 28)           # Black cloth shoes
    COLOR_PLAYER_SOLE = (225, 225, 225)         # White shoe sole line
    COLOR_PLAYER_BUTTON = (255, 220, 110)       # Gold button clasps
    COLOR_PLAYER_LANTERN_WOOD = (125, 75, 38)   # Bamboo lantern frame
    COLOR_PLAYER_LANTERN_GLOW = (255, 215, 75)  # Warm amber lantern glow
    COLOR_PLAYER_LANTERN_CORE = (255, 255, 220) # White-hot candle center
    COLOR_PLAYER_LANTERN_TASSEL = (225, 45, 45) # Red silk lantern tassel
    COLOR_PLAYER_JACKET = (45, 172, 240)        # Backwards-compatible alias

    # Khuê Văn Các Destination Sanctuary Palette
    COLOR_ROOF_TILE = (175, 72, 45)          # Weathered terracotta roof tile
    COLOR_ROOF_SHADOW = (115, 42, 28)        # Dark roof eaves shadow
    COLOR_ROOF_RIDGE = (195, 190, 180)       # Stone / mortar ridge ornament
    COLOR_WOOD_RED = (195, 45, 35)           # Vermilion red painted wood
    COLOR_WOOD_DARK = (125, 25, 20)          # Dark lacquer timber frame
    COLOR_GOLD = (245, 205, 75)              # Gold board lettering & accents
    COLOR_STONE_LIGHT = (225, 228, 232)      # White stone pillar with carved relief
    COLOR_STONE_DARK = (165, 172, 180)       # Carved stone cloud shading
    COLOR_STONE_BASE = (110, 118, 128)       # Greenish-grey stone plinth
    COLOR_WINDOW_GLOW = (255, 220, 85)       # Radiant sunburst window core
    COLOR_WINDOW_CORE = (255, 255, 230)      # White-hot inner window light
    COLOR_FLAG_RED = (220, 35, 35)           # Festival flag vermilion
    COLOR_FLAG_YELLOW = (255, 215, 50)       # Festival flag gold fringe
    COLOR_FLAG_GREEN = (40, 160, 80)         # Festival flag jade trim

    # Backwards-compatible aliases
    COLOR_HOUSE_WALL = (195, 45, 35)
    COLOR_HOUSE_ROOF = (175, 72, 45)
    COLOR_HOUSE_DOOR = (110, 118, 128)
    COLOR_HOUSE_WINDOW = (255, 220, 85)

    # Laser colors
    COLOR_LASER_CORE = (255, 255, 255)       # White-hot inner core
    COLOR_LASER_BEAM = (255, 30, 90)         # Blazing scarlet/magenta laser beam
    COLOR_LASER_GLOW = (255, 80, 160)        # Ethereal outer glow aura

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

            # Core internal logic canvas (800 x 400 pixels, 24-bit RGB without alpha channel to prevent alpha clobbering)
            self._canvas = pygame.Surface((self.canvas_width, self.canvas_height), depth=24)

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
            self._draw_ghost(canvas, ghost, camera_x, env.step_count)

        for item in env.items:
            if not item.is_collected:
                self._draw_item(canvas, item, camera_x, env.step_count)

        self._draw_home(canvas, env.goal, camera_x, env.step_count)

        # Draw low-ground creeping fog over the cemetery grass
        self._draw_ground_fog(canvas, camera_x, env.step_count, env.floor_y)

        # Draw player
        self._draw_player(canvas, env.player, camera_x, env.step_count)

        # Draw dynamic ambient lantern light casting forward onto the graveyard
        self._draw_lantern_light(canvas, env.player, camera_x, env.step_count)

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

        # Semi-transparent dark overlay over the frozen frame using surface alpha
        # Note: Avoid pygame.SRCALPHA on display surfaces to prevent Porter-Duff zero-alpha blackouts on macOS/SDL2
        overlay = pygame.Surface((sw, sh))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(160)
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
        elif events.hit_laser:
            self.show_overlay(
                title="VAPORIZED!",
                subtitle="Sliced by a rotating ghost eye-laser!",
                color=(255, 60, 120),  # Blazing laser magenta
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
                color=(165, 230, 255),  # Glacial frost / icy cyan
                duration_ms=3000,
            )

    

    # ------------------------------------------
    # Drawing Helpers (Midnight Graveyard Theme)
    # ------------------------------------------
    def _draw_background(self, canvas: pygame.Surface, env: "GameEnvironment", camera_x: float) -> None:
        canvas.fill(self.COLOR_SKY)

        # Twinkling night sky stars
        for i, (sx, sy) in enumerate(self.STARS):
            twinkle = int(math.sin(env.step_count * 0.12 + i * 1.7) * 35)
            b = max(160, min(255, 220 + twinkle))
            star_color = (max(0, b - 20), max(0, b - 10), b)
            sz = 1 if i % 3 != 0 else 2
            pygame.draw.circle(canvas, star_color, (sx, sy), sz)

        # Full moon with soft atmospheric outer glow
        moon_cx, moon_cy = self.canvas_width - 90, 65
        pygame.draw.circle(canvas, (32, 38, 68), (moon_cx, moon_cy), 36)
        pygame.draw.circle(canvas, (45, 52, 90), (moon_cx, moon_cy), 32)
        pygame.draw.circle(canvas, self.COLOR_MOON, (moon_cx, moon_cy), 28)

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

    def _draw_ghost(self, canvas: pygame.Surface, ghost: Ghost, camera_x: float, step_count: int = 0) -> None:
        x = ghost.x - camera_x
        y, w, h = ghost.y, ghost.width, ghost.height
        radius = int(w / 2)

        # Translucent ethereal trail behind ghost movement
        trail_dir = -1.0 if ghost.speed >= 0 else 1.0
        trail_surf = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
        pygame.draw.circle(trail_surf, (215, 242, 255, 60), (int(radius), int(radius)), radius)
        pygame.draw.rect(trail_surf, (215, 242, 255, 60), (0, radius, int(w), int(h - radius - 4)))
        canvas.blit(trail_surf, (x + trail_dir * 4, y))

        # Ghost head dome + body
        pygame.draw.circle(canvas, self.COLOR_GHOST, (int(x + radius), int(y + radius)), radius)
        pygame.draw.rect(canvas, self.COLOR_GHOST, (x, y + radius, w, h - radius - 4))

        # Wavy ghost tail (3 small circles along the bottom)
        wave_r = int(w / 6)
        for i in range(3):
            wx = int(x + wave_r + i * (w / 3))
            wy = int(y + h - wave_r)
            pygame.draw.circle(canvas, self.COLOR_GHOST, (wx, wy), wave_r)

        # Two dark hollow eyes with spooky icy specular glint
        eye_y = int(y + h * 0.4)
        e1_x = int(x + w * 0.35)
        e2_x = int(x + w * 0.65)
        pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (e1_x, eye_y), 3)
        pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (e2_x, eye_y), 3)
        pygame.draw.circle(canvas, (165, 230, 255), (e1_x, eye_y), 1)
        pygame.draw.circle(canvas, (165, 230, 255), (e2_x, eye_y), 1)

        # Draw 360-degree rotating eye-laser beam
        if ghost.has_laser:
            seg = ghost.get_laser_segment(step_count)
            if seg is not None:
                (sx, sy), (ex, ey) = seg
                p1 = (sx - camera_x, sy)
                p2 = (ex - camera_x, ey)
                # 3-layer beam: outer glow (5px), main beam (3px), white core (1px)
                pygame.draw.line(canvas, self.COLOR_LASER_GLOW, p1, p2, 5)
                pygame.draw.line(canvas, self.COLOR_LASER_BEAM, p1, p2, 3)
                pygame.draw.line(canvas, self.COLOR_LASER_CORE, p1, p2, 1)
                # Eye sockets flash with laser energy
                pygame.draw.circle(canvas, self.COLOR_LASER_BEAM, (e1_x, eye_y), 4)
                pygame.draw.circle(canvas, self.COLOR_LASER_CORE, (e1_x, eye_y), 2)
                pygame.draw.circle(canvas, self.COLOR_LASER_BEAM, (e2_x, eye_y), 4)
                pygame.draw.circle(canvas, self.COLOR_LASER_CORE, (e2_x, eye_y), 2)
                # Bright contact spark at beam tip
                pygame.draw.circle(canvas, self.COLOR_LASER_CORE, (int(p2[0]), int(p2[1])), 4)
                pygame.draw.circle(canvas, self.COLOR_LASER_BEAM, (int(p2[0]), int(p2[1])), 7, 1)

    def _draw_item(
        self,
        canvas: pygame.Surface,
        item: WarmthItem,
        camera_x: float,
        step_count: int = 0,
    ) -> None:
        cx = int(item.x - camera_x + item.width / 2)
        hover = math.sin(step_count * 0.15 + item.x * 0.05) * 2.5
        cy = int(item.y + item.height / 2 + hover)
        pulse = int(math.sin(step_count * 0.2 + item.x) * 2)

        # Ambient golden glow aura
        pygame.draw.circle(canvas, (180, 130, 30), (cx, cy), 13 + pulse, width=1)
        pygame.draw.circle(canvas, self.COLOR_ITEM_GLOW, (cx, cy), 8 + pulse)
        pygame.draw.circle(canvas, self.COLOR_ITEM_CORE, (cx, cy), 4)
        pygame.draw.circle(canvas, (255, 255, 255), (cx, cy), 2)

    def _draw_home(
        self,
        canvas: pygame.Surface,
        goal: GoalFlag,
        camera_x: float,
        step_count: int = 0,
    ) -> None:
        """Renders the authentic Khuê Văn Các (Pavilion of the Constellation of Literature) destination.
        
        Faithfully reproduces the iconic Temple of Literature architecture:
        - Carved white stone pillars and stepped stone courtyard with open passage
        - Spilling sanctuary light through the central portal
        - Upper vermilion red timber pavilion with decorative balustrade
        - Horizontal lacquered board inscribed in gold: '奎文閣'
        - Iconic circular sunburst window (Cửa Sổ Sao Khuê) with radiating wooden rays
        - Double-tiered curved roofs (Mái Chồng Diêm) with terracotta tiles and rooftop ornaments
        """
        x = goal.x - camera_x
        y, w, h = goal.y, goal.width, goal.height
        cx = int(x + w / 2)
        floor_y = int(y + h)

        # ----------------------------------------------------
        # 1. Lower Level: 4 Carved Stone Pillars & Open Archway
        # ----------------------------------------------------
        pillar_top = int(y + 44)
        pillar_h = int(floor_y - pillar_top)
        pillar_w = 14
        portal_w = int(w - pillar_w * 2)

        # Stone courtyard steps in center passage (Bậc tam cấp)
        pygame.draw.rect(canvas, (90, 95, 105), (x + 10, floor_y - 6, w - 20, 6))
        pygame.draw.rect(canvas, (135, 142, 152), (x + 12, floor_y - 4, w - 24, 4))
        pygame.draw.rect(canvas, (175, 182, 192), (x + 14, floor_y - 2, w - 28, 2))

        # Translucent warm light spilling through open portal from sanctuary
        portal_glow = pygame.Surface((portal_w, pillar_h - 6), pygame.SRCALPHA)
        portal_glow.fill((255, 215, 80, 32))
        canvas.blit(portal_glow, (int(x + pillar_w), pillar_top + 6))

        # Left & Right Carved White Stone Pillars (Chân tảng & Cột đá vuông chạm nổi)
        for px in [x, x + w - pillar_w]:
            # Pillar stone base plinth (Chân đế đá)
            pygame.draw.rect(canvas, self.COLOR_STONE_BASE, (px - 1, floor_y - 8, pillar_w + 2, 8))
            # White carved stone column shaft
            pygame.draw.rect(canvas, self.COLOR_STONE_LIGHT, (px, pillar_top + 2, pillar_w, pillar_h - 10))
            # Carved cloud / dragon relief motifs on pillar face
            for ry in range(int(pillar_top + 8), int(floor_y - 12), 8):
                pygame.draw.rect(canvas, self.COLOR_STONE_DARK, (px + 3, ry, pillar_w - 6, 2))
                pygame.draw.rect(canvas, (245, 248, 250), (px + 4, ry - 1, pillar_w - 8, 1))
            # Pillar capital trim
            pygame.draw.rect(canvas, self.COLOR_STONE_BASE, (px - 1, pillar_top, pillar_w + 2, 4))

        # Horizontal red crossbeam spanning above the pillars
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (x - 2, pillar_top - 2, w + 4, 5))
        pygame.draw.rect(canvas, self.COLOR_WOOD_RED, (x - 1, pillar_top - 1, w + 2, 3))

        # ----------------------------------------------------
        # 2. Middle Level: Red Wooden Pavilion & Balustrade
        # ----------------------------------------------------
        pav_top = int(y + 18)
        pav_h = int(pillar_top - pav_top)
        pav_w = 46
        pav_x = int(cx - pav_w / 2)

        # Vermilion timber walls
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (pav_x - 1, pav_top, pav_w + 2, pav_h))
        pygame.draw.rect(canvas, self.COLOR_WOOD_RED, (pav_x, pav_top + 1, pav_w, pav_h - 2))

        # Red Balustrade (Lan can con tiện đỏ)
        bal_y = int(pillar_top - 7)
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (pav_x - 4, bal_y, pav_w + 8, 7))
        pygame.draw.rect(canvas, self.COLOR_WOOD_RED, (pav_x - 3, bal_y + 1, pav_w + 6, 2))
        for bx in range(pav_x - 2, pav_x + pav_w + 4, 4):
            pygame.draw.rect(canvas, (235, 75, 60), (bx, bal_y + 3, 2, 3))
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (pav_x - 4, bal_y + 6, pav_w + 8, 2))

        # Gold Inscribed Horizontal Board "奎文閣" (Khuê Văn Các)
        board_w = 24
        board_x = int(cx - board_w / 2)
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (board_x, pav_top + 2, board_w, 6))
        pygame.draw.rect(canvas, (170, 30, 25), (board_x + 1, pav_top + 3, board_w - 2, 4))
        # Golden characters "奎文閣"
        pygame.draw.rect(canvas, self.COLOR_GOLD, (board_x + 3, pav_top + 4, 4, 2))
        pygame.draw.rect(canvas, self.COLOR_GOLD, (board_x + 10, pav_top + 4, 4, 2))
        pygame.draw.rect(canvas, self.COLOR_GOLD, (board_x + 17, pav_top + 4, 4, 2))

        # ----------------------------------------------------
        # 3. Centerpiece: The Iconic Circular Sunburst Window (Cửa Sổ Sao Khuê)
        # ----------------------------------------------------
        win_cx = cx
        win_cy = int(pav_top + 15)
        win_r = 9

        # Ambient warm glow radiating from the window
        pulse = int(math.sin(step_count * 0.15) * 2)
        win_glow = pygame.Surface((36, 36), pygame.SRCALPHA)
        pygame.draw.circle(win_glow, (255, 215, 80, 50), (18, 18), 16 + pulse)
        canvas.blit(win_glow, (win_cx - 18, win_cy - 18))

        # Outer round wooden frame
        pygame.draw.circle(canvas, self.COLOR_WOOD_DARK, (win_cx, win_cy), win_r + 2)
        pygame.draw.circle(canvas, self.COLOR_WOOD_RED, (win_cx, win_cy), win_r + 1)

        # Glowing amber interior
        pygame.draw.circle(canvas, self.COLOR_WINDOW_GLOW, (win_cx, win_cy), win_r - 1)
        pygame.draw.circle(canvas, self.COLOR_WINDOW_CORE, (win_cx, win_cy), win_r - 4)

        # 8 Radiating Wooden Sunburst Spokes (Nan gỗ sao Khuê)
        spoke_len = win_r - 1
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            sx = int(win_cx + math.cos(rad) * spoke_len)
            sy = int(win_cy + math.sin(rad) * spoke_len)
            pygame.draw.line(canvas, self.COLOR_WOOD_DARK, (win_cx, win_cy), (sx, sy), 1)

        # Hanging lantern inside center of round window
        pygame.draw.rect(canvas, self.COLOR_WOOD_DARK, (win_cx - 2, win_cy - 3, 4, 6), 1)
        pygame.draw.circle(canvas, (255, 255, 255), (win_cx, win_cy), 1)

        # ----------------------------------------------------
        # 4. Double-Tiered Flared Curved Roofs (Mái Chồng Diêm)
        # ----------------------------------------------------
        # Lower Roof Tier (Mái dưới che hiên)
        low_roof_y = pav_top + 1
        low_pts = [
            (cx - 38, low_roof_y + 3),   # Left upturned eave tip
            (cx - 34, low_roof_y + 1),
            (cx - 24, low_roof_y - 2),
            (cx + 24, low_roof_y - 2),
            (cx + 34, low_roof_y + 1),
            (cx + 38, low_roof_y + 3),   # Right upturned eave tip
            (cx + 32, low_roof_y + 5),
            (cx - 32, low_roof_y + 5),
        ]
        pygame.draw.polygon(canvas, self.COLOR_ROOF_SHADOW, [(px, py + 1) for px, py in low_pts])
        pygame.draw.polygon(canvas, self.COLOR_ROOF_TILE, low_pts)

        # Upper Roof Tier (Mái trên chính)
        up_top = int(y + 4)
        up_pts = [
            (cx - 32, up_top + 13),      # Left curved eave tip
            (cx - 28, up_top + 11),
            (cx - 20, up_top + 4),
            (cx - 14, up_top + 1),       # Ridge left
            (cx + 14, up_top + 1),       # Ridge right
            (cx + 20, up_top + 4),
            (cx + 28, up_top + 11),
            (cx + 32, up_top + 13),      # Right curved eave tip
            (cx + 26, up_top + 15),
            (cx - 26, up_top + 15),
        ]
        pygame.draw.polygon(canvas, self.COLOR_ROOF_SHADOW, [(px, py + 1) for px, py in up_pts])
        pygame.draw.polygon(canvas, self.COLOR_ROOF_TILE, up_pts)

        # Roof tiles vertical ridge lines
        for tx in range(cx - 18, cx + 20, 4):
            pygame.draw.line(canvas, (140, 52, 32), (tx, up_top + 3), (int(tx * 1.15 - cx * 0.15), up_top + 13), 1)

        # Ornamental Ridge & Pinnacle (Bình hồ lô & Kìm nóc)
        pygame.draw.rect(canvas, self.COLOR_ROOF_RIDGE, (cx - 15, up_top, 30, 2))
        pygame.draw.rect(canvas, self.COLOR_ROOF_RIDGE, (cx - 17, up_top - 3, 3, 4))
        pygame.draw.rect(canvas, self.COLOR_ROOF_RIDGE, (cx + 14, up_top - 3, 3, 4))
        pygame.draw.circle(canvas, self.COLOR_ROOF_RIDGE, (cx, up_top - 2), 3)
        pygame.draw.circle(canvas, (255, 235, 140), (cx, up_top - 2), 1)

    def _draw_ground_fog(
        self,
        canvas: pygame.Surface,
        camera_x: float,
        step_count: int,
        floor_y: float,
    ) -> None:
        """Renders low-ground creeping mist/fog drifting across the cemetery floor."""
        fog_surf = pygame.Surface((self.canvas_width, 36), pygame.SRCALPHA)
        for fx in range(-30, self.canvas_width + 40, 50):
            drift = math.sin(step_count * 0.04 + fx * 0.015) * 4
            alpha = int(24 + math.sin(step_count * 0.03 + fx * 0.02) * 12)
            pygame.draw.ellipse(fog_surf, (150, 185, 230, alpha), (fx, int(8 + drift), 85, 20))
        canvas.blit(fog_surf, (0, int(floor_y - 20)))

    def _draw_lantern_light(
        self,
        canvas: pygame.Surface,
        player: Player,
        camera_x: float,
        step_count: int,
    ) -> None:
        """Renders a compact circular warm light aura around Hoá's lantern."""
        x = player.x - camera_x
        y, w, h = player.y, player.width, player.height
        cx = x + w / 2
        f = 1 if player.facing_right else -1

        if player.is_crouching:
            lant_cx = int(cx + 12 * f)
            lant_cy = int(y + 10)
        else:
            is_running = abs(player.vx) > 0.5
            sway = math.sin(step_count * 0.3) * 1.5 if is_running else 0
            lant_cx = int(cx + 12 * f)
            lant_cy = int(y + 19 + sway)

        # Compact circular warm glow around the lantern
        light_w, light_h = 48, 48
        light_surf = pygame.Surface((light_w, light_h))
        light_surf.fill((0, 0, 0))
        center = (light_w // 2, light_h // 2)

        # Neat warm ambient glow circles
        pygame.draw.circle(light_surf, (45, 32, 12), center, 20)
        pygame.draw.circle(light_surf, (95, 70, 26), center, 14)
        pygame.draw.circle(light_surf, (155, 120, 48), center, 8)

        canvas.blit(
            light_surf,
            (lant_cx - light_w // 2, lant_cy - light_h // 2),
            special_flags=pygame.BLEND_ADD,
        )

    def _draw_player(
        self,
        canvas: pygame.Surface,
        player: Player,
        camera_x: float,
        step_count: int = 0,
    ) -> None:
        """Renders Hoá in traditional Vietnamese Áo Ngũ Thân May Kép Đệm Bông, matching Khăn Đóng & Lồng Đèn."""
        x = player.x - camera_x
        y, w, h = player.y, player.width, player.height
        cx = x + w / 2
        f = 1 if player.facing_right else -1

        # ----------------------------------------------------
        # 1. State: Crouch / Momentum Slide (Hitbox shrinks to h = 18px)
        # ----------------------------------------------------
        if player.is_crouching:
            robe_y = y + 6
            robe_h = 8
            robe_x = cx - 9 if player.facing_right else cx - 7
            pygame.draw.rect(canvas, self.COLOR_PLAYER_ROBE, (robe_x, robe_y, 16, robe_h), border_radius=2)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_LIGHT, (robe_x + 3, robe_y + 2), (robe_x + 13, robe_y + 2), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_SHADOW, (robe_x + 2, robe_y + 5), (robe_x + 14, robe_y + 5), 1)

            # White trousers tucked low under robe
            pants_x = robe_x - 3 * f
            pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (pants_x, y + h - 5, 8, 4), border_radius=1)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (pants_x - 2 * f, y + h - 3, 6, 3))
            pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (pants_x - 2 * f, y + h - 1), (pants_x + 4 * f, y + h - 1), 1)

            # Slide friction dust/frost particles when sliding fast
            if abs(player.vx) > 0.5:
                dust_x = pants_x - 4 * f
                pygame.draw.rect(canvas, (185, 215, 235), (dust_x, y + h - 2, 2, 2))
                pygame.draw.rect(canvas, (145, 180, 205), (dust_x - 3 * f, y + h - 4, 2, 2))

            # Head lowered with hair & matching blue Khăn Đóng
            head_x = cx + 2 * f
            head_y = y + 4
            # Dark hair contour behind
            pygame.draw.circle(canvas, self.COLOR_PLAYER_HAIR, (int(head_x - 1 * f), int(head_y)), 5)
            # Face skin
            pygame.draw.circle(canvas, self.COLOR_PLAYER_HEAD, (int(head_x), int(head_y)), 4)
            # Hair fringe
            pygame.draw.rect(canvas, self.COLOR_PLAYER_HAIR, (int(head_x - 1 * f), int(head_y - 4), 3, 2))

            # Blue Khăn Đóng matching robe
            pygame.draw.rect(canvas, self.COLOR_PLAYER_KHAN, (int(head_x - 5), int(head_y - 5), 10, 4), border_radius=2)
            pygame.draw.line(canvas, self.COLOR_PLAYER_KHAN_FOLD, (int(head_x - 4), int(head_y - 3)), (int(head_x + 4), int(head_y - 3)), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_KHAN_LIGHT, (int(head_x - 3), int(head_y - 4)), (int(head_x + 3), int(head_y - 4)), 1)

            # Directional eye
            eye_x = int(head_x + 2 * f)
            pygame.draw.circle(canvas, self.COLOR_GHOST_EYE, (eye_x, int(head_y)), 1)

            # Low Lantern held forward
            lant_x = int(cx + 12 * f)
            lant_y = int(y + 6)
            pygame.draw.rect(canvas, (80, 65, 30), (lant_x - 4, lant_y - 1, 9, 11), 1)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_WOOD, (lant_x - 3, lant_y, 7, 9), border_radius=1)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_GLOW, (lant_x - 2, lant_y + 2, 5, 5))
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_CORE, (lant_x - 1, lant_y + 3, 3, 3))
            # Red silk tassel hanging below
            pygame.draw.line(canvas, self.COLOR_PLAYER_LANTERN_TASSEL, (lant_x, lant_y + 9), (lant_x, lant_y + 12), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_LANTERN_WOOD, (lant_x, lant_y), (int(head_x + 3 * f), int(head_y + 4)), 1)

        # ----------------------------------------------------
        # 2. State: Standing / Running / Airborne (h = 32px)
        # ----------------------------------------------------
        else:
            is_running = abs(player.vx) > 0.5
            is_airborne = not player.is_grounded
            leg_phase = math.sin(step_count * 0.45) if is_running else 0.0

            # White Trousers & Black Shoes (Underneath Robe)
            if is_airborne:
                # Jump stance: tucked legs
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx - 5, y + 19, 4, 6))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx + 1, y + 18, 4, 6))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx - 6, y + 24, 5, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx - 6, y + 27), (cx - 1, y + 27), 1)
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx + 1, y + 23, 5, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx + 1, y + 26), (cx + 6, y + 26), 1)
            elif is_running:
                leg_offset = int(leg_phase * 4)
                # Left leg
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx - 4 - leg_offset, y + 21, 4, 7))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx - 5 - leg_offset, y + 27, 6, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx - 5 - leg_offset, y + 30), (cx + 1 - leg_offset, y + 30), 1)
                # Right leg
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx + leg_offset, y + 21, 4, 7))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx - 1 + leg_offset, y + 27, 6, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx - 1 + leg_offset, y + 30), (cx + 5 + leg_offset, y + 30), 1)
            else:
                # Standing still
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx - 5, y + 22, 4, 6))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_PANTS, (cx + 1, y + 22, 4, 6))
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx - 6, y + 27, 5, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx - 6, y + 30), (cx - 1, y + 30), 1)
                pygame.draw.rect(canvas, self.COLOR_PLAYER_SHOES, (cx + 1, y + 27, 5, 4))
                pygame.draw.line(canvas, self.COLOR_PLAYER_SOLE, (cx + 1, y + 30), (cx + 6, y + 30), 1)

            # Áo Ngũ Thân Đệm Bông (Quilted Padded Cyan-Blue Robe)
            robe_top = y + 9
            robe_bot = y + 23
            flutter = int(abs(leg_phase) * 3) if is_running else (2 if is_airborne else 0)
            robe_pts = [
                (cx - 5, robe_top),
                (cx + 5, robe_top),
                (cx + 7, robe_bot),
                (cx - 7 - flutter * f, robe_bot),
            ] if player.facing_right else [
                (cx - 5, robe_top),
                (cx + 5, robe_top),
                (cx + 7 + flutter, robe_bot),
                (cx - 7, robe_bot),
            ]
            pygame.draw.polygon(canvas, self.COLOR_PLAYER_ROBE, robe_pts)

            # Quilted diamond pattern stitches
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_LIGHT, (cx - 3, robe_top + 3), (cx + 3, robe_top + 7), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_LIGHT, (cx - 4, robe_top + 8), (cx + 4, robe_top + 12), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_SHADOW, (cx + 3, robe_top + 3), (cx - 3, robe_top + 7), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_SHADOW, (cx + 4, robe_top + 8), (cx - 4, robe_top + 12), 1)

            # Traditional white inner collar rim (Cổ lập lót trắng quý phái)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_COLLAR, (cx - 3, robe_top - 1, 6, 2), border_radius=1)

            # Diagonal collar seam (cổ ngũ thân cài chéo sang phải)
            pygame.draw.line(canvas, self.COLOR_PLAYER_ROBE_SHADOW, (cx - 2, robe_top + 1), (cx + 4 * f, robe_top + 4), 1)
            # Gold button clasps (khuy cài áo)
            pygame.draw.circle(canvas, self.COLOR_PLAYER_BUTTON, (int(cx + 3 * f), int(robe_top + 3)), 1)
            pygame.draw.circle(canvas, self.COLOR_PLAYER_BUTTON, (int(cx + 4 * f), int(robe_top + 6)), 1)

            # Head & Khăn Đóng (Matching Blue Turban + Hair)
            head_cx = int(cx)
            head_cy = int(y + 6)
            # Hair silhouette behind
            pygame.draw.circle(canvas, self.COLOR_PLAYER_HAIR, (head_cx - 1 * f, head_cy), 5)
            # Face skin
            pygame.draw.circle(canvas, self.COLOR_PLAYER_HEAD, (head_cx, head_cy), 5)

            # Front hair sideburn tuft
            pygame.draw.rect(canvas, self.COLOR_PLAYER_HAIR, (head_cx - 3 * f, head_cy - 2, 2, 4))

            # Khăn Đóng (Matching Cyan-Blue Turban!)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_KHAN, (head_cx - 6, head_cy - 7, 12, 5), border_radius=2)
            pygame.draw.line(canvas, self.COLOR_PLAYER_KHAN_FOLD, (head_cx - 5, head_cy - 4), (head_cx + 5, head_cy - 4), 1)
            pygame.draw.line(canvas, self.COLOR_PLAYER_KHAN_LIGHT, (head_cx - 4, head_cy - 6), (head_cx + 4, head_cy - 6), 1)
            # Turban búi tóc knot
            knot_x = head_cx - 2 * f
            pygame.draw.rect(canvas, self.COLOR_PLAYER_KHAN, (knot_x - 2, head_cy - 9, 4, 3), border_radius=1)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_KHAN_FOLD, (knot_x - 1, head_cy - 8, 2, 2))

            # Expressive eye facing movement direction
            eye_x = int(head_cx + 2 * f)
            eye_y = int(head_cy - 1)
            pygame.draw.rect(canvas, (245, 245, 250), (eye_x - 1, eye_y - 1, 3, 2))  # white sclera
            pupil_x = eye_x if player.facing_right else eye_x - 1
            pygame.draw.rect(canvas, self.COLOR_GHOST_EYE, (pupil_x, eye_y - 1, 2, 2))

            # Lantern (Chiếc Lồng Đèn Soi Đường + Tua Rua Đỏ)
            lant_cx = int(cx + 12 * f)
            sway = math.sin(step_count * 0.3) * 1.5 if is_running else 0
            lant_cy = int(y + 15 + sway)

            # Lantern halo & frame
            pygame.draw.rect(canvas, (80, 65, 30), (lant_cx - 6, lant_cy - 5, 12, 13), 1)
            hand_x = int(cx + 6 * f)
            hand_y = int(y + 13)
            pygame.draw.circle(canvas, self.COLOR_PLAYER_HEAD, (hand_x, hand_y), 2)
            pygame.draw.line(canvas, self.COLOR_PLAYER_LANTERN_WOOD, (hand_x, hand_y), (lant_cx, lant_cy - 4), 1)

            # Lantern body (hexagonal bamboo frame & glowing paper core)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_WOOD, (lant_cx - 4, lant_cy - 4, 8, 11), border_radius=1)
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_GLOW, (lant_cx - 3, lant_cy - 2, 6, 7))
            pygame.draw.rect(canvas, self.COLOR_PLAYER_LANTERN_CORE, (lant_cx - 1, lant_cy, 3, 4))
            pygame.draw.line(canvas, self.COLOR_PLAYER_LANTERN_WOOD, (lant_cx, lant_cy - 3), (lant_cx, lant_cy + 5), 1)

            # Red silk hanging tassel below lantern!
            tassel_sway = int(sway * 1.2)
            pygame.draw.line(canvas, self.COLOR_PLAYER_LANTERN_TASSEL, (lant_cx, lant_cy + 7), (lant_cx - tassel_sway, lant_cy + 11), 1)
            pygame.draw.circle(canvas, self.COLOR_PLAYER_BUTTON, (lant_cx, lant_cy + 7), 1)

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
