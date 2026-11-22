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
    COLOR_SKY = (16, 10, 24)
    COLOR_MOON = (245, 240, 200)
    COLOR_STAR = (210, 220, 255)
    COLOR_SOIL = (38, 46, 44)
    COLOR_GRASS = (52, 78, 60)
    COLOR_TOMBSTONE = (115, 122, 136)
    COLOR_TOMBSTONE_DETAIL = (75, 80, 92)

    # Enhanced Tombstone & Granite Palettes
    COLOR_TOMB_GRANITE_HI = (195, 210, 228)    # Polished specular edge on black granite
    COLOR_TOMB_GRANITE_MIDHI = (125, 140, 158) # Granite bevel highlight
    COLOR_TOMB_GRANITE_BODY = (46, 54, 66)     # Polished dark granite body
    COLOR_TOMB_GRANITE_DARK = (28, 33, 42)     # Deep black granite face
    COLOR_TOMB_GRANITE_SHD = (14, 17, 22)      # Deep granite shadow / contour
    COLOR_TOMB_GOLD_HI = (255, 245, 160)       # Radiant gold inscription sparkle
    COLOR_TOMB_GOLD_MID = (232, 198, 75)       # Rich gold inscription & chữ Vạn 卍
    COLOR_TOMB_GOLD_SHD = (165, 125, 40)       # Antique shadow gold
    COLOR_TOMB_GRASS = (62, 108, 66)           # Grass turf on Catholic tomb slab
    COLOR_TOMB_GRASS_HI = (92, 145, 85)        # Highlight on turf
    COLOR_TOMB_STONE_HI = (185, 198, 212)      # Traditional sandstone highlight
    COLOR_TOMB_STONE_MID = (125, 138, 152)     # Traditional sandstone body
    COLOR_TOMB_STONE_SHD = (75, 85, 98)        # Traditional sandstone shadow / grooves
    COLOR_TOMB_STONE_DEEP = (40, 46, 56)       # Deep stone contours
    COLOR_TOMB_NICHE_BG = (20, 24, 32)         # Deep shrine cavity
    COLOR_TOMB_INCENSE_POT = (85, 75, 65)      # Bronze incense pot
    COLOR_TOMB_INCENSE_RED = (220, 60, 45)     # Glowing red incense tip
    COLOR_TOMB_MOSS = (56, 84, 60)             # Weathered stone moss
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

    # Blood Moon & Mystical Eye Palette (Reference Art)
    COLOR_MOON_AURA_1 = (35, 18, 28)
    COLOR_MOON_AURA_2 = (62, 20, 32)
    COLOR_MOON_AURA_3 = (92, 24, 36)
    COLOR_MOON_BASE = (150, 26, 32)
    COLOR_MOON_MID = (185, 38, 44)
    COLOR_MOON_LIGHT = (215, 52, 54)
    COLOR_MOON_HIGHLIGHT = (235, 75, 75)
    COLOR_MOON_CRATER = (130, 22, 28)

    # Ancient Tree & Roots Palette
    COLOR_TREE_DARK = (24, 20, 22)
    COLOR_TREE_MID = (54, 44, 40)
    COLOR_TREE_LIGHT = (90, 75, 65)

    # All-Seeing Eye Palette
    COLOR_EYE_SCLERA = (244, 238, 220)
    COLOR_EYE_IRIS_RING = (36, 30, 28)
    COLOR_EYE_PUPIL_BG = (226, 210, 170)
    COLOR_EYE_RUNE = (34, 26, 24)

    # Traditional Vietnamese Scroll Clouds (Luminous Standout Vân Mây Palette)
    COLOR_CLOUD_OUTLINE = (32, 18, 42)        # Deep midnight plum border (high-contrast separation)
    COLOR_CLOUD_BORDER = (32, 18, 42)         # Backwards-compatible alias
    COLOR_CLOUD_SHADOW = (138, 108, 158)      # Amethyst/orchid shadow
    COLOR_CLOUD_BODY = (212, 192, 232)        # Luminous lilac (bright & stands out against sky)
    COLOR_CLOUD_LIGHT = (252, 248, 255)       # Moonlit pearl ivory crest
    COLOR_CLOUD_SPIRAL = (56, 32, 72)         # Dark plum filigree spirals
    COLOR_CLOUD_GOLD = (235, 195, 95)         # Warm gold accents

    # 16-Bit Dithered Night Sky Palette (Matching Reference Art)
    SKY_PALETTE = [
        (16, 10, 24),    # Band 0: Deep Obsidian Void
        (26, 14, 38),    # Band 1: Deep Midnight Violet
        (38, 20, 56),    # Band 2: Dark Royal Aubergine
        (52, 28, 78),    # Band 3: Twilight Amethyst
        (70, 44, 92),    # Band 4: Muted Dusk Purple
        (92, 64, 110),   # Band 5: Soft Mauve Twilight
        (102, 72, 120),  # Band 6: Warm Horizon Glow
    ]

    # 4x4 Ordered Bayer Dither Matrix normalized to 0..15
    BAYER_4X4 = [
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ]

    # Retro Pixel-Art Star Palette
    COLOR_STAR_GOLD = (255, 238, 175)
    COLOR_STAR_WHITE = (255, 255, 255)
    COLOR_STAR_LAVENDER = (220, 212, 245)
    COLOR_STAR_DIM = (165, 145, 190)

    # Pixel-Art Stars Specification: (x, y, kind, color_type)
    # kind: 'cross' (4-pointed twinkle), 'square' (2x2), 'dot' (1x1)
    STARS_SPEC = [
        (75, 42, "square", "gold"),
        (190, 26, "dot", "lavender"),
        (235, 68, "square", "gold"),
        (300, 88, "dot", "lavender"),
        (398, 92, "square", "dim"),
        (405, 65, "dot", "lavender"),
        (515, 52, "square", "gold"),
        (590, 35, "dot", "lavender"),
        (690, 58, "square", "gold"),
        (740, 38, "dot", "lavender"),
        (45, 168, "dot", "lavender"),
        (145, 185, "square", "gold"),
        (205, 215, "square", "gold"),
        (270, 195, "dot", "lavender"),
        (335, 195, "dot", "gold"),
        (405, 198, "cross", "white"),    # Center 4-pointed sparkle star
        (565, 155, "square", "gold"),
        (625, 222, "dot", "lavender"),
        (740, 182, "cross", "white"),    # Right 4-pointed sparkle star
        (55, 235, "square", "gold"),
        (115, 260, "dot", "dim"),
        (290, 290, "square", "gold"),
        (435, 280, "dot", "dim"),
        (605, 275, "dot", "lavender"),
        (710, 285, "dot", "dim"),
    ]

    # Backwards-compatible coordinates
    STARS = [(s[0], s[1]) for s in STARS_SPEC]

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
        self._sky_surface: Optional[pygame.Surface] = None
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
            self._sky_surface = None
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
        # 1. 16-bit dithered pixel-art night sky gradient
        canvas.blit(self._get_sky_surface(env.floor_y), (0, 0))

        # 2. Twinkling retro pixel-art stars matching reference aesthetic
        self._draw_stars(canvas, env.step_count)

        # Distant drifting traditional scroll clouds across the night sky
        self._draw_sky_clouds(canvas, camera_x, env.step_count)

        # Upper framing clouds behind the Blood Moon
        moon_cx, moon_cy = self.canvas_width - 130.0, 82.0
        self._draw_scroll_cloud(canvas, moon_cx - 58, moon_cy - 28, scale=0.85, flip_h=False, step_count=env.step_count, phase=0.5)
        self._draw_scroll_cloud(canvas, moon_cx + 55, moon_cy - 22, scale=0.88, flip_h=True, step_count=env.step_count, phase=1.5)

        # Sacred Crimson Blood Moon with Gnarled Tree & All-Seeing Eye (Reference Art)
        self._draw_blood_moon(canvas, moon_cx, moon_cy, env.step_count)

        # Foreground ornamental scroll clouds framing the Blood Moon
        self._draw_moon_clouds(canvas, moon_cx, moon_cy, env.step_count)

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
        h = pit.fire_height
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

    def _get_sky_surface(self, floor_y: float) -> pygame.Surface:
        """Retrieves or lazily builds the cached 16-bit dithered retro night sky gradient surface."""
        if self._sky_surface is None:
            self._sky_surface = self._build_dithered_sky(floor_y)
        return self._sky_surface

    def _build_dithered_sky(self, floor_y: float) -> pygame.Surface:
        """Pre-renders an authentic 16-bit dithered pixel-art night sky matching the reference image."""
        sky_surf = pygame.Surface((self.canvas_width, self.canvas_height), depth=24)
        sky_surf.fill(self.SKY_PALETTE[0])

        num_transitions = len(self.SKY_PALETTE) - 1
        band_height = floor_y / num_transitions
        dither_thickness = 22.0
        block_size = 2

        px_array = pygame.PixelArray(sky_surf)

        for y in range(0, int(floor_y)):
            band_idx = int(y / band_height)
            if band_idx >= num_transitions:
                band_idx = num_transitions - 1

            c_top = self.SKY_PALETTE[band_idx]
            c_bot = self.SKY_PALETTE[band_idx + 1]

            transition_center = (band_idx + 1) * band_height
            trans_top = transition_center - dither_thickness / 2.0
            trans_bot = transition_center + dither_thickness / 2.0

            if y < trans_top:
                color = c_top
                for x in range(self.canvas_width):
                    px_array[x, y] = color
            elif y > trans_bot:
                color = c_bot
                for x in range(self.canvas_width):
                    px_array[x, y] = color
            else:
                t = (y - trans_top) / dither_thickness
                threshold = t * 16.0
                for x in range(self.canvas_width):
                    bx = (x // block_size) % 4
                    by = (y // block_size) % 4
                    if self.BAYER_4X4[by][bx] < threshold:
                        px_array[x, y] = c_bot
                    else:
                        px_array[x, y] = c_top

        del px_array
        return sky_surf

    def _draw_stars(self, canvas: pygame.Surface, step_count: int) -> None:
        """Renders authentic retro pixel-art stars (twinkle sparkle crosses, glowing squares, and pinprick dots)."""
        color_map = {
            "gold": self.COLOR_STAR_GOLD,
            "white": self.COLOR_STAR_WHITE,
            "lavender": self.COLOR_STAR_LAVENDER,
            "dim": self.COLOR_STAR_DIM,
        }
        for i, (sx, sy, kind, ctype) in enumerate(self.STARS_SPEC):
            twinkle = math.sin(step_count * 0.12 + i * 1.5)
            base_color = color_map[ctype]

            if kind == "cross":
                # 4-pointed sparkle star with center core and cardinal arms
                pygame.draw.circle(canvas, (175, 155, 210), (sx, sy), 3)
                pygame.draw.line(canvas, self.COLOR_STAR_LAVENDER, (sx - 2, sy), (sx + 2, sy), 1)
                pygame.draw.line(canvas, self.COLOR_STAR_LAVENDER, (sx, sy - 2), (sx, sy + 2), 1)
                pygame.draw.rect(canvas, self.COLOR_STAR_WHITE, (sx, sy, 1, 1))
            elif kind == "square":
                sz = 2 if twinkle > -0.4 else 1
                pygame.draw.rect(canvas, base_color, (sx, sy, sz, sz))
            else:
                if twinkle > -0.6:
                    pygame.draw.rect(canvas, base_color, (sx, sy, 1, 1))

    def _draw_sky_clouds(self, canvas: pygame.Surface, camera_x: float, step_count: int) -> None:
        """Renders stylized traditional scroll clouds drifting across the upper graveyard sky."""
        cloud_configs = [
            (80.0, 50.0, 1.05, False, 0.0),
            (240.0, 80.0, 0.85, True, 1.4),
            (390.0, 48.0, 1.00, False, 2.8),
            (530.0, 76.0, 0.82, True, 4.2),
        ]
        wrap_w = self.canvas_width + 180.0
        for base_x, y, scale, flip_h, phase in cloud_configs:
            # Subtle parallax scrolling across horizontal progression
            px = ((base_x - camera_x * 0.12) % wrap_w) - 90.0
            self._draw_scroll_cloud(canvas, px, y, scale=scale, flip_h=flip_h, step_count=step_count, phase=phase)

    def _draw_moon_clouds(self, canvas: pygame.Surface, moon_cx: float, moon_cy: float, step_count: int) -> None:
        """Renders ornamental scroll clouds framing and partially overlapping the Blood Moon."""
        moon_clouds = [
            (-70.0, 20.0, 0.80, False, 2.5),
            (65.0, 26.0, 0.82, True, 3.5),
            (-50.0, 52.0, 0.88, False, 4.5),
            (52.0, 58.0, 0.76, True, 5.5),
        ]
        for rel_x, rel_y, scale, flip_h, phase in moon_clouds:
            self._draw_scroll_cloud(
                canvas,
                moon_cx + rel_x,
                moon_cy + rel_y,
                scale=scale,
                flip_h=flip_h,
                step_count=step_count,
                phase=phase,
            )

    def _draw_scroll_cloud(
        self,
        canvas: pygame.Surface,
        cx: float,
        cy: float,
        scale: float = 1.0,
        flip_h: bool = False,
        step_count: int = 0,
        phase: float = 0.0,
    ) -> None:
        """Renders authentic Vietnamese/East Asian Ruyi scroll clouds (Vân Mây) with luminous standout colors."""
        hover = math.sin(step_count * 0.04 + phase) * 1.5
        base_cy = cy + hover
        sign = -1.0 if flip_h else 1.0

        # Multi-tiered cascading cloud lobes matching reference artwork
        lobes = [
            (0.0, -8.0, 14.0),
            (-12.0, -4.0, 12.0),
            (13.0, -4.0, 12.5),
            (-24.0, 2.0, 10.5),
            (25.0, 2.0, 11.0),
            (-34.0, 8.0, 8.5),
            (35.0, 8.0, 9.0),
            (-12.0, 9.0, 10.0),
            (12.0, 9.0, 10.5),
            (0.0, 11.0, 10.0),
        ]

        transformed_lobes = []
        for lx, ly, lr in lobes:
            tx = cx + lx * scale * sign
            ty = base_cy + ly * scale
            tr = max(2, int(lr * scale))
            transformed_lobes.append((tx, ty, tr))

        # Serpentine wispy trailing ribbon tail (S-curve)
        tail_pts = [
            (-28.0, 10.0),
            (-38.0, 13.0),
            (-50.0, 12.0),
            (-62.0, 9.0),
            (-72.0, 7.0),
            (-82.0, 10.0),
            (-90.0, 14.0),
        ]
        t_tail = [(cx + px * scale * sign, base_cy + py * scale) for px, py in tail_pts]

        # Pass 1: Dark outer plum outline (Guarantees crisp contrast against purple night sky)
        if len(t_tail) >= 2:
            pygame.draw.lines(canvas, self.COLOR_CLOUD_OUTLINE, False, [(int(p[0]), int(p[1])) for p in t_tail], max(2, int(7 * scale)))
        for tx, ty, tr in transformed_lobes:
            pygame.draw.circle(canvas, self.COLOR_CLOUD_OUTLINE, (int(tx), int(ty)), tr + 2)

        # Pass 2: Amethyst/orchid shadow layer
        if len(t_tail) >= 2:
            pygame.draw.lines(canvas, self.COLOR_CLOUD_SHADOW, False, [(int(p[0]), int(p[1])) for p in t_tail], max(1, int(4 * scale)))
        for tx, ty, tr in transformed_lobes:
            pygame.draw.circle(canvas, self.COLOR_CLOUD_SHADOW, (int(tx), int(ty)), tr)

        # Pass 3: Luminous lilac body (High luminance, pops out prominently from the sky)
        if len(t_tail) >= 2:
            t_tail_inner = [(p[0], p[1] - 1.0 * scale) for p in t_tail]
            pygame.draw.lines(canvas, self.COLOR_CLOUD_BODY, False, [(int(p[0]), int(p[1])) for p in t_tail_inner], max(1, int(3 * scale)))
        for tx, ty, tr in transformed_lobes:
            pygame.draw.circle(canvas, self.COLOR_CLOUD_BODY, (int(tx), int(ty - 1.5 * scale)), max(1, tr - 1))

        # Pass 4: Pearlescent moonlit ivory crest highlights
        for tx, ty, tr in transformed_lobes:
            pygame.draw.circle(canvas, self.COLOR_CLOUD_LIGHT, (int(tx), int(ty - 3.5 * scale)), max(1, int(tr * 0.65)))

        # Pass 5: Engraved traditional spiral swirls (Vân Xoáy) inside prominent lobes
        swirl_specs = [
            (0.0, -8.0, 7.0),
            (-12.0, -4.0, 6.0),
            (13.0, -4.0, 6.5),
            (-24.0, 2.0, 5.5),
            (25.0, 2.0, 5.5),
            (-34.0, 8.0, 4.5),
            (35.0, 8.0, 4.5),
        ]
        for sx, sy, sr in swirl_specs:
            scx = cx + sx * scale * sign
            scy = base_cy + sy * scale - 1.5 * scale
            r = max(2, int(sr * scale))
            pygame.draw.arc(canvas, self.COLOR_CLOUD_SPIRAL, (int(scx - r), int(scy - r), r * 2, r * 2), 0.4, 5.8, 1)
            if r >= 4:
                in_r = r - 2
                pygame.draw.arc(canvas, self.COLOR_CLOUD_SPIRAL, (int(scx - in_r + 1 * sign), int(scy - in_r), in_r * 2, in_r * 2), 2.0, 6.2, 1)

        # Tail decorative filigree wave line
        if len(t_tail) >= 2:
            pygame.draw.lines(canvas, self.COLOR_CLOUD_SPIRAL, False, [(int(p[0]), int(p[1] - 1.0 * scale)) for p in t_tail], 1)

    def _draw_blood_moon(self, canvas: pygame.Surface, moon_cx: float, moon_cy: float, step_count: int = 0) -> None:
        """Renders the sacred Crimson Blood Moon with the ancient All-Seeing Eye and gnarled tree roots."""
        r = 50.0

        # 1. Cosmic atmospheric crimson aura
        pygame.draw.circle(canvas, self.COLOR_MOON_AURA_1, (int(moon_cx), int(moon_cy)), int(r + 16))
        pygame.draw.circle(canvas, self.COLOR_MOON_AURA_2, (int(moon_cx), int(moon_cy)), int(r + 10))
        pygame.draw.circle(canvas, self.COLOR_MOON_AURA_3, (int(moon_cx), int(moon_cy)), int(r + 4))

        # 2. Layered crimson blood moon sphere
        pygame.draw.circle(canvas, self.COLOR_MOON_BASE, (int(moon_cx), int(moon_cy)), int(r))
        pygame.draw.circle(canvas, self.COLOR_MOON_MID, (int(moon_cx - 5), int(moon_cy - 5)), int(r - 6))
        pygame.draw.circle(canvas, self.COLOR_MOON_LIGHT, (int(moon_cx - 4), int(moon_cy - 6)), int(r - 16))
        pygame.draw.circle(canvas, self.COLOR_MOON_HIGHLIGHT, (int(moon_cx - 3), int(moon_cy - 8)), int(r - 28))

        # Subtle dark crater textures
        craters = [(-22, -20, 7), (-14, 25, 9), (24, -22, 6), (28, 12, 5), (-32, 10, 6)]
        for rel_x, rel_y, cr in craters:
            pygame.draw.circle(canvas, self.COLOR_MOON_CRATER, (int(moon_cx + rel_x), int(moon_cy + rel_y)), cr)

        # 3. Jagged fractured "bitten" edge on bottom-right rim
        bite_teeth = [
            (moon_cx + 49, moon_cy + 12),
            (moon_cx + 36, moon_cy + 22),
            (moon_cx + 46, moon_cy + 28),
            (moon_cx + 28, moon_cy + 34),
            (moon_cx + 42, moon_cy + 40),
            (moon_cx + 22, moon_cy + 44),
            (moon_cx + 32, moon_cy + 49),
            (moon_cx + 10, moon_cy + 50),
            (moon_cx + 16, moon_cy + 54),
            (moon_cx + 56, moon_cy + 56),
            (moon_cx + 56, moon_cy + 12),
        ]
        pygame.draw.polygon(canvas, self.COLOR_SKY, bite_teeth)
        for i in range(len(bite_teeth) - 3):
            pygame.draw.line(canvas, (18, 12, 16), bite_teeth[i], bite_teeth[i + 1], 2)

        # 4. Gnarled Ancient Tree Branches & Roots
        branches = [
            [(moon_cx, moon_cy - 12), (moon_cx - 14, moon_cy - 24), (moon_cx - 30, moon_cy - 28), (moon_cx - 40, moon_cy - 24)],
            [(moon_cx - 14, moon_cy - 24), (moon_cx - 22, moon_cy - 38), (moon_cx - 32, moon_cy - 42)],
            [(moon_cx - 14, moon_cy - 24), (moon_cx - 8, moon_cy - 40), (moon_cx - 12, moon_cy - 46)],
            [(moon_cx, moon_cy - 12), (moon_cx - 2, moon_cy - 28), (moon_cx - 4, moon_cy - 42), (moon_cx - 2, moon_cy - 48)],
            [(moon_cx - 2, moon_cy - 28), (moon_cx + 6, moon_cy - 38), (moon_cx + 12, moon_cy - 45)],
            [(moon_cx, moon_cy - 12), (moon_cx + 14, moon_cy - 22), (moon_cx + 28, moon_cy - 26), (moon_cx + 38, moon_cy - 30)],
            [(moon_cx + 14, moon_cy - 22), (moon_cx + 22, moon_cy - 36), (moon_cx + 28, moon_cy - 44)],
            [(moon_cx + 28, moon_cy - 26), (moon_cx + 38, moon_cy - 20), (moon_cx + 44, moon_cy - 14)],
            [(moon_cx - 18, moon_cy - 8), (moon_cx - 34, moon_cy - 8), (moon_cx - 42, moon_cy - 4)],
            [(moon_cx - 20, moon_cy + 2), (moon_cx - 36, moon_cy + 6), (moon_cx - 44, moon_cy + 12)],
            [(moon_cx + 18, moon_cy - 6), (moon_cx + 32, moon_cy - 6), (moon_cx + 42, moon_cy - 2)],
        ]
        for b in branches:
            pygame.draw.lines(canvas, self.COLOR_TREE_DARK, False, b, 2)
            if len(b) >= 2:
                pygame.draw.line(canvas, self.COLOR_TREE_MID, b[0], b[1], 1)

        root_tendrils = [
            [(moon_cx - 4, moon_cy + 12), (moon_cx - 8, moon_cy + 24), (moon_cx - 6, moon_cy + 36), (moon_cx, moon_cy + 48), (moon_cx + 4, moon_cy + 58), (moon_cx + 10, moon_cy + 64), (moon_cx + 16, moon_cy + 62)],
            [(moon_cx + 2, moon_cy + 12), (moon_cx - 2, moon_cy + 24), (moon_cx - 2, moon_cy + 38), (moon_cx + 3, moon_cy + 52), (moon_cx + 8, moon_cy + 60)],
            [(moon_cx - 10, moon_cy + 10), (moon_cx - 16, moon_cy + 20), (moon_cx - 18, moon_cy + 32), (moon_cx - 14, moon_cy + 46), (moon_cx - 8, moon_cy + 54)],
            [(moon_cx - 16, moon_cy + 8), (moon_cx - 24, moon_cy + 18), (moon_cx - 26, moon_cy + 30), (moon_cx - 22, moon_cy + 40), (moon_cx - 16, moon_cy + 44)],
            [(moon_cx + 8, moon_cy + 10), (moon_cx + 12, moon_cy + 20), (moon_cx + 10, moon_cy + 32), (moon_cx + 4, moon_cy + 44)],
            [(moon_cx + 14, moon_cy + 6), (moon_cx + 18, moon_cy + 16), (moon_cx + 16, moon_cy + 28)],
        ]
        for r_pts in root_tendrils:
            pygame.draw.lines(canvas, self.COLOR_TREE_DARK, False, r_pts, 3)
            pygame.draw.lines(canvas, self.COLOR_TREE_MID, False, r_pts, 1)

        # 5. The All-Seeing Eye in the center
        upper_lid = [
            (moon_cx - 20, moon_cy),
            (moon_cx - 14, moon_cy - 9),
            (moon_cx - 6, moon_cy - 13),
            (moon_cx + 6, moon_cy - 13),
            (moon_cx + 14, moon_cy - 9),
            (moon_cx + 20, moon_cy),
        ]
        lower_lid = [
            (moon_cx - 20, moon_cy),
            (moon_cx - 14, moon_cy + 9),
            (moon_cx - 6, moon_cy + 13),
            (moon_cx + 6, moon_cy + 13),
            (moon_cx + 14, moon_cy + 9),
            (moon_cx + 20, moon_cy),
        ]
        eye_poly = [
            (moon_cx - 18, moon_cy),
            (moon_cx - 12, moon_cy - 7),
            (moon_cx - 5, moon_cy - 10),
            (moon_cx + 5, moon_cy - 10),
            (moon_cx + 12, moon_cy - 7),
            (moon_cx + 18, moon_cy),
            (moon_cx + 12, moon_cy + 7),
            (moon_cx + 5, moon_cy + 10),
            (moon_cx - 5, moon_cy + 10),
            (moon_cx - 12, moon_cy + 7),
        ]
        pygame.draw.polygon(canvas, self.COLOR_EYE_SCLERA, eye_poly)
        pygame.draw.lines(canvas, self.COLOR_TREE_DARK, False, upper_lid, 3)
        pygame.draw.lines(canvas, self.COLOR_TREE_DARK, False, lower_lid, 3)
        pygame.draw.lines(canvas, self.COLOR_TREE_LIGHT, False, upper_lid, 1)

        pygame.draw.circle(canvas, self.COLOR_EYE_IRIS_RING, (int(moon_cx), int(moon_cy)), 9)
        pygame.draw.circle(canvas, self.COLOR_EYE_PUPIL_BG, (int(moon_cx), int(moon_cy)), 7)
        pygame.draw.circle(canvas, self.COLOR_EYE_IRIS_RING, (int(moon_cx), int(moon_cy)), 7, 1)

        # Sacred Mystic Rune / Glyph (ॐ Om / mystic spiral rune)
        pygame.draw.arc(canvas, self.COLOR_EYE_RUNE, (int(moon_cx - 5), int(moon_cy - 4), 5, 4), 0.5, 4.5, 1)
        pygame.draw.arc(canvas, self.COLOR_EYE_RUNE, (int(moon_cx - 5), int(moon_cy), 5, 4), 1.5, 5.5, 1)
        pygame.draw.line(canvas, self.COLOR_EYE_RUNE, (int(moon_cx - 1), int(moon_cy + 1)), (int(moon_cx + 3), int(moon_cy + 3)), 1)
        pygame.draw.arc(canvas, self.COLOR_EYE_RUNE, (int(moon_cx), int(moon_cy - 5), 4, 3), 3.2, 6.2, 1)
        pygame.draw.circle(canvas, self.COLOR_EYE_RUNE, (int(moon_cx + 2), int(moon_cy - 5)), 1)

    def _draw_tombstone(self, canvas: pygame.Surface, obs: Obstacle, camera_x: float) -> None:
        x = obs.x - camera_x
        y, w, h = obs.y, obs.width, obs.height

        if obs.style == "cross":
            # ----------------------------------------------------
            # Catholic Tombstone: Polished Black Granite Monument
            # Massive Latin Cross occupies >60% of total height,
            # standing on an upright inscribed stele & stepped slab.
            # ----------------------------------------------------
            cx = int(x + w / 2)
            cross_h = int(h * 0.61)
            base_h = h - cross_h
            base_y = y + cross_h

            # --- 1. Towering Latin Cross (>60% of height) ---
            shaft_w = max(4, int(w * 0.20))
            shaft_x = int(cx - shaft_w / 2)

            # Vertical pillar
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_BODY, (shaft_x, int(y + 2), shaft_w, int(cross_h - 4)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (shaft_x, int(y + 3)), (shaft_x, int(base_y - 3)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_SHD, (shaft_x + shaft_w - 1, int(y + 3)), (shaft_x + shaft_w - 1, int(base_y - 3)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (shaft_x, int(y + 2)), (shaft_x + shaft_w - 1, int(y + 2)))

            # Horizontal crossbar
            bar_w = int(w * 0.72)
            bar_h = shaft_w
            bar_x = int(cx - bar_w / 2)
            bar_y = int(y + cross_h * 0.26)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_BODY, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (bar_x, bar_y), (bar_x + bar_w - 1, bar_y))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_SHD, (bar_x, bar_y + bar_h - 1), (bar_x + bar_w - 1, bar_y + bar_h - 1))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (bar_x, bar_y), (bar_x, bar_y + bar_h - 1))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_SHD, (bar_x + bar_w - 1, bar_y), (bar_x + bar_w - 1, bar_y + bar_h - 1))

            # Plinth collar connecting cross to stele
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_MIDHI, (cx - shaft_w, int(base_y - 3), shaft_w * 2, 2))
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_BODY, (cx - shaft_w - 2, int(base_y - 1), shaft_w * 2 + 4, 2))

            # --- 2. Upright Headstone Stele ---
            stele_h = int(base_h * 0.55)
            stele_w = int(w * 0.82)
            stele_x = int(cx - stele_w / 2)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_DARK, (stele_x, int(base_y + 2), stele_w, int(stele_h - 2)))
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_DARK, (stele_x + 2, int(base_y), stele_w - 4, 2))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (stele_x + 2, int(base_y)), (stele_x + stele_w - 3, int(base_y)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_MIDHI, (stele_x, int(base_y + 2)), (stele_x, int(base_y + stele_h - 1)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_SHD, (stele_x + stele_w - 1, int(base_y + 2)), (stele_x + stele_w - 1, int(base_y + stele_h - 1)))

            # Inscribed gold border & epitaph motifs
            if stele_h >= 10:
                pygame.draw.rect(canvas, self.COLOR_TOMB_GOLD_SHD, (stele_x + 3, int(base_y + 3), stele_w - 6, stele_h - 5), 1)
                pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_HI, (stele_x + 4, int(base_y + 4)), (stele_x + stele_w - 5, int(base_y + 4)))
                # Mini gold cross
                pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx, int(base_y + 5)), (cx, int(base_y + 8)))
                pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx - 1, int(base_y + 6)), (cx + 1, int(base_y + 6)))
                # Inscription text lines
                if stele_h >= 14:
                    pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx - 4, int(base_y + 10)), (cx + 4, int(base_y + 10)))
                    pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_SHD, (cx - 3, int(base_y + 12)), (cx + 3, int(base_y + 12)))

            # --- 3. Stepped Granite Tomb Base with Grass Patch ---
            slab_y = base_y + stele_h
            slab_h = (y + h) - slab_y
            step1_w = int(w * 0.90)
            step1_h = max(3, int(slab_h * 0.50))
            step1_x = int(cx - step1_w / 2)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_BODY, (step1_x, int(slab_y), step1_w, step1_h))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (step1_x, int(slab_y)), (step1_x + step1_w - 1, int(slab_y)))

            # Rectangular grass turf on center of tomb slab
            grass_w = int(step1_w * 0.45)
            grass_x = int(cx - grass_w / 2)
            grass_h = max(2, step1_h - 2)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRASS, (grass_x, int(slab_y + 1), grass_w, grass_h))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRASS_HI, (grass_x, int(slab_y + 1)), (grass_x + grass_w - 1, int(slab_y + 1)))

            # Scroll corners (tai bia đá)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_MIDHI, (step1_x, int(slab_y), 2, step1_h))
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_SHD, (step1_x + step1_w - 2, int(slab_y), 2, step1_h))

            # Bottom foundation slab
            step2_y = slab_y + step1_h
            step2_h = max(2, int((y + h) - step2_y))
            step2_w = int(w)
            step2_x = int(x)
            pygame.draw.rect(canvas, self.COLOR_TOMB_GRANITE_DARK, (step2_x, int(step2_y), step2_w, step2_h))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_HI, (step2_x, int(step2_y)), (step2_x + step2_w - 1, int(step2_y)))
            pygame.draw.line(canvas, self.COLOR_TOMB_GRANITE_SHD, (step2_x, int(step2_y + step2_h - 1)), (step2_x + step2_w - 1, int(step2_y + step2_h - 1)))

        elif obs.style == "platform":
            # ----------------------------------------------------
            # Floating Stone Platform (Mario-style Stone Block)
            # Weathered masonry block with top stone bevel, mortise
            # joints, and subtle moss accents.
            # ----------------------------------------------------
            ix, iy, iw, ih = int(x), int(y), int(w), int(h)
            # Body
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_MID, (ix, iy, iw, ih))
            # 3D bevel highlights: top and left
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_HI, (ix, iy), (ix + iw - 1, iy), 2)
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_HI, (ix, iy), (ix, iy + ih - 1))
            # 3D bevel shadows: bottom and right
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_DEEP, (ix, iy + ih - 1), (ix + iw - 1, iy + ih - 1), 2)
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_DEEP, (ix + iw - 1, iy), (ix + iw - 1, iy + ih - 1))
            # Stone masonry brick seam
            mid_seam_x = ix + iw // 2
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_SHD, (mid_seam_x, iy + 3), (mid_seam_x, iy + ih - 3))
            # Weathered moss dots
            pygame.draw.circle(canvas, self.COLOR_TOMB_MOSS, (ix + 4, iy + 3), 1)
            pygame.draw.circle(canvas, self.COLOR_TOMB_MOSS, (ix + iw - 5, iy + ih - 4), 1)

        else:
            # ----------------------------------------------------
            # Traditional East Asian / Vietnamese Buddhist Tomb
            # 2-tier curved pagoda roofs (mái đao cong), lotus finial,
            # dragon pillars, and prominent radiant chữ VẠN 卍.
            # ----------------------------------------------------
            cx = int(x + w / 2)
            scale_y = h / 64.0
            scale_x = w / 36.0

            # 1. LOTUS BUD FINIAL (Nụ hoa sen đá)
            lotus_h = int(13 * scale_y)
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_MID, (cx - 4, int(y + lotus_h - 3), 8, 3))
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_HI, (cx - 4, int(y + lotus_h - 3)), (cx + 3, int(y + lotus_h - 3)))

            bud_pts = [
                (cx - 1, int(y + 2)), (cx, int(y + 2)),
                (cx + 4, int(y + lotus_h * 0.45)), (cx + 3, int(y + lotus_h - 4)),
                (cx - 4, int(y + lotus_h - 4)), (cx - 5, int(y + lotus_h * 0.45))
            ]
            pygame.draw.polygon(canvas, self.COLOR_TOMB_STONE_MID, bud_pts)
            # Golden center vein line & tip spark
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx, int(y + 3)), (cx, int(y + lotus_h - 4)))
            pygame.draw.circle(canvas, self.COLOR_TOMB_GOLD_HI, (cx, int(y + 2)), 1)

            # 2. UPPER PAGODA ROOF (Mái đao tầng 1)
            r1_y = y + lotus_h
            r1_h = int(10 * scale_y)
            r1_pts = [
                (cx - int(3 * scale_x), int(r1_y)),
                (cx + int(2 * scale_x), int(r1_y)),
                (cx + int(9 * scale_x), int(r1_y + r1_h * 0.7)),
                (cx + int(12 * scale_x), int(r1_y + r1_h * 0.45)),
                (cx + int(11 * scale_x), int(r1_y + r1_h)),
                (cx - int(12 * scale_x), int(r1_y + r1_h)),
                (cx - int(13 * scale_x), int(r1_y + r1_h * 0.45)),
                (cx - int(10 * scale_x), int(r1_y + r1_h * 0.7)),
            ]
            pygame.draw.polygon(canvas, self.COLOR_TOMB_STONE_MID, r1_pts)
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_HI, (cx - int(11 * scale_x), int(r1_y + r1_h - 1)), (cx + int(10 * scale_x), int(r1_y + r1_h - 1)))
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_DEEP, (cx - int(11 * scale_x), int(r1_y + r1_h)), (cx + int(10 * scale_x), int(r1_y + r1_h)))

            # 3. ARCHITRAVE (Cổ diêm)
            arch_y = r1_y + r1_h
            arch_h = max(2, int(4 * scale_y))
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_SHD, (cx - int(7 * scale_x), int(arch_y), int(14 * scale_x), arch_h))
            pygame.draw.circle(canvas, self.COLOR_TOMB_GOLD_SHD, (cx, int(arch_y + arch_h / 2)), 1)

            # 4. LOWER PAGODA ROOF (Mái đao tầng 2)
            r2_y = arch_y + arch_h
            r2_h = int(10 * scale_y)
            r2_pts = [
                (cx - int(6 * scale_x), int(r2_y)),
                (cx + int(5 * scale_x), int(r2_y)),
                (cx + int(14 * scale_x), int(r2_y + r2_h * 0.6)),
                (cx + int(17 * scale_x), int(r2_y + r2_h * 0.3)),
                (cx + int(16 * scale_x), int(r2_y + r2_h - 1)),
                (cx + int(14 * scale_x), int(r2_y + r2_h)),
                (cx - int(15 * scale_x), int(r2_y + r2_h)),
                (cx - int(17 * scale_x), int(r2_y + r2_h - 1)),
                (cx - int(18 * scale_x), int(r2_y + r2_h * 0.3)),
                (cx - int(15 * scale_x), int(r2_y + r2_h * 0.6)),
            ]
            pygame.draw.polygon(canvas, self.COLOR_TOMB_STONE_MID, r2_pts)
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_HI, (cx - int(15 * scale_x), int(r2_y + r2_h - 1)), (cx + int(14 * scale_x), int(r2_y + r2_h - 1)))
            pygame.draw.line(canvas, self.COLOR_TOMB_STONE_DEEP, (cx - int(15 * scale_x), int(r2_y + r2_h)), (cx + int(14 * scale_x), int(r2_y + r2_h)))

            # 5. SHRINE NICHE & DRAGON PILLARS
            niche_y = r2_y + r2_h
            niche_h = int(14 * scale_y)
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_MID, (cx - int(14 * scale_x), int(niche_y), int(28 * scale_x), niche_h))

            # Dark Cavity
            pygame.draw.rect(canvas, self.COLOR_TOMB_NICHE_BG, (cx - 6, int(niche_y + 1), 12, niche_h - 1))

            # Side pillars
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_SHD, (cx - int(13 * scale_x), int(niche_y), 5, niche_h))
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_HI, (cx - int(12 * scale_x), int(niche_y + 1), 3, niche_h - 2))
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_SHD, (cx + int(8 * scale_x), int(niche_y), 5, niche_h))
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_HI, (cx + int(9 * scale_x), int(niche_y + 1), 3, niche_h - 2))

            # Dragon pillar carvings/dots
            for dy in [3, 7, 11]:
                pygame.draw.circle(canvas, self.COLOR_TOMB_STONE_DEEP, (cx - int(11 * scale_x), int(niche_y + dy * scale_y)), 1)
                pygame.draw.circle(canvas, self.COLOR_TOMB_STONE_DEEP, (cx + int(10 * scale_x), int(niche_y + dy * scale_y)), 1)

            # RADIANT CHỮ VẠN 卍 (Sauwastika)
            van_y = int(niche_y + 5 * scale_y)
            # Central cross bars
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx - 1, van_y - 2), (cx - 1, van_y + 2))
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_MID, (cx - 3, van_y), (cx + 1, van_y))
            # 4 bent arms of 卍
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_HI, (cx - 2, van_y - 2), (cx - 3, van_y - 2))  # Top -> Left
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_HI, (cx - 3, van_y + 1), (cx - 3, van_y + 2))  # Left -> Down
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_HI, (cx, van_y + 2), (cx + 1, van_y + 2))      # Bottom -> Right
            pygame.draw.line(canvas, self.COLOR_TOMB_GOLD_HI, (cx + 1, van_y - 1), (cx + 1, van_y - 2))  # Right -> Up
            pygame.draw.circle(canvas, self.COLOR_TOMB_GOLD_HI, (cx - 1, van_y), 1)

            # Miniature bronze incense burner & red tips
            pot_y = int(niche_y + 10 * scale_y)
            pygame.draw.rect(canvas, self.COLOR_TOMB_INCENSE_POT, (cx - 3, pot_y, 6, 3))
            pygame.draw.circle(canvas, self.COLOR_TOMB_INCENSE_RED, (cx - 1, pot_y - 1), 1)
            pygame.draw.circle(canvas, self.COLOR_TOMB_INCENSE_RED, (cx + 1, pot_y - 1), 1)

            # 6. ALTAR SLAB & BASE PEDESTAL
            base_y = niche_y + niche_h
            base_h = (y + h) - base_y
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_HI, (cx - int(15 * scale_x), int(base_y), int(30 * scale_x), 3))
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_MID, (cx - int(14 * scale_x), int(base_y + 3), int(28 * scale_x), max(2, int(base_h - 5))))

            # Center medallion
            if base_h >= 8:
                pygame.draw.ellipse(canvas, self.COLOR_TOMB_STONE_SHD, (cx - 4, int(base_y + 4), 8, 4), 1)
                pygame.draw.circle(canvas, self.COLOR_TOMB_GOLD_SHD, (cx, int(base_y + 6)), 1)

            # Foundation slab
            pygame.draw.rect(canvas, self.COLOR_TOMB_STONE_SHD, (int(x), int(y + h - 3), int(w), 3))
            pygame.draw.circle(canvas, self.COLOR_TOMB_MOSS, (int(x + 3), int(y + h - 2)), 1)
            pygame.draw.circle(canvas, self.COLOR_TOMB_MOSS, (int(x + w - 4), int(y + h - 2)), 1)

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
