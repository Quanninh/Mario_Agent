# Visual & Game Design Spec — Hoá's Midnight Escape

> Theme, color palette, Pygame visual components, and level balance rules for Member 1 (`env/`).

---

## 1. Story & Atmosphere
- **Setting**: A cold, quiet graveyard at midnight.
- **Protagonist**: **Hoá**, who lives near the graveyard, wakes up stranded inside it at night—terrified and freezing—and runs toward the warm glow of his home.
- **Visual Style**: Minimalist 2D vector/pixel-clean Pygame shapes (zero external image/font asset dependencies so the game runs identically on every machine).

---

## 2. Color Palette (RGB)

| Element | Role | RGB Value | Visual Feel |
|:---|:---|:---|:---|
| **Midnight Sky** | Background | `(15, 20, 45)` | Deep cold night blue |
| **Full Moon** | Sky backdrop | `(245, 240, 200)` | Soft pale moonlight |
| **Stars** | Sky accents | `(210, 220, 255)` | Subtle twinkling dots |
| **Graveyard Soil** | Ground | `(38, 46, 44)` | Dark damp earth |
| **Grass / Moss Trim** | Top edge of ground | `(52, 78, 60)` | Dark misty graveyard grass |
| **Tombstone Stone** | Static Obstacles | `(115, 122, 136)` | Weathered stone gray |
| **Tombstone Detail** | Cross / border | `(75, 80, 92)` | Carved engraving shadow |
| **Ghost Body** | Hovering Enemies | `(215, 242, 255)` | Pale eerie spectral glow |
| **Ghost Eyes** | Ghost detail | `(20, 25, 45)` | Dark hollow eyes |
| **Warmth Item (Glow / Core)** | Collectible (`WarmthItem`) | `(255, 195, 65)` / `(255, 245, 180)` | Warm golden ember glow |
| **Hoá (Jacket / Head)** | Player (`Player`) | `(65, 145, 230)` / `(245, 210, 180)` | Distinct contrast against dark night |
| **House Wall / Roof / Door** | Goal (`GoalFlag`) | `(185, 135, 90)` / `(145, 50, 50)` / `(100, 65, 40)` | Cozy wooden/brick home |
| **House Window** | Goal Beacon | `(255, 230, 115)` | Warm bright yellow light beckoning Hoá home |

---

## 3. Component Visuals & Hitboxes

- **Window & Camera**: `800 x 400` px window, `DEFAULT_FLOOR_Y = 320.0`, horizontal camera keeps Hoá around `30%` from the left edge (`camera_x = max(0, x - width * 0.3)`).
- **Hoá (`Player` — `24 x 32` px)**: Simple character silhouette (head + winter jacket body + eye facing movement direction via `facing_right`).
- **Tombstone (`Obstacle` — `32 x [44..68]` px)**: Gray stone body with a rounded arched top and an engraved cross (`+`).
- **Open Grave Pit (`Pit` — `65..75` px wide)**: Dark void cutting through the ground strip.
- **Floating Ghost (`Ghost` — `24 x 28` px)**: Rounded top dome, wavy 3-circle bottom tail, two dark hollow eyes. Patrols horizontally between `[patrol_min_x, patrol_max_x]` with sinusoidal vertical bobbing (`_base_y + float_amplitude * sin(_phase)`). Each ghost supports custom `speed`, `float_amplitude`, and `float_speed`.
  - *Low ghost* (`y ≈ floor_y - 38..40`): Hoá must **jump over**.
  - *High ghost* (`y ≈ floor_y - 82..86`): Hoá can **walk underneath**.
- **Warmth Collectible (`WarmthItem` — `14 x 14` px)**: Warm golden glowing orb (outer amber halo + bright core) placed along paths, above tombstones, or over pits.
- **Hoá's House (`GoalFlag` — `64 x 88` px, `y = floor_y - 88`)**: House wall, triangular red roof, wooden front door, and bright yellow glowing window.

---

## 4. Physics & Level Balance Constraints
- **Constants**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0`.
- **Max Jump Height**: $\approx 105\text{ px}$ $\to$ keep tombstones $\le 72\text{ px}$ tall.
- **Max Jump Span**: $\approx 130\text{ px}$ $\to$ keep grave pits $\le 80\text{ px}$ wide.
- **5-Level Curriculum Progression** (`env/map_levels.py`):
  - **Level 1 (`Quiet Graveyard`)**: 2 tombstones, 0 pits, 0 ghosts, 0 items — sanity-check basic jumping.
  - **Level 2 (`Open Graves Pits`)**: 3 tombstones + 2 grave pits, 0 ghosts, 0 items — introduces pit jumping & camera scrolling.
  - **Level 3 (`Cold Trail`)**: Shifted tombstones + pits + **3 warmth items** — introduces collectible items & layout generalization.
  - **Level 4 (`Whispering Graves`)**: 3 tombstones + 2 pits + 3 warmth items + **2 uniform ghosts** (same default `speed=1.5`, `float_amplitude=8.0`, `float_speed=0.08`) — introduces low & high ghost dodging.
  - **Level 5 (`Midnight Escape`)**: 4 tombstones + 2 pits + 4 warmth items + **4 diverse ghosts** (distinct speeds `1.0` to `2.4`, amplitudes `4.0` to `13.0`, wave speeds `0.05` to `0.14`) — full challenge stage.
