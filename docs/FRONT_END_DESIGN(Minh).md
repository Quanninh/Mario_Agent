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
| **Warmth Item** | Collectible (*TBD*) | `(255, 195, 65)` | Warm flickering flame glow |
| **Hoá (Jacket / Head)** | Player | `(65, 145, 230)` / `(245, 210, 180)` | Distinct contrast against dark night |
| **House Wall / Roof** | Goal Destination | `(185, 135, 90)` / `(145, 50, 50)` | Cozy wooden/brick home |
| **House Window** | Goal Beacon | `(255, 230, 115)` | Warm bright yellow light beckoning Hoá home |

---

## 3. Component Visuals & Hitboxes

- **Window & Camera**: `800 x 400` px window, `DEFAULT_FLOOR_Y = 320.0`, horizontal camera keeps Hoá around `30%` from the left edge (`camera_x = max(0, x - width * 0.3)`).
- **Hoá (Player — `24 x 32` px)**: Simple character silhouette (head + winter jacket body + eyes facing movement direction).
- **Tombstone (Obstacle — `32 x [44..68]` px)**: Gray stone body with a rounded arched top and an engraved cross (`+`).
- **Open Grave Pit (`65..75` px wide)**: Dark void cutting through the ground strip.
- **Floating Ghost (`24 x 28` px — *v2*)**: Rounded top dome, wavy bottom tail, two dark eyes. Patrols horizontally between `[patrol_min_x, patrol_max_x]` with gentle vertical floating (`base_y + amplitude * sin(phase)`).
  - *Low ghost*: Hoá must **jump over**.
  - *High ghost*: Hoá can **walk underneath**.
- **Warmth Collectible (`14 x 14` px — *v2*)**: Small warm golden glow floating along safe or jump-arc paths *(exact theme TBD: Candle / Matchstick / Lantern)*.
- **Hoá's House (`64 x 88` px visual, goal trigger at front door)**: House rectangle, triangular roof, front door, and bright yellow glowing window.

---

## 4. Physics & Level Balance Constraints
- **Constants**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0`.
- **Max Jump Height**: $\approx 105\text{ px}$ $\to$ keep tombstones $\le 72\text{ px}$ tall.
- **Max Jump Span**: $\approx 130\text{ px}$ $\to$ keep grave pits $\le 80\text{ px}$ wide.
- **Curriculum Progression**:
  - **Level 1 (Baseline)**: 2 tombstones, 0 pits, 0 ghosts (sanity-check basic jumping).
  - **Level 2 (Standard)**: Tombstones + grave pits + hovering ghosts + warmth items.
  - **Level 3 (Generalization Test)**: Different order/spacing of pits, tombstones, and ghosts to test RL generalization.
