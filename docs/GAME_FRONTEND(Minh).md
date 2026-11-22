# Member 1: Environment & Physics Progress Log

> **Owner**: Member 1 (Minh)  
> **Role**: Environment & Game Physics Lead  
> **Game Concept**: **Hoá's Midnight Escape** *(2D Graveyard Platformer)*  
> **Scope**: 2D Platformer Engine (Pygame), Physics, Collisions, Levels, and Human Test Mode  

---

## 1. Role Objective & Creative Theme

Build a lightweight, headless-compatible 2D platformer environment implementing the `BaseEnvironment` interface. The environment simulates physical movement, gravity, collisions, hovering ghosts, warmth collectibles, and proximity-based state observations for the Tabular Q-Learning agent.

### Story & Theme — *Hoá's Midnight Escape*
Hoá lives near a graveyard. Somehow, he wakes up in the middle of the graveyard late at night—freezing cold and terrified—and desperately tries to find his way back home:

| Story Element | Class / Role in Engine | Behavior / Mechanics |
|:---|:---|:---|
| **Hoá** | `Player` (Agent) | Walks left/right and jumps (`STAY`, `LEFT`, `JUMP`, `RIGHT`, `RIGHT_JUMP`). |
| **Cold Midnight Graveyard** | `GameEnvironment` | Dark night sky, full moon, stars, and shadowy graveyard soil. |
| **Tall Graves / Tombstones** | `Obstacle` | Block the path; Hoá must jump over or land on top of them. |
| **Open Grave Pits** | `Pit` | Bottomless gaps; falling in ends the episode (`fell_in_pit=True`, `done=True`). |
| **Floating Ghosts** | `Ghost` | Hover and patrol in mid-air with customizable speed & sinusoidal wave; touching one ends the episode (`hit_ghost=True`, `done=True`). Jump over low ghosts or pass beneath high ghosts. |
| **Warmth Item** | `WarmthItem` | Golden glowing warmth collectible along the path (`collected_item=True`), satisfying Project 5's *"collecting useful items"* requirement. |
| **Hoá's House** | `GoalFlag` | Warm, glowing house with lit yellow window at the end of the stage (`reached_goal=True`, `done=True`). |

---

## 2. Key Design Decisions

- **Action Space**: 5 discrete actions (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`).
- **State Representation**: Uses **local vision / relative proximity** (tracking nearest upcoming tombstone, pit, ghost + altitude, warmth item, and home) rather than full-level coordinates, keeping the state space compact for Q-Learning.
- **Velocity Simplification**: Excluded horizontal sliding velocity (`velocity_x`) from `RawState` and `Player` since movement strictly mirrors current input. Vertical motion uses `velocity_y` + `is_grounded` for gravity and jump control.
- **SOLID Compliance**: High-level modules depend on `core.BaseEnvironment`. Level layouts use a Registry Pattern (`LEVEL_BUILDERS`) with `DEFAULT_FLOOR_Y = 320.0`, enabling progressive 5-level curriculum design without touching core physics.

---

## 3. Implementation Checklist

### Phase 1 — Core Platformer Engine (v1 — Complete)
- [x] **Step 1: Core Abstraction & Contracts**
  - [x] `core/base_env.py`: Defined `Action`, `RawState` (without `velocity_x`), `GameEvents`, and `BaseEnvironment`.
  - [x] `core/__init__.py`: Package export for clean imports.
- [x] **Step 2: Base Game Entities** (`env/entities.py` & `env/__init__.py`)
  - [x] Player, Obstacle, Pit, GoalFlag dataclasses with hitbox properties (`rect`, `right`, `bottom`).
- [x] **Step 3: Environment Engine**
  - [x] Gravity (`0.8`), jump impulse (`-13.0`), walk speed (`4.0`), axis-separated X/Y collisions, pit anti-teleport guard (`old_bottom <= floor_y`), and lazy Pygame rendering with camera follow.
- [x] **Step 4: Modular Level Registry** (`env/map_levels.py`)
  - [x] `LevelConfig`, `LEVEL_BUILDERS`, `DEFAULT_FLOOR_Y = 320.0`, `get_level()`, `create_env()`.
- [x] **Step 5: Playable Human Mode** (`play_human.py`)
  - [x] Keyboard controls (`A`/`D`/`Space`, `R` reset, `Esc` quit) and fixed `R` reset window freeze by calling `env.render()` before `continue`.

### Phase 2 — Hoá's Midnight Escape & Dynamic Hazards (v2 — Complete)
- [x] **Step 6: Graveyard Theme, Ghosts, Warmth Items & 5-Level Curriculum**
  - [x] Updated `core/base_env.py`: Renamed `mario_x`/`mario_y` $\to$ `player_x`/`player_y`; added `dist_to_ghost`, `ghost_y`, `dist_to_item` to `RawState` and `hit_ghost`, `collected_item` to `GameEvents`.
  - [x] Updated `env/entities.py`: Renamed `Mario` $\to$ `Player` (added `facing_right`); added `Ghost` (patrol bounds, customizable `speed`, `float_amplitude`, `float_speed`, wrapped `_phase`, `update()`, `reset()`) and `WarmthItem` (`is_collected`, `reset()`). Updated `GoalFlag` size to `64x88`.
  - [x] Renamed `env/mario_env.py` $\to$ `env/game_env.py` (`GameEnvironment`): Integrated ghost updates/lethal collisions, warmth item pickups, local-vision ghost/item tracking, and full midnight graveyard Pygame drawing helpers (`_draw_background`, `_draw_tombstone`, `_draw_ghost`, `_draw_item`, `_draw_home`, `_draw_player`).
  - [x] Expanded `env/map_levels.py` to **5 progressive levels**: Levels 1–2 (baseline & pits), Level 3 (shifted layout + warmth items), Level 4 (2 uniform ghosts + items), Level 5 (4 diverse ghosts with distinct speeds & wave patterns + items).
  - [x] Updated `play_human.py` for live switching across Levels `1–5`, `state.player_x`, warmth pickup logs, and ghost/pit death feedback.

### Phase 3 — Verification & Team Hand-off
- [ ] **Step 7: Headless Verification & PR to `main`**
  - [x] Add automated headless test (`tests/test_env.py`) — all 6/6 tests pass in `0.022s` (smoke test across Levels 1–5, full reset check, item pickup, ghost death, pit death, home win).
  - [x] Add `.gitignore` to exclude `__pycache__/`, `*.pyc`, virtualenvs, and `.DS_Store`.
  - [ ] Merge `game_frontend` into `main` so Members 2, 3, and 4 can integrate against v2.

### Phase 4 — v3 Enhancement Backlog *(Sorted: Easiest $\to$ Most Complex)*

> [!TIP]
> **Tier 1 & Tier 2 (#1–#8)** do **not** break `core/base_env.py` and are safe to build without disrupting Members 2–4.  
> **Tier 3 (#9–#12)** changes `Action` / `RawState` / `GameEvents`—coordinate with Members 2 & 3 (or branch after v2 is merged to `main`).

#### Tier 1 — Quick Visual & HUD Polish *(Easiest, Zero RL Contract Impact)*
- [x] **1. Larger Game Window**: Scaled game display canvas to `1400x700` via Pygame surface scaling (`pygame.transform.scale`), keeping internal physics canvas fixed at `800x400` so physics, hitboxes, and unit tests remain 100% stable.
- [x] **2. Display `"Hoá"` Label Above Player**: Rendered `"Hoá"` floating directly above Hoá's head (`y - 16` px) with high-contrast drop shadow in `_draw_player()`.
- [x] **3. Display Current Level Name HUD**: Wired `level_name` through `create_env()` and rendered it at the top-left corner (`x=16, y=14`).
- [x] **4. Warmth Counter & Countdown Steps Left HUD**: Stacked top-left HUD showing level name, remaining steps countdown (`Steps Left: 980`, turning warning red/orange at $\le 150$), and collected warmth items (`Warmth: 0/4`). Fixed star positioning to prevent background overlap.
- [x] **5. Diverse Tombstone Styles**: Added visual variants for Christian cross stone monuments (`style="cross"`) with base pedestals and horizontal crossbars, and classic arch tombstones engraved with Buddhist sauwastikas (`style="buddhist"` / *chữ Vạn* `卍`).
- [x] **6. Brief Game Over / Win Overlay Before Reset**: Implemented semi-transparent dark banner overlay pause before level reset in `play_human.py` with instant skip on `R`, `Space`, or `Enter` *(keeping `GameEnvironment.step()` 100% instant for headless RL training)*.

#### Tier 2 — Clean Architecture & Goal Optimization *(Resolved)*
- [x] **7. SOLID Dynamic Level Architecture**: Fully resolved via the Registry Pattern (`LEVEL_BUILDERS`) in `env/map_levels.py`, allowing unlimited curriculum stages to be added without modifying the core simulation engine.
- [x] **8. Win with Fewest Steps & Most Warmth Items**: Successfully delegated to **Member 3's `RewardStrategy`** (negative step penalty `-1` + positive warmth reward `+30` + goal bonus `+100`), ensuring the Q-Learning agent naturally converges to the optimal path without redundant client-side UI code.

#### Tier 3 — New Gameplay & Dynamic Hazard Mechanics *(Active Focus)*
- [x] **9. Mid-air Double Jump & Ledge Recovery (`can_double_jump`)**: Allow Hoá to execute a second jump while airborne (`not is_grounded`), allowing recovery from miscalculated pit leaps and dodging high hovering ghosts. Also grants 1 recovery jump if walking off a ledge into the air. Reuses existing `Action.JUMP` and `Action.RIGHT_JUMP` without expanding the action space, while exposing a binary `can_double_jump: bool` flag in `RawState` for the RL agent.
- [x] **10. Player Crouch & Momentum Slide (`Action.CROUCH`)**: Lowers Hoá's hitbox height from $32\text{px} \to 18\text{px}$ (`S` / `Down` key) to duck under hovering ghosts. Implements physical ducking in place when stationary and momentum-based sliding when running ($v_{x, t} = v_{x, t-1} \cdot 0.82$), keeping feet anchored to the ground *(expands `Action.NUM_ACTIONS` from 5 to 6)*.
- [x] **11. Fire Erupting from Grave Pits**: Periodic flame jets shooting upward out of selected pits on a cycle (`idle -> warning glow -> active lethal flame -> cooldown`), with pit-fire phase exposed in `RawState` so the AI can time its jumps. Hitbox spans the full pit width and depth from ground flame height down to canvas bottom: `(pit.x_start, floor_y - FIRE_HEIGHT, pit.x_end - pit.x_start, canvas_height - (floor_y - FIRE_HEIGHT))`.
- [ ] **12. Ghosts Shooting Rotating Eye-Laser Beams**: Opt-in per ghost (`has_laser=True`) — fires a lethal laser beam from the ghost's eyes that **rotates continuously in a circle (`0..2π`) while the ghost patrols** (using `(eye_x + L*cos(θ), eye_y + L*sin(θ))`, pure-Python line-segment vs. player AABB collision for headless speed, and nearest-laser angle/danger flag in `RawState`).
- [ ] **13. Rising & Charging Skeleton Ambush from Tombs**: Special mausoleum tombs that trigger when Hoá approaches (proximity $< 180\text{px}$): a skeletal warrior bursts upward from the grave, leaps forward, and sprints aggressively toward the player (`dx = player.x - skeleton.x`), demanding quick reaction timing or a high double jump to evade. Requires a new `Skeleton` entity with ground physics, pursuit logic, and nearest skeleton proximity in `RawState`.

---

## 4. Work Log

- **Step 1**: Defined `Action`, `RawState`, `GameEvents`, and `BaseEnvironment` in `core/base_env.py`. Verified package initialization with `core/__init__.py`. Removed redundant `velocity_x` from `RawState`.
- **Step 2**: Implemented physical entities in `env/entities.py` and `env/__init__.py` package exports.
- **Step 3**: Built 2D platformer engine with axis-separated AABB collisions, pit-edge anti-teleport guard (`old_bottom <= floor_y`), local-vision `_get_raw_state()`, and lazy-initialized Pygame rendering with camera follow.
- **Step 4**: Built modular level registry in `env/map_levels.py` using `DEFAULT_FLOOR_Y = 320.0`.
- **Step 5**: Created `play_human.py` manual keyboard runner and fixed the macOS window freeze on `R` reset by pumping Pygame events via `env.render()`. Committed v1 on `game_frontend`.
- **Step 6**: Completed full v2 upgrade (**Hoá's Midnight Escape**): refactored `MarioEnvironment` $\to$ `GameEnvironment` (`env/game_env.py`) and `Mario` $\to$ `Player`; added hovering `Ghost` and `WarmthItem`; updated `RawState`/`GameEvents`; built midnight graveyard Pygame visuals; expanded `env/map_levels.py` to 5 progressive levels (uniform ghosts in Level 4, diverse ghosts in Level 5, items from Level 3 onwards); updated `play_human.py` for Levels `1–5` and play-tested successfully.
- **Step 7**: Added headless unit tests in `tests/test_env.py` (6/6 passing in `0.022s`) and created `.gitignore`. Documented the prioritized 13-item v3 Enhancement Backlog (Tier 1–3, easiest to most complex).
- **Step 8**: Implemented Tier 1 visual & HUD overhaul:
  - Canvas window scaling to `1400x700` (`canvas_width=800, canvas_height=400` logic scaled via `pygame.transform.scale`).
  - Rendered `"Hoá"` floating name tag with contrast drop-shadow above player head.
  - Implemented fixed 3-line top-left HUD (`level_name`, countdown `Steps Left` with warning coloring at $\le 150$, and `Warmth: X/Y` item counter).
  - Repositioned star coordinates to `(70, 75)` to eliminate background overlap.
  - Added Mid-air Double Jump (`can_double_jump`) design specification to roadmap and root `README.md`.
  - Re-verified all 6 headless unit tests pass cleanly in `0.020s`.
- **Step 9**: Implemented diverse tombstones and game over / victory overlay:
  - Added `style` attribute on `Obstacle` supporting `"cross"` (stone cross monuments) and `"buddhist"` (arch tombstones with engraved `卍` sauwastikas).
  - Implemented semi-transparent pause overlay in `play_human.py` with instant key-press skipping (`R`, `Space`, `Enter`). All 6 headless unit tests remain 100% passing.
- **Step 10**: Implemented Mid-air Double Jump & Ledge-Slip Recovery:
  - Extended `core/base_env.py` `RawState` with `can_double_jump: bool = False` (maintaining 5-action space for Member 2 & 3).
  - Added `can_double_jump: bool = False` to `Player` entity in `env/entities.py`.
  - Added `DOUBLE_JUMP_RATIO = 0.88` to `GameEnvironment` in `env/game_env.py`. Enabled double jump on initial jump and granted 1 recovery jump on stepping off ledges (`was_grounded and not is_grounded`). Reset to `False` upon landing on obstacles or the ground.
  - Implemented tap-only air jump input detection in `play_human.py`.
  - Expanded test suite in `tests/test_env.py` from 6 to 9 tests (`test_double_jump_and_recovery`, `test_ledge_slip_recovery_over_pit`, `test_pit_death_after_double_jump`), all passing cleanly in `0.026s`.
- **Step 11**: Implemented Momentum-Based Crouch & Slide Mechanics:
  - Expanded `Action` space to 6 (`Action.CROUCH = 5`, `NUM_ACTIONS = 6`).
  - Reorganized `RawState` into 2 clean logical groups (Kinematics & Proximity) without dataclass argument ordering issues, adding `velocity_x: float` and `is_crouching: bool`.
  - Added `vx: float = 0.0` and `is_crouching: bool = False` to `Player` in `env/entities.py`.
  - Added physics constants in `env/game_env.py`: `SLIDE_FRICTION = 0.90`, `STANDING_HEIGHT = 32.0`, `CROUCH_HEIGHT = 18.0`.
  - Implemented physical crouch logic: ducks in place when still ($v_x = 0$), momentum-slides with friction when running ($v_{x, t} = v_{x, t-1} \cdot 0.82$), and smoothly anchors feet to the floor during transitions.
  - Mapped `S` / `Down Arrow` in `play_human.py` for crouch/slide control.
  - Added `test_crouch_and_momentum_slide` in `tests/test_env.py`, bringing the automated test suite to 10/10 passing tests in `0.043s`.
- **Step 12**: Implemented Periodic Fire Erupting from Grave Pits:
  - Added flame timing attributes and helpers on `Pit` in `env/entities.py`: `has_fire`, `fire_offset`, `FIRE_CYCLE = 100`, `FIRE_WARN_START = 65`, `FIRE_ACTIVE_START = 80`, `FIRE_HEIGHT = 40.0`, `get_fire_phase()`, and `get_flame_rect()`.
  - Hitbox spans full pit width and depth: `(pit.x_start, floor_y - FIRE_HEIGHT, pit.x_end - pit.x_start, canvas_height - (floor_y - FIRE_HEIGHT))`.
  - Added `hit_fire: bool = False` to `GameEvents` in `core/base_env.py`.
  - Added `pit_fire_state: int` (0: safe, 1: warning, 2: active) in `RawState` for agent observation.
  - Implemented multi-layered flame rendering in `env/game_env.py`: glowing amber embers during warning phase, crimson pillar + bright incandescent core + pointed tip during active phase.
  - Added death event handling and `"INCINERATED! Burned by the grave's inferno!"` overlay banner in `play_human.py`.
  - Configured flame pits in `env/map_levels.py`: Level 4 Pit 2, Level 5 Pit 1 & 2 (with staggered 50-tick offset for alternating eruption).
  - Added 3 unit tests (`test_pit_fire_phases_and_bounding_box`, `test_pit_fire_lethal_collision`, `test_pit_fire_safe_crossing_during_dormant_phase`), bringing automated test suite to 13/13 passing tests in `0.041s`.
- **Step 13**: Decoupled UI Rendering from Physics Simulation Engine (Single Responsibility Principle Refactor):
  - Extracted all Pygame rendering, display scaling, camera offset, color palettes, and drawing methods into a dedicated `GameRenderer` class in `env/game_renderer.py`.
  - Refactored `GameEnvironment` in `env/game_env.py` to be a pure physics simulation engine and MDP contract, completely eliminating Pygame surface and font dependencies.
  - Removed display dimensions (`screen_width`, `screen_height`) from `GameEnvironment`, preserving only physical world logic dimensions (`canvas_width = 800`, `canvas_height = 400`, `floor_y = 320.0`).
  - Added `show_overlay()` and `show_game_over(env, events)` to `GameRenderer`, centralizing all banner presentation, drop-shadow typography, and event-pump pauses.
  - Refactored `play_human.py` into a lightweight controller loop by replacing manual surface drawing with a one-line call to `env.show_game_over(events)`.
  - Exported `GameRenderer` in `env/__init__.py`. All 14 headless unit tests pass cleanly in `0.040s`.
- **Step 14**: Redesigned Pit Fire Aesthetics to 5-Layer Pixel-Art Spires & Scaled Fire Height:
  - Redesigned pit flame graphics in `env/game_renderer.py` to match pixel-art sprite aesthetics: rectangular 5-layer column below floor level (dark maroon border, scarlet red, radiant orange, golden yellow, white-hot center core) with dynamic multi-spire jagged eruption above floor level, sinusoidal flicker, and floating ember sparks.
  - Increased lethal flame height `FIRE_HEIGHT` from `40.0px` to `65.0px` on `Pit` in `env/entities.py`, scaling both physical AABB collision hitbox and visual spires proportionally (~$2\times$ Hoá's standing height).
  - Added grounded magma bed at the bottom of the pit trench (near `canvas_height`) with bubbling ember sparks rising toward ground level during the warning phase (`_draw_pit_warning`).
  - All 14 automated unit tests passing cleanly in `0.038s`.

---

## 5. Module Definitions & Structure *(Current v2 Codebase)*

### 5.1 `core/base_env.py` — Environment Contract & Abstractions
Defines the shared interface and data structures decoupling the physics engine from the RL agent, discretizer, and reward modules (Dependency Inversion Principle).

- **`Action`**: Discrete action constants (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`, `CROUCH=5`, `NUM_ACTIONS=6`).
- **`RawState` *(dataclass)***: Continuous physical snapshot passed to Member 2's discretizer:
  - Player kinematics: `player_x`, `player_y`, `velocity_x`, `velocity_y`, `is_grounded`, `can_double_jump`, `is_crouching`.
  - Local vision ahead: `dist_to_obstacle`, `obstacle_height`, `dist_to_pit`, `dist_to_ghost`, `ghost_y`, `dist_to_item`, `dist_to_goal`.
- **`GameEvents` *(dataclass)***: Per-step outcome flags passed to Member 3's reward strategy (`dx`, `reached_goal`, `fell_in_pit`, `hit_obstacle`, `hit_ghost`, `collected_item`).
- **`BaseEnvironment(ABC)`**: Abstract contract enforcing `reset()`, `step(action)`, `render()`, and `close()`.

### 5.2 `env/entities.py` — Physical Game Objects
Holds lightweight data models and hitbox geometry with zero rendering or RL coupling (Single Responsibility Principle).

- **`Player`**: Hoá's entity (`x`, `y`, hitbox `24x32`, `vx`, `vy`, `is_grounded`, `can_double_jump`, `is_crouching`, `is_alive`, `facing_right`) with `reset(start_x, start_y)` and bounding-box properties (`rect`, `right`, `bottom`).
- **`Obstacle`**: Tall tombstone (`x`, `y`, default `32x48`) with `rect`, `right`, `bottom`.
- **`Pit`**: Bottomless open grave (`x_start`, `x_end`) with `contains(x, width)` center-of-mass check.
- **`Ghost`**: Hovering ghost (`x`, `y`, `patrol_min_x`, `patrol_max_x`, `speed=1.5`, `width=24`, `height=28`, `float_amplitude=8.0`, `float_speed=0.08`) with wrapped sinusoidal wave (`% 2*pi`), `update()`, `reset()`, and `rect`/`right`/`bottom`.
- **`WarmthItem`**: Collectible warmth item (`x`, `y`, `14x14`, `is_collected=False`) with `reset()` and `rect`/`right`/`bottom`.
- **`GoalFlag`**: Hoá's warm home (`x`, `y`, `64x88`) with `is_reached(player)` win detection.

### 5.3 `env/game_env.py` — 2D Physics Engine & Simulation Contract
Concrete `GameEnvironment(BaseEnvironment)` driving physics kinematics, axis-separated AABB collisions, dynamic hazards, state extraction, and headless execution. Free of all Pygame surface drawing code.

- **Physics Constants & World Bounds**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0`, `DOUBLE_JUMP_RATIO = 0.88`, `SLIDE_FRICTION = 0.82`, `STANDING_HEIGHT = 32.0`, `CROUCH_HEIGHT = 18.0`. Physical logic runs on `canvas_width=800, canvas_height=400` with `floor_y=320.0`.
- **Lifecycle Methods**:
  - `reset() -> RawState`: Restores Hoá, all ghosts, all warmth items, and `step_count = 0`.
  - `step(action) -> (RawState, GameEvents, bool)`: Runs axis-separated X/Y AABB collisions, pit-fall guard (`old_bottom <= floor_y`), updates ghosts, checks warmth pickups, checks clean termination (`hit_ghost`, `hit_fire`, `fell_in_pit`, `reached_goal`, `max_steps`), and returns state + events.
  - `render()` / `close()`: Lazily instantiates `GameRenderer` on demand and delegates drawing. Stepping headlessly leaves renderer as `None` for zero overhead.
  - `show_game_over(events)`: Delegates end-screen banner display to the active renderer.
- **Helpers**: `_get_raw_state()` (nearest upcoming tombstone, pit + fire phase, ghost + Y, uncollected item, and home) and `_rects_overlap()`.

### 5.4 `env/game_renderer.py` — Pygame Graphical View & Presentation
Dedicated `GameRenderer` class encapsulating all visual elements, display scaling, and user feedback:

- **Display Management**: Maps internal `800x400` logic canvas to `1400x700` window display with 60 FPS clock synchronization.
- **Midnight Graveyard Aesthetics**: Moon, decorative stars, layered soil/grass, and complete color palette.
- **Layered Entity Drawing**:
  - `_draw_background()`: Sky, moon, stars, soil, grass, grave pits, and multi-layered flame jets (embers in warning phase, crimson pillar + core + peaked tip in active phase).
  - `_draw_tombstone()`: Latin crosses (`"cross"`) and arched Buddhist sauwastikas (`"buddhist"` / `卍`).
  - `_draw_ghost()`: Spectral body, wavy tails, and hollow eyes.
  - `_draw_item()`: Glowing warmth stamina pickups.
  - `_draw_home()`: Sanctuary home with warm yellow window.
  - `_draw_player()`: Directional winter jacket, head, and floating `"Hoá"` nametag with contrast drop shadow.
  - `_draw_hud()`: Level name, remaining steps countdown, and warmth item counter.
- **Interactive UI Banners**: `show_overlay()` (semi-transparent alpha overlay with key-press skipping) and `show_game_over()` (automatic title, subtitle, and color selection based on outcome).

### 5.5 `env/map_levels.py` — 5-Level Curriculum Registry
Separates stage definitions from `GameEnvironment` (Open/Closed Principle) with `DEFAULT_FLOOR_Y = 320.0`.

- **`LevelConfig` *(dataclass)***: Bundles `name`, `obstacles`, `pits`, `goal`, `ghosts`, `items`, `floor_y`, `max_steps`.
- **`LEVEL_BUILDERS`** (`1` to `5`):
  - `1 -> _build_level_1` (*Quiet Graveyard*): 2 tombstones, no pits/ghosts/items (`max_steps=800`).
  - `2 -> _build_level_2` (*Open Graves Pits*): 3 tombstones + 2 pits (`max_steps=1000`).
  - `3 -> _build_level_3` (*Whispering Graves*): 3 tombstones + 2 pits (including 1 hellfire pit) + 2 uniform ghosts (jump-over & walk-under) + 3 warmth items (`max_steps=1000`).
  - `4 -> _build_level_4` (*Nightmare Catacombs*): 5 tombstones + 4 hellfire pits + 5 diverse ghosts (including dual counter-rotating laser scissors) + 5 warmth items (`max_steps=1400`).
  - `5 -> _build_level_5` (*The Eternal Abyss*): Variable flame heights, multi-speed laser ghosts, varied tombs, and mandatory crouch tunnels (`max_steps=2400`).
- **Public Factories**: `get_level(level_id)` and `create_env(level_id)`.

### 5.6 Package Entry Exports (`core/__init__.py`, `env/__init__.py`)
- **`core/__init__.py`**: Exports `Action`, `RawState`, `GameEvents`, `BaseEnvironment`.
- **`env/__init__.py`**: Exports `Player`, `Obstacle`, `Pit`, `Ghost`, `WarmthItem`, `GoalFlag`, `GameEnvironment`, `GameRenderer`, `LevelConfig`, `LEVEL_BUILDERS`, `get_level`, `create_env`.

### 5.7 `play_human.py` — Interactive Playable Entry Point
Lightweight keyboard runner (`A`/`Left`, `D`/`Right`, `Space`/`W`/`Up`, `S`/`Down`) stepping `GameEnvironment` at 60 FPS with live level switching (`1–5`), freeze-safe resets (`R`), automatic victory/death overlays (`env.show_game_over(events)`), and exit (`Esc`).

### 5.8 Traditional Vietnamese Pixel Art & Visual Polish (Phase 3 — Complete)
- **Authentic Khuê Văn Các Destination**:
  - Replaced generic Western house with Hanoi's iconic Temple of Literature pavilion (*Khuê Văn Các*).
  - Four carved white stone pillars with cloud/dragon relief carvings and stone plinths (*chân tảng đá*).
  - Open central archway with three-tier stone courtyard steps (*bậc tam cấp*) and warm amber sanctuary light spilling through.
  - Upper vermilion wooden pavilion (*gác gỗ sơn son thếp vàng*) with turned balustrade (*lan can con tiện đỏ*).
  - Lacquered horizontal board inscribed in gold: "奎文閣".
  - Iconic circular sunburst window (*Cửa Sổ Sao Khuê*) with radiating wooden spokes and glowing amber core.
  - Double-tiered flared curved roofs (*mái chồng diêm 2 tầng 8 mái*) with weathered terracotta tiles, ornamental ridge, cloud dragons (*kìm nóc*), and central gourd symbol (*bình hồ lô*).
- **Hoá Traditional Attire Redesign**:
  - **Áo Ngũ Thân May Kép Đệm Bông**: Quilted cyan-blue long robe with diamond stitching, white inner collar rim (*cổ lập lót trắng*), diagonal closure, and gold clasps.
  - **Khăn Đóng**: Matching cyan-blue pleated turban with hair silhouette and rear knot (*búi tóc*).
  - **Bamboo Lantern & Silk Tassel**: Handheld bamboo lantern with warm interior glow and dynamic swinging red silk tassel.
  - **Lower Garments**: Traditional white inner trousers and black cloth shoes with white sole lines.
- **Atmospheric Enhancements & Background Overhaul**:
  - **16-Bit Dithered Twilight Sky**: 4x4 Bayer matrix ordered dithering transitioning between 7 twilight purple gradient bands with shimmering sparkle stars.
  - **Crimson Blood Moon**: Multi-ring cosmic aura, ancient gnarled branches, fractured rim bite, and central sacred Om (ॐ) glyph.
  - **Luminous Vân Mây Ruyi**: Traditional Vietnamese scroll clouds with cascading stepped lobes, trailing serpentine ribbon tails, and deep plum border.
  - **Redesigned Authentic Tombstones**:
    - **Mộ Công Giáo (`style="cross"`)**: Polished black granite cross (>60% of height) with 3D specular bevels, standing on an upright inscribed stele with gold frame/motifs and stepped granite base with a central green grass turf.
    - **Mộ Người Bình Thường (`style="buddhist"`)**: Carved lotus bud finial, 2-tier curved roofs (*mái đao cong vảy rồng*), dragon pillars, radiant golden Sauwastika (*CHỮ VẠN 卍*), and incense burner with glowing red joss sticks.
    - **Floating Stone Platform (`style="platform"`)**: Weathered masonry stone blocks with 3D bevels, brick seams, and moss accents.
  - **Creeping Ground Mist**: Low-altitude drifting fog across the cemetery lawn (`_draw_ground_fog`).
  - **Ghost Speed Fix**: Corrected attribute reference to `ghost.speed` in `_draw_ghost` to eliminate Level 4/5 runtime crashes.
  - **Porter-Duff Zero-Alpha Blackout Fix**: Fixed black bar artifacts on macOS display surfaces by initializing internal canvas as 24-bit RGB (`depth=24`) and switching `show_overlay()` to standard surface-level alpha (`set_alpha(160)`).
  - **Compact Lantern Ambient Glow**: Scaled lantern light radius down to a focused 48x48 ambient aura around Hoá.
  - **Icy Cyan Timeout Banner**: Shifted timeout game-over text to glacial frost cyan `(165, 230, 255)`.
