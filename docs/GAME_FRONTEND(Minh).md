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

### Phase 3 — Verification & Team Hand-off (Remaining)
- [ ] **Step 7: Headless Verification & PR to `main`**
  - [ ] Add automated headless test (`tests/test_env.py`) verifying fast stepping across all 5 levels, clean resets (player, ghosts, items), and termination/event flags.
  - [ ] Freeze final contract and merge `game_frontend` into `main` for Members 2, 3, and 4.

---

## 4. Work Log

- **Step 1**: Defined `Action`, `RawState`, `GameEvents`, and `BaseEnvironment` in `core/base_env.py`. Verified package initialization with `core/__init__.py`. Removed redundant `velocity_x` from `RawState`.
- **Step 2**: Implemented physical entities in `env/entities.py` and `env/__init__.py` package exports.
- **Step 3**: Built 2D platformer engine with axis-separated AABB collisions, pit-edge anti-teleport guard (`old_bottom <= floor_y`), local-vision `_get_raw_state()`, and lazy-initialized Pygame rendering with camera follow.
- **Step 4**: Built modular level registry in `env/map_levels.py` using `DEFAULT_FLOOR_Y = 320.0`.
- **Step 5**: Created `play_human.py` manual keyboard runner and fixed the macOS window freeze on `R` reset by pumping Pygame events via `env.render()`. Committed v1 on `game_frontend`.
- **Step 6**: Completed full v2 upgrade (**Hoá's Midnight Escape**): refactored `MarioEnvironment` $\to$ `GameEnvironment` (`env/game_env.py`) and `Mario` $\to$ `Player`; added hovering `Ghost` and `WarmthItem`; updated `RawState`/`GameEvents`; built midnight graveyard Pygame visuals; expanded `env/map_levels.py` to 5 progressive levels (uniform ghosts in Level 4, diverse ghosts in Level 5, items from Level 3 onwards); updated `play_human.py` for Levels `1–5` and play-tested successfully.

---

## 5. Module Definitions & Structure *(Current v2 Codebase)*

### 5.1 `core/base_env.py` — Environment Contract & Abstractions
Defines the shared interface and data structures decoupling the physics engine from the RL agent, discretizer, and reward modules (Dependency Inversion Principle).

- **`Action`**: Discrete action constants (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`, `NUM_ACTIONS=5`).
- **`RawState` *(dataclass)***: Continuous physical snapshot passed to Member 2's discretizer:
  - Player kinematics: `player_x`, `player_y`, `velocity_y`, `is_grounded`.
  - Local vision ahead: `dist_to_obstacle`, `obstacle_height`, `dist_to_pit`, `dist_to_ghost`, `ghost_y`, `dist_to_item`, `dist_to_goal`.
- **`GameEvents` *(dataclass)***: Per-step outcome flags passed to Member 3's reward strategy (`dx`, `reached_goal`, `fell_in_pit`, `hit_obstacle`, `hit_ghost`, `collected_item`).
- **`BaseEnvironment(ABC)`**: Abstract contract enforcing `reset()`, `step(action)`, `render()`, and `close()`.

### 5.2 `env/entities.py` — Physical Game Objects
Holds lightweight data models and hitbox geometry with zero rendering or RL coupling (Single Responsibility Principle).

- **`Player`**: Hoá's entity (`x`, `y`, hitbox `24x32`, `vy`, `is_grounded`, `is_alive`, `facing_right`) with `reset(start_x, start_y)` and bounding-box properties (`rect`, `right`, `bottom`).
- **`Obstacle`**: Tall tombstone (`x`, `y`, default `32x48`) with `rect`, `right`, `bottom`.
- **`Pit`**: Bottomless open grave (`x_start`, `x_end`) with `contains(x, width)` center-of-mass check.
- **`Ghost`**: Hovering ghost (`x`, `y`, `patrol_min_x`, `patrol_max_x`, `speed=1.5`, `width=24`, `height=28`, `float_amplitude=8.0`, `float_speed=0.08`) with wrapped sinusoidal wave (`% 2*pi`), `update()`, `reset()`, and `rect`/`right`/`bottom`.
- **`WarmthItem`**: Collectible warmth item (`x`, `y`, `14x14`, `is_collected=False`) with `reset()` and `rect`/`right`/`bottom`.
- **`GoalFlag`**: Hoá's warm home (`x`, `y`, `64x88`) with `is_reached(player)` win detection.

### 5.3 `env/game_env.py` — 2D Physics Engine & Graveyard Renderer
Concrete `GameEnvironment(BaseEnvironment)` driving physics, collisions, dynamic hazards, state extraction, and midnight graveyard Pygame rendering.

- **Physics Constants**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0` (max jump height $\approx 105\text{ px}$, max jump span $\approx 130\text{ px}$).
- **Lifecycle Methods**:
  - `reset() -> RawState`: Restores Hoá, all ghosts, all warmth items, and `step_count = 0`.
  - `step(action) -> (RawState, GameEvents, bool)`: Runs axis-separated X/Y AABB collisions, pit-fall guard (`old_bottom <= floor_y`), updates ghosts, checks warmth pickups, checks termination (`hit_ghost`, `fell_in_pit`, `reached_goal`, `max_steps`), and returns state + events.
  - `render()` / `close()`: Lazy-initializes Pygame (`"Hoá's Midnight Escape"`) and delegates rendering to `_draw_background`, `_draw_tombstone`, `_draw_ghost`, `_draw_item`, `_draw_home`, and `_draw_player` with horizontal camera scrolling (`camera_x`).
- **Helpers**: `_get_raw_state()` (nearest upcoming tombstone, pit, ghost + Y, uncollected item, and home) and `_rects_overlap()`.

### 5.4 `env/map_levels.py` — 5-Level Curriculum Registry
Separates stage definitions from `GameEnvironment` (Open/Closed Principle) with `DEFAULT_FLOOR_Y = 320.0`.

- **`LevelConfig` *(dataclass)***: Bundles `name`, `obstacles`, `pits`, `goal`, `ghosts`, `items`, `floor_y`, `max_steps`.
- **`LEVEL_BUILDERS`** (`1` to `5`):
  - `1 -> _build_level_1` (*Quiet Graveyard*): 2 tombstones, no pits/ghosts/items (`max_steps=800`).
  - `2 -> _build_level_2` (*Open Graves Pits*): 3 tombstones + 2 pits (`max_steps=1000`).
  - `3 -> _build_level_3` (*Cold Trail*): Shifted tombstones/pits + 3 warmth items (`max_steps=1000`).
  - `4 -> _build_level_4` (*Whispering Graves*): 3 tombstones + 2 pits + 3 items + 2 uniform ghosts (`max_steps=1000`).
  - `5 -> _build_level_5` (*Midnight Escape*): 4 tombstones + 2 pits + 4 items + 4 diverse ghosts (`max_steps=1000`).
- **Public Factories**: `get_level(level_id)` and `create_env(level_id)`.

### 5.5 Package Entry Exports (`core/__init__.py`, `env/__init__.py`)
- **`core/__init__.py`**: Exports `Action`, `RawState`, `GameEvents`, `BaseEnvironment`.
- **`env/__init__.py`**: Exports `Player`, `Obstacle`, `Pit`, `Ghost`, `WarmthItem`, `GoalFlag`, `GameEnvironment`, `LevelConfig`, `LEVEL_BUILDERS`, `get_level`, `create_env`.

### 5.6 `play_human.py` — Interactive Playable Entry Point
Standalone keyboard runner (`A`/`Left`, `D`/`Right`, `Space`/`W`/`Up`) stepping `GameEnvironment` at 60 FPS with live switching across Levels `1–5`, freeze-safe manual resets (`R`), event logs, and exit (`Esc`).
