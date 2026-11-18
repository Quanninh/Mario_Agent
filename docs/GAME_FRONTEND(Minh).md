# Member 1: Environment & Physics Progress Log

> **Owner**: Member 1  
> **Role**: Environment & Game Physics Lead  
> **Game Concept**: **Hoá's Midnight Escape** *(2D Graveyard Platformer)*  
> **Scope**: 2D Platformer Engine (Pygame), Physics, Collisions, Levels, and Human Test Mode  

---

## 1. Role Objective & Creative Theme

Build a lightweight, headless-compatible 2D platformer environment implementing the `BaseEnvironment` interface. The environment simulates physical movement, gravity, collisions, and proximity-based state observations for the Tabular Q-Learning agent.

### Story & Theme — *Hoá's Midnight Escape* (v2 Direction)
Hoá lives near a graveyard. Somehow, he wakes up in the middle of the graveyard late at night—freezing cold and terrified—and desperately tries to find his way back home:

| Story Element | Role in RL Engine | Behavior / Mechanics |
|:---|:---|:---|
| **Hoá** | Player / Agent | Walks left/right and jumps (`STAY`, `LEFT`, `JUMP`, `RIGHT`, `RIGHT_JUMP`). |
| **Cold Midnight Graveyard** | World Setting | Dark night sky, moon, stars, and shadowy graveyard ground. |
| **Tall Graves / Tombstones** | Static Obstacles (`Obstacle`) | Block the path; Hoá must jump over or land on top of them. |
| **Open Grave Pits** | Floor Hazards (`Pit`) | Bottomless gaps; falling in ends the episode (`done = True`). |
| **Floating Ghosts** | Moving Enemies (`Ghost` — *v2*) | Hover and patrol in mid-air (sinusoidal float); touching one scares Hoá into passing out (`done = True`), so he must jump over low ghosts or pass beneath high ghosts. |
| **Warmth Item** *(e.g. Candle / Matchstick / Lantern)* | Collectible Item (*v2*) | Optional small positive reward item along the path to satisfy Project 5's *"collecting useful items"* requirement. |
| **Hoá's House** | Destination (`Goal`) | Warm, glowing house with lit windows at the end of the stage (`reached_goal = True`, `done = True`). |

---

## 2. Key Design Decisions

- **Action Space**: 5 discrete actions (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`).
- **State Representation**: Uses **local vision / relative proximity** (tracking nearest upcoming tombstone, pit, ghost, and home) rather than full-level coordinates, keeping the state space compact for Q-Learning.
- **Velocity Simplification**: Excluded horizontal sliding velocity (`velocity_x`) from `RawState` and player entity since movement strictly mirrors current input. Vertical motion uses `velocity_y` + `is_grounded` for gravity and jump control.
- **SOLID Compliance**: High-level modules depend on `core.BaseEnvironment`. Level layouts use a Registry Pattern (`LEVEL_BUILDERS`) with `DEFAULT_FLOOR_Y = 320.0`, allowing v2 additions (floating ghosts, warmth items, new maps) without rewriting core physics.

---

## 3. Implementation Checklist

### Phase 1 — Core Platformer Engine (v1 — Complete)
- [x] **Step 1: Core Abstraction & Contracts**
  - [x] `core/base_env.py`: Defined `Action`, `RawState` (without `velocity_x`), `GameEvents`, and `BaseEnvironment`.
  - [x] `core/__init__.py`: Package export for clean imports.
- [x] **Step 2: Game Entities** (`env/entities.py` & `env/__init__.py`)
  - [x] `Mario` *(player)*: Position (`x`, `y`), hitbox (`24x32`), vertical velocity (`vy`), `is_grounded`, `is_alive`, and `reset()`.
  - [x] `Obstacle` *(tombstone)*: Solid bounding box (`32x48`).
  - [x] `Pit` *(grave pit)*: Floor gap hazard with `contains()` center-of-mass check.
  - [x] `GoalFlag` *(home)*: Finish line (`16x96`) with `is_reached()` detection.
  - [x] `env/__init__.py`: Exported all entity classes.
- [x] **Step 3: Environment Engine** (`env/mario_env.py`)
  - [x] Gravity (`0.8`), jump impulse (`-13.0`), walk speed (`4.0`), and axis-separated X/Y collisions.
  - [x] Obstacle side/top/bottom collisions and pit-fall death checks (with `old_bottom <= floor_y` anti-teleport fix).
  - [x] `reset()` & `step(action)`: Physics tick returning `(RawState, GameEvents, done)` with local-vision proximity features.
  - [x] `render()` & `close()`: Lazy-initialized Pygame visual rendering with horizontal camera scrolling (headless-friendly).
- [x] **Step 4: Level Layouts** (`env/map_levels.py`)
  - [x] `LevelConfig` dataclass & `LEVEL_BUILDERS` registry using `DEFAULT_FLOOR_Y = 320.0`.
  - [x] Level 1 (Baseline): Flat floor with 2 obstacles, no pits (`max_steps=800`).
  - [x] Level 2 (Standard): 3 obstacles + 2 pits with camera scrolling (`max_steps=1000`).
  - [x] Level 3 (Generalization): Shifted obstacle/pit order & spacing (`max_steps=1000`).
  - [x] `get_level(level_id)` & `create_env(level_id)` helpers exported in `env/__init__.py`.
- [x] **Step 5: Playable Human Mode** (`play_human.py`)
  - [x] Manual keyboard controls (`A`/`Left`, `D`/`Right`, `Space`/`W`/`Up`, `R` reset, `Esc` quit) and live level switching (`1`/`2`/`3`).
  - [x] Fixed `R` reset window-freeze bug by calling `env.render()` to pump Pygame's OS event queue before `continue`.

### Phase 2 — Graveyard Theme & Dynamic Hazards (v2 — Planned)
- [ ] **Step 6: Hoá's Midnight Escape Upgrade**
  - [ ] Update `core/base_env.py`: Add ghost proximity (`dist_to_enemy`, `enemy_y`) and item proximity to `RawState`, plus ghost-hit & item-pickup flags to `GameEvents`.
  - [ ] Update `env/entities.py`: Add hovering `Ghost` (horizontal patrol + vertical sinusoidal floating) and collectible warmth item (exact item TBD); align naming/theme for Hoá, Tombstones, Ghosts, and Home.
  - [ ] Upgrade engine & `render()`: Add ghost movement/collision, item pickup, and midnight graveyard visuals (night sky, moon, stars, gravestones, glowing ghosts, Hoá character, and warm lit house).
  - [ ] Update `env/map_levels.py`: Place hovering ghosts and warmth items across Levels 1–3.
- [ ] **Step 7: Headless Verification & Team Hand-off**
  - [ ] Add automated headless test verifying fast stepping, resets (player, ghosts, items), and termination flags.
  - [ ] Freeze final contract and merge `game_frontend` into `main` for Members 2, 3, and 4.

---

## 4. Work Log

- **Step 1**: Defined `Action`, `RawState`, `GameEvents`, and `BaseEnvironment` in `core/base_env.py`. Verified package initialization with `core/__init__.py`. Removed redundant `velocity_x` from `RawState`.
- **Step 2**: Implemented physical entities in `env/entities.py` (`Mario`, `Obstacle`, `Pit`, `GoalFlag`) and created `env/__init__.py` package exports. Added hitbox bounding boxes and collision helper properties (`rect`, `right`, `bottom`, `is_reached`).
- **Step 3**: Built `MarioEnvironment` in `env/mario_env.py` with axis-separated AABB collisions, pit-edge anti-teleport guard (`old_bottom <= floor_y`), local-vision `_get_raw_state()`, and lazy-initialized Pygame rendering with camera follow. Exported in `env/__init__.py`.
- **Step 4**: Built modular level configurations in `env/map_levels.py` (`LevelConfig`, `LEVEL_BUILDERS`, `get_level`, `create_env`) for Levels 1–3 using `DEFAULT_FLOOR_Y = 320.0`, exported via `env/__init__.py`.
- **Step 5**: Created `play_human.py` manual keyboard runner (`A`/`D`/`Space`, `1`/`2`/`3` level switching, `R` reset, `Esc` quit). Fixed a window freeze when pressing `R` by calling `env.render()` before `continue` so `pygame.event.get()` keeps pumping events. Committed v1 on branch `game_frontend`.
- **v2 Design**: Defined the creative theme **"Hoá's Midnight Escape"** (graveyard at night, tall tombstones, floating ghosts, warmth collectibles, glowing home) as the roadmap for Step 6 before coding.

---

## 5. Module Definitions & Structure *(Current v1 Codebase)*

### 5.1 `core/base_env.py` — Environment Contract & Abstractions
Defines the shared interface and data structures decoupling the physics engine from the RL agent, discretizer, and reward modules (Dependency Inversion Principle).

- **`Action`**: Discrete action constants (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`, `NUM_ACTIONS=5`).
- **`RawState` *(dataclass)***: Continuous physical snapshot passed to Member 2's discretizer:
  - Player kinematics: `mario_x`, `mario_y`, `velocity_y`, `is_grounded`.
  - Local vision ahead: `dist_to_obstacle`, `obstacle_height`, `dist_to_pit`, `dist_to_goal`.
- **`GameEvents` *(dataclass)***: Per-step outcome flags passed to Member 3's reward strategy (`dx`, `reached_goal`, `fell_in_pit`, `hit_obstacle`).
- **`BaseEnvironment(ABC)`**: Abstract contract enforcing `reset()`, `step(action)`, `render()`, and `close()`.

### 5.2 `env/entities.py` — Physical Game Objects
Holds lightweight data models and hitbox geometry with zero rendering or RL coupling (Single Responsibility Principle).

- **`Mario`**: Player entity (`x`, `y`, hitbox `24x32`, `vy`, `is_grounded`, `is_alive`) with `reset(start_x, start_y)` and bounding-box properties (`rect`, `right`, `bottom`).
- **`Obstacle`**: Solid obstacle/tombstone (`x`, `y`, default `32x48`) with `rect`, `right`, `bottom`.
- **`Pit`**: Bottomless floor gap (`x_start`, `x_end`) with `contains(x, width)` using center-of-mass detection.
- **`GoalFlag`**: Level destination (`x`, `y`, default `16x96`) with `is_reached(mario)` win detection.

### 5.3 `env/mario_env.py` — 2D Physics Engine & Renderer
Concrete implementation of `BaseEnvironment` driving simulation physics, collisions, state extraction, and optional Pygame rendering.

- **Physics Constants**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0` (max jump height $\approx 105\text{ px}$, max jump span $\approx 130\text{ px}$).
- **Lifecycle Methods**:
  - `reset() -> RawState`: Restores player to `(start_x, start_y)` and resets `step_count = 0`.
  - `step(action) -> (RawState, GameEvents, bool)`: Runs axis-separated X then Y AABB collision resolution, pit-fall detection (guarded by `old_bottom <= floor_y`), goal/timeout checks, and returns the updated state and events.
  - `render()` / `close()`: Lazy-initializes Pygame only when called (enabling high-speed headless training) and draws the level with horizontal camera scrolling (`camera_x`).
- **Internal Helpers**: `_get_raw_state()` (nearest upcoming obstacle/pit + goal distance) and `_rects_overlap()` (AABB intersection test).

### 5.4 `env/map_levels.py` — Modular Level Registry
Separates stage definitions from the game engine so new maps can be added without modifying the engine (Open/Closed Principle).

- **`DEFAULT_FLOOR_Y = 320.0`**: Shared ground elevation across all stages.
- **`LevelConfig` *(dataclass)***: Bundles `name`, `obstacles`, `pits`, `goal`, `floor_y`, and `max_steps`.
- **`LEVEL_BUILDERS`**: Dictionary mapping level IDs to builder functions:
  - `1 -> _build_level_1`: Baseline (2 obstacles, no pits, goal at `x=720`).
  - `2 -> _build_level_2`: Standard (3 obstacles, 2 pits, goal at `x=1100`).
  - `3 -> _build_level_3`: Generalization test (pit first + shifted obstacles, goal at `x=1120`).
- **Public Factories**: `get_level(level_id)` returns a fresh `LevelConfig`; `create_env(level_id)` constructs a ready-to-run `MarioEnvironment`.

### 5.5 Package Entry Exports (`core/__init__.py`, `env/__init__.py`)
- **`core/__init__.py`**: Exports `Action`, `RawState`, `GameEvents`, `BaseEnvironment`.
- **`env/__init__.py`**: Exports `Mario`, `Obstacle`, `Pit`, `GoalFlag`, `MarioEnvironment`, `LevelConfig`, `LEVEL_BUILDERS`, `get_level`, `create_env`.

### 5.6 `play_human.py` — Interactive Playable Entry Point
Standalone runner mapping keyboard inputs (`A`/`Left`, `D`/`Right`, `Space`/`W`/`Up`) into discrete `Action` IDs, stepping the environment at 60 FPS, and supporting live level switching (`1`/`2`/`3`), freeze-safe manual resets (`R`), and exit (`Esc`).
