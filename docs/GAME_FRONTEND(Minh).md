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
| **Hoá** | `Player` (Agent) | Walks left/right, jumps, double jumps in mid-air, and crouches/momentum-slides (`STAY`, `LEFT`, `JUMP`, `RIGHT`, `RIGHT_JUMP`, `CROUCH`). |
| **Cold Midnight Graveyard** | `GameEnvironment` | Dithered 16-bit twilight night sky, blood moon with ancient branches & Om rune, scrolling Ruyi clouds, and shadowy graveyard soil. |
| **Tall Graves / Tombstones** | `Obstacle` | Block the path; Hoá must jump over or land on top of them (`"cross"`, `"buddhist"`, `"platform"`). |
| **Open Grave Pits & Hellfire** | `Pit` | Bottomless gaps; falling in or touching erupting flame jets ends the episode (`fell_in_pit=True`, `hit_fire=True`, `done=True`). |
| **Base Floating Ghosts** | `Ghost` | Hover and patrol in mid-air with customizable speed & sinusoidal wave; touching one ends the episode (`hit_ghost=True`, `done=True`). |
| **Rotating Laser Ghosts** | `LaserGhost` | Wears Nón Lá straw hat, talisman, and Bagua mirror; fires a lethal 360° continuously rotating scarlet laser beam (`hit_laser=True`, `done=True`). |
| **Wisp Maidens (*Bạch Y Ma Nữ*)** | `WispGhost` | White mourning robes with flowing cyan hair; charges and launches parabolic Will-o'-the-Wisp projectiles across sky and ground. |
| **Will-o'-the-Wisp Fireballs** | `WillOWisp` | Parabolic bouncing spirit projectiles with ground friction and rolling embers (`hit_wisp=True`, `done=True`). |
| **Quỷ Nhập Tràng (Corpse Demon)** | `QuyNhapTrang` (`GraveMonster`) | Ambushes from tombstones when Hoá approaches ($<180\text{px}$); sprints aggressively with billowing black hair, white shroud, red eyes, and vaults over open pits (`hit_monster=True`, `done=True`). |
| **Cương Thi (Jiangshi / Hopping Vampire)** | `CuongThi` (`GraveMonster`) | Dormant grave with yellow talisman & hovering cyan spirit flames; bursts out in Qing navy robe, prominent red official hat, and pale cyan limbs; performs rhythmic leftward hops with dynamic draped/fluttering forehead talisman and autonomous pit gap vaulting (`hit_monster=True`, `done=True`). |
| **Warding Garlic Bulb (*Củ Tỏi Trừ Tà*)** | `WarmthItem` | Sacred garlic bulb collectible along the path (`collected_item=True`), reflecting the Vietnamese tradition of carrying garlic into cemeteries to ward off wandering spirits and malevolent ghosts. Features expanding lilac smoke and golden star burst upon pickup. |
| **Khuê Văn Các Pavilion** | `GoalFlag` | Hanoi's iconic Temple of Literature pavilion with glowing sunburst window (*Cửa Sổ Sao Khuê*) and amber sanctuary light (`reached_goal=True`, `done=True`). |

---

## 2. Key Design Decisions

- **Action Space**: 6 discrete actions (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`, `CROUCH=5`, `NUM_ACTIONS=6`).
- **State Representation**: Uses **local vision / relative proximity** (tracking nearest upcoming tombstone, pit + fire phase, ghost + Y, nearest laser segment, active wisps, garlic item, and goal) rather than full-level coordinates, keeping the state space compact for Q-Learning.
- **Clean Object-Oriented Ghost Hierarchy**: Extends `Ghost` into specialized subclasses (`LaserGhost` and `WispGhost`) rather than adding boolean flags to the base class, preserving strict polymorphism and Liskov Substitution Principle.
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

#### Tier 3 — New Gameplay & Dynamic Hazard Mechanics *(Completed & Active)*
- [x] **9. Mid-air Double Jump & Ledge Recovery (`can_double_jump`)**: Allow Hoá to execute a second jump while airborne (`not is_grounded`), allowing recovery from miscalculated pit leaps and dodging high hovering ghosts. Also grants 1 recovery jump if walking off a ledge into the air. Reuses existing `Action.JUMP` and `Action.RIGHT_JUMP` without expanding the action space, while exposing a binary `can_double_jump: bool` flag in `RawState` for the RL agent.
- [x] **10. Player Crouch & Momentum Slide (`Action.CROUCH`)**: Lowers Hoá's hitbox height from $32\text{px} \to 18\text{px}$ (`S` / `Down` key) to duck under hovering ghosts. Implements physical ducking in place when stationary and momentum-based sliding when running ($v_{x, t} = v_{x, t-1} \cdot 0.82$), keeping feet anchored to the ground *(expands `Action.NUM_ACTIONS` from 5 to 6)*.
- [x] **11. Fire Erupting from Grave Pits**: Periodic flame jets shooting upward out of selected pits on a cycle (`idle -> warning glow -> active lethal flame -> cooldown`), with pit-fire phase exposed in `RawState` so the AI can time its jumps. Hitbox spans the full pit width and depth from ground flame height down to canvas bottom: `(pit.x_start, floor_y - FIRE_HEIGHT, pit.x_end - pit.x_start, canvas_height - (floor_y - FIRE_HEIGHT))`.
- [x] **12. Ghosts Shooting Rotating Eye-Laser Beams (`LaserGhost`)**: Implemented via dedicated `LaserGhost(Ghost)` subclass. Fires a lethal scarlet laser beam from the ghost's eyes that **rotates continuously 360° (`0..2π`) while the ghost patrols** (`laser_length=90.0`, `laser_rotation_speed=0.04`). Emits `events.hit_laser` upon collision.
- [x] **13. Wisp Maidens & Will-o'-the-Wisp Projectiles (`WispGhost`, `WillOWisp`)**: Specialized ghost (*Bạch Y Ma Nữ*) hovering across sky and ground that charges and casts parabolic will-o'-the-wisp fireballs that bounce on the floor with ground friction and rolling embers. Emits `events.hit_wisp`.
- [x] **14. Authentic Vietnamese Warding Garlic Bulb (*Củ Tỏi Trừ Tà*) & Collection Burst**: Replaced generic warmth orb with authentic sacred garlic bulb (arched sickle hook stalk, soft lilac/violet shading, pure white specular gleam, compact root nub, form-fitting dotted golden aura perimeter, floating embers). When collected, triggers dynamic burst with expanding lilac/white smoke puffs and radiating 4-pointed golden sparkle stars.
- [x] **15. Redesigned Level 5 — Valley of the Wisp Maidens**: Intermediate-hard vertical gauntlet featuring 18 ghosts: 8 high-sky and mid-air Wisp Maidens between $y = 70$ and $y = 155$, 5 ground-level LaserGhosts sweeping 360° rotating beams, 2 ground Wisp Maidens, 3 patrolling base ghosts, 7 tombstones, 5 pits (including 3 hellfire eruptions), and 6 garlic bulbs (`max_steps=2400`).
- [x] **16. Clean Polymorphic Ghost Architecture**: Eliminated all dummy boolean flags (`has_laser`) in favor of strict object-oriented inheritance (`LaserGhost`, `WispGhost`). Decoupled test suite to use `isinstance()` checks.
- [ ] **17. Rising & Charging Skeleton Ambush from Tombs**: Special mausoleum tombs that trigger when Hoá approaches (proximity $< 180\text{px}$): a skeletal warrior bursts upward from the grave, leaps forward, and sprints aggressively toward the player (`dx = player.x - skeleton.x`), demanding quick reaction timing or a high double jump to evade. Requires a new `Skeleton` entity with ground physics, pursuit logic, and nearest skeleton proximity in `RawState`.

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
- **Step 15**: Implemented Continuous 360° Rotating Laser Ghosts (`LaserGhost`):
  - Defined `LaserGhost(Ghost)` in `env/entities.py` with continuous circular beam kinematics (`laser_length=90.0`, `laser_rotation_speed=0.04`, `laser_initial_angle`).
  - Implemented pure-Python line segment vs. player AABB intersection algorithm (`_segment_intersects_aabb`) in `env/game_env.py` for headless speed.
  - Added `hit_laser: bool = False` to `GameEvents` in `core/base_env.py`.
  - Implemented procedural pixel art in `env/game_renderer.py`: conical straw hat (*Nón Lá*), red Daoist warding talisman (*Bùa Lệnh Trừ Ma*), Bagua chest mirror (*Kính Bát Quái*), and rotating scarlet beam with ethereal glow aura.
  - Configured rotating laser ghosts in Level 4 (dual counter-rotating scissor lasers) and Level 5.
- **Step 16**: Implemented Wisp Maidens & Parabolic Will-o'-the-Wisp Projectiles (`WispGhost`, `WillOWisp`):
  - Created `WispGhost(Ghost)` in `env/entities.py` with telegraph charging cycle and alternating high/low parabolic launch trajectories.
  - Created `WillOWisp` entity with flight kinematics, gravity, ground bounce damping, floor rolling friction, and lifespan timeout.
  - Added `hit_wisp: bool = False` to `GameEvents` in `core/base_env.py`.
  - Rendered Wisp Maiden in `env/game_renderer.py`: flowing emerald/cyan hair, white mourning robes (*Bạch Y Ma Nữ*), charging cyan spirit-flame, and airborne will-o'-the-wisp with trailing and rolling embers.
- **Step 17**: Redesigned Level 5 into Vertical Sky & Ground Gauntlet (*Valley of the Wisp Maidens*):
  - Completely redesigned `_build_level_5()` in `env/map_levels.py` with 18 ghosts: 8 high-sky and mid-air Wisp Maidens flying between $y = 70$ and $y = 155$, 5 ground-level LaserGhosts sweeping 360° beams, 2 ground Wisp Maidens, 3 patrolling base ghosts, 7 tombstones, 5 pits (including 3 hellfire eruptions), and 6 collectible items (`max_steps=2400`).
- **Step 18**: Authentic Vietnamese Warding Garlic Bulb (*Củ Tỏi Trừ Tà*) & Collection Burst Animation:
  - Replaced generic warmth orb with authentic sacred garlic bulb: squat 3D teardrop lobes, soft lilac/violet crevice shading, pure white specular shine, arched sickle hook stalk, compact woody root nub, form-fitting dotted golden aura perimeter, and floating twinkling embers.
  - Added dynamic Collection Burst animation in `_draw_item_bursts()`: expanding soft lilac/white smoke cloud puffs, radiant 4-pointed golden sparkle stars with white-hot cores, and warm inner radial glow.
  - Updated HUD and victory overlay counter strings to `"Garlic: X/Y"`.
- **Step 19**: Clean OOP Architecture & Removing Redundant `has_laser` Attribute:
  - Removed redundant `has_laser` boolean attribute from `env/entities.py`.
  - Updated `tests/test_env.py` to use `isinstance(g, LaserGhost)`.
  - Exported `LaserGhost` and `WispGhost` in `env/__init__.py`.
  - All 17 automated headless unit tests passing cleanly in `0.32s`.

---

## 5. Module Definitions & Structure *(Current v2 Codebase)*

### 5.1 `core/base_env.py` — Environment Contract & Abstractions
Defines the shared interface and data structures decoupling the physics engine from the RL agent, discretizer, and reward modules (Dependency Inversion Principle).

- **`Action`**: Discrete action constants (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`, `CROUCH=5`, `NUM_ACTIONS=6`).
- **`RawState` *(dataclass)***: Continuous physical snapshot passed to Member 2's discretizer:
  - Player kinematics: `player_x`, `player_y`, `velocity_x`, `velocity_y`, `is_grounded`, `can_double_jump`, `is_crouching`.
  - Local vision ahead: `dist_to_obstacle`, `obstacle_height`, `dist_to_pit`, `pit_fire_state`, `dist_to_ghost`, `ghost_y`, `dist_to_item`, `dist_to_goal`.
- **`GameEvents` *(dataclass)***: Per-step outcome flags passed to Member 3's reward strategy (`dx`, `reached_goal`, `fell_in_pit`, `hit_obstacle`, `hit_ghost`, `hit_fire`, `hit_laser`, `hit_wisp`, `collected_item`).
- **`BaseEnvironment(ABC)`**: Abstract contract enforcing `reset()`, `step(action)`, `render()`, and `close()`.

### 5.2 `env/entities.py` — Physical Game Objects
Holds lightweight data models and hitbox geometry with zero rendering or RL coupling (Single Responsibility Principle).

- **`Player`**: Hoá's entity (`x`, `y`, hitbox `24x32`, `vx`, `vy`, `is_grounded`, `can_double_jump`, `is_crouching`, `is_alive`, `facing_right`) with `reset(start_x, start_y)` and bounding-box properties (`rect`, `right`, `bottom`). Hitbox shrinks to `24x18` when crouching.
- **`Obstacle`**: Tall tombstone (`x`, `y`, default `32x48`, `style="cross"|"buddhist"|"platform"`) with `rect`, `right`, `bottom`.
- **`Pit`**: Open grave pit (`x_start`, `x_end`, `has_fire`, `fire_offset`, `FIRE_CYCLE`, `FIRE_HEIGHT = 65.0`) with `contains()` and `get_flame_rect()`.
- **`Ghost`**: Base hovering ghost (`x`, `y`, `patrol_min_x`, `patrol_max_x`, `speed=1.5`, `width=24`, `height=28`, `float_amplitude=8.0`, `float_speed=0.08`) with wrapped sinusoidal wave (`% 2*pi`), `update()`, `reset()`, and `rect`/`right`/`bottom`.
- **`LaserGhost(Ghost)`**: Rotating laser ghost (`laser_length=90.0`, `laser_rotation_speed=0.04`, `laser_initial_angle=0.0`) with `get_laser_segment(step_count)`.
- **`WispGhost(Ghost)`**: Wisp Maiden ghost (`wisp_cycle=140`, `wisp_offset=0`, `wisp_mode="alternating"`) with `is_charging_wisp(step_count)`.
- **`WillOWisp`**: Spirit fireball projectile (`x`, `y`, `vx`, `vy`, `radius=4.0`, `gravity=0.45`, `bounce_damping=0.65`, `floor_friction=0.92`, `lifespan=160`) with `update()`, `is_dead`, and `rect`.
- **`WarmthItem`**: Collectible Warding Garlic Bulb (*Củ Tỏi Trừ Tà*) (`x`, `y`, `16x16`, `is_collected=False`) with `reset()` and `rect`/`right`/`bottom`.
- **`GoalFlag`**: Hanoi's iconic Khuê Văn Các pavilion (`x`, `y`, `64x88`) with `is_reached(player)` win detection.

### 5.3 `env/game_env.py` — 2D Physics Engine & Simulation Contract
Concrete `GameEnvironment(BaseEnvironment)` driving physics kinematics, axis-separated AABB collisions, dynamic hazards, state extraction, and headless execution. Free of all Pygame surface drawing code.

- **Physics Constants & World Bounds**: `GRAVITY = 0.8`, `WALK_SPEED = 4.0`, `JUMP_FORCE = -13.0`, `DOUBLE_JUMP_RATIO = 0.88`, `SLIDE_FRICTION = 0.82`, `STANDING_HEIGHT = 32.0`, `CROUCH_HEIGHT = 18.0`. Physical logic runs on `canvas_width=800, canvas_height=400` with `floor_y=320.0`.
- **Lifecycle Methods**:
  - `reset() -> RawState`: Restores Hoá, all ghosts, all warmth items, clears active wisps, and resets `step_count = 0`.
  - `step(action) -> (RawState, GameEvents, bool)`: Runs axis-separated X/Y AABB collisions, pit-fall guard (`old_bottom <= floor_y`), updates ghosts, advances rotating laser segments, updates & spawns will-o'-the-wisp projectiles, checks garlic pickups, checks clean termination (`hit_ghost`, `hit_fire`, `hit_laser`, `hit_wisp`, `fell_in_pit`, `reached_goal`, `max_steps`), and returns state + events.
  - `render()` / `close()`: Lazily instantiates `GameRenderer` on demand and delegates drawing. Stepping headlessly leaves renderer as `None` for zero overhead.
  - `show_game_over(events)`: Delegates end-screen banner display to the active renderer.
- **Helpers**: `_get_raw_state()` (nearest upcoming tombstone, pit + fire phase, ghost + Y, garlic item, and goal), `_segment_intersects_aabb()`, and `_rects_overlap()`.

### 5.4 `env/game_renderer.py` — Pygame Graphical View & Presentation
Dedicated `GameRenderer` class encapsulating all visual elements, display scaling, and user feedback:

- **Display Management**: Maps internal `800x400` logic canvas to `1400x700` window display with 60 FPS clock synchronization.
- **Midnight Graveyard Aesthetics**: Moon, decorative stars, layered soil/grass, and complete color palette.
- **Layered Entity Drawing**:
  - `_draw_background()`: Sky, blood moon with Om rune, stars, soil, grass, grave pits, and multi-layered flame spires.
  - `_draw_tombstone()`: Latin crosses (`"cross"`), arched Buddhist sauwastikas (`"buddhist"` / `卍`), and floating platforms (`"platform"`).
  - `_draw_ghost()`: Dispatches to `_draw_laser_ghost()`, `_draw_wisp_maiden()`, or `_draw_base_ghost()`.
  - `_draw_wisp()`: Parabolic will-o'-the-wisp with trailing sparks and ground rolling embers.
  - `_draw_item()`: Authentic Vietnamese Warding Garlic Bulb (*Củ Tỏi Trừ Tà*) with arched sickle hook stalk, soft lilac/violet shading, pure white central specular gleam, compact root nub, form-fitting dotted golden aura perimeter, and floating twinkling embers.
  - `_draw_item_bursts()`: Collection burst animation with expanding soft lilac/white smoke cloud puffs, radiant 4-pointed golden sparkle stars, and warm golden inner glow.
  - `_draw_home()`: Khuê Văn Các pavilion with carved stone pillars, flared terracotta roofs, and glowing sunburst window.
  - `_draw_player()`: Áo Ngũ Thân robe, Khăn Đóng turban, bamboo lantern with dynamic swinging red tassel, and floating `"Hoá"` nametag with contrast drop shadow.
  - `_draw_hud()`: Level name, remaining steps countdown, and `Garlic: X/Y` item counter.
- **Interactive UI Banners**: `show_overlay()` (semi-transparent alpha overlay with key-press skipping) and `show_game_over()` (automatic title, subtitle, and color selection based on outcome).

### 5.5 `env/map_levels.py` — 7-Level Curriculum Registry
Separates stage definitions from `GameEnvironment` (Open/Closed Principle) with `DEFAULT_FLOOR_Y = 320.0`.

- **`LevelConfig` *(dataclass)***: Bundles `name`, `obstacles`, `pits`, `goal`, `ghosts`, `items`, `monsters`, `floor_y`, `max_steps`.
- **`LEVEL_BUILDERS`** (`1` to `7`):
  - `1 -> _build_level_1` (*Quiet Graveyard*): 2 tombstones, no pits/ghosts/items (`max_steps=800`).
  - `2 -> _build_level_2` (*Open Graves Pits*): 3 tombstones + 2 pits (`max_steps=1000`).
  - `3 -> _build_level_3` (*Whispering Graves*): 3 tombstones + 2 pits (including 1 hellfire pit) + 2 uniform ghosts (jump-over & walk-under) + 3 garlic bulbs (`max_steps=1000`).
  - `4 -> _build_level_4` (*Nightmare Catacombs*): 5 tombstones + 4 hellfire pits + 5 diverse ghosts (including dual counter-rotating laser scissors) + 5 garlic bulbs (`max_steps=1400`).
  - `5 -> _build_level_5` (*Valley of the Wisp Maidens*): Intermediate-hard vertical gauntlet featuring 18 ghosts: 8 high-sky and mid-air Wisp Maidens flying between $y = 70$ and $y = 155$, 5 ground-level LaserGhosts sweeping 360° rotating beams, 2 ground Wisp Maidens, 3 patrolling base ghosts, 7 tombstones, 5 pits (including 3 hellfire eruptions), and 6 garlic bulbs (`max_steps=2400`).
  - `6 -> _build_level_6` (*Awakening of the Corpse Demon*): Introduces Quỷ Nhập Tràng grave ambushes, pit gap vaulting, LaserGhosts, Wisp Maidens, and flame pits (`max_steps=1500`).
  - `7 -> _build_level_7` (*Cataclysm of the Undead*): Master challenge gauntlet combining all entities under the single-monster-per-grave rule: 8 varied obstacles (Buddhist steles, Catholic crosses, floating stone platforms), 6 pits with staggered Hellfire flame cycles, 4 distinct grave ambushes (Cương Thi hopping vampires and Quỷ Nhập Tràng corpse demons), 11 diverse ghosts (laser scissors, sky/turf wisp maidens, twitchers), and 8 garlic bulbs (`max_steps=10000`).
- **Public Factories**: `get_level(level_id)` and `create_env(level_id)`.

### 5.6 Package Entry Exports (`core/__init__.py`, `env/__init__.py`)
- **`core/__init__.py`**: Exports `Action`, `RawState`, `GameEvents`, `BaseEnvironment`.
- **`env/__init__.py`**: Exports `Player`, `Obstacle`, `Pit`, `Ghost`, `LaserGhost`, `WispGhost`, `GraveMonster`, `GraveMonsterState`, `QuyNhapTrang`, `CuongThi`, `WarmthItem`, `GoalFlag`, `GameEnvironment`, `GameRenderer`, `LevelConfig`, `LEVEL_BUILDERS`, `get_level`, `create_env`.

### 5.7 `play_human.py` — Interactive Playable Entry Point
Lightweight keyboard runner (`A`/`Left`, `D`/`Right`, `Space`/`W`/`Up`, `S`/`Down`) stepping `GameEnvironment` at 60 FPS with live level switching (`1–7`), God Mode invincibility toggle (`G`), freeze-safe resets (`R`), automatic victory/death overlays (`env.show_game_over(events)`), and exit (`Esc`).

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

### 5.9 Grave Monsters, God Mode & Gauntlet Stages (Phase 4 — Complete)
- **Grave Monster Hierarchy (`GraveMonster`)**:
  - Base class providing lifecycle states (`DORMANT`, `EMERGING`, `ACTIVE`), proximity trigger check, upward emergence animation, and autonomous pit gap vaulting (`_get_upcoming_pit`, `_vault_pit`).
  - **Quỷ Nhập Tràng (`QuyNhapTrang`)**: Sprinting corpse demon with accelerating pursuit, wild billowing hair, fluttering white burial shroud, and piercing red eyes.
  - **Cương Thi (`CuongThi`)**: Jiangshi hopping vampire with 3 motion states:
    - *Dormant*: Yellow Taoist talisman on tombstone face and 3 hovering cyan spirit flames.
    - *Emerging*: Pink/white upward starburst rays, flying paper talismans, and rising stiff Qing navy robe.
    - *Active (Hop/Stand)*: Stiff bounding motion, pale cyan limbs matching arms, prominent ceremonial red mandarin hat with gold crown jewel, and dynamic forehead talisman (drapes straight down when standing; flutters back during hops to unveil piercing red eyes).
- **Single-Monster-Per-Grave Rule**:
  - Every monster is tied to a single, distinct tombstone obstacle coordinates; no tombstone shares multiple ambushes.
- **God Mode Testing Mode (`env.god_mode`)**:
  - Toggleable in human play via `[G]`; renders a golden halo above the `"Hoá"` player label and a `[GOD MODE: INVINCIBLE]` top-left HUD banner.
  - Prevents death from ghosts, lasers, flame jets, and monsters; bounces Hoá back up if falling into pits (`player.vy = -10.0`).
  - **Strict Jump Playability Rule**: Ground and mid-air jump impulses in God Mode remain strictly unboosted (`JUMP_FORCE = -13.0`) so human testers can reliably verify whether jumps and gaps are physically reachable for normal players and RL agents.

### 5.10 Procedural Level Generator & Fair Blind Evaluation Dataset (Phase 5 — Complete)
- **Dedicated Generator Architecture (`env/level_generator.py`)**:
  - Implements `LevelGenerator` following SOLID / Single Responsibility Principles.
  - Dynamic procedural layout generator for Levels 1 through 5, accepting an explicit `seed: int` parameter.
  - Generates 100% fair, physically solvable levels using cursor-based placement (`cursor_x`) with strict invariant bounds:
    - **Level 1**: Static tombstones only (`Obstacle`), 0 pits, 0 ghosts. Tombstone heights $\le 60\text{px}$ (single jump max height $\approx 105.6\text{px}$).
    - **Level 2**: Tombs + Pits (`Pit`). Pit widths bounded to $[65\text{px}, 85\text{px}]$ with run-ups $\ge 150\text{px}$ (well within single jump reach of $128\text{px}$).
    - **Level 3**: Tombs + Pits + Warmth items (`WarmthItem`) + standard patrol Ghosts (`Ghost`) with safe wave amplitudes.
    - **Level 4**: Level 3 elements + rotating laser specters (`LaserGhost`) with safe traversal intervals.
    - **Level 5**: Level 4 elements + multi-tier sinusoidal Wisp Maidens (`WispGhost`) with bounded wave amplitude and safe clearances.
- **Modular Level Integration (`env/map_levels.py`)**:
  - `get_level(level_id, seed=None)` and `create_env(level_id, seed=None)` automatically delegate to `LevelGenerator.generate(level_id, seed)` when `seed` is provided, while preserving backward compatibility with static handcrafted levels when `seed=None`.
- **Zero-Disk-Overhead Dataset Philosophy**:
  - Training pipeline can generate thousands of distinct maps on the fly (`seed = random.randint(0, 99999)`).
  - Evaluation uses fixed holdout seeds (e.g. `TEST_SEEDS = range(90000, 90100)`) for fair blind testing, preventing memorization and evaluating true generalization.
- **Expanded Automated Test Suite (`tests/test_env.py`)**:
  - 24 automated unit tests verifying all curriculum stages, God Mode standard jump invariants, procedural reproducibility, seed divergence, and curriculum entity constraints.
