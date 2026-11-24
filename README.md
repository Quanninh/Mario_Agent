# Hoá's Midnight Escape — Reinforcement Learning Platformer

A 2D midnight graveyard platformer environment engineered for **Tabular Q-Learning** research and benchmark evaluation. Built with a strict **SOLID** architecture, decoupled MDP abstractions, and high-speed headless physics simulation (>100,000 steps/sec).

---

## 1. Story & Gameplay Overview

Hoá wakes up stranded in an eerie midnight graveyard far from home. The cold is creeping in, open graves gape in the dark, and spectral ghosts hover in sinusoidal flight paths. In Vietnamese folklore, travelers through cemeteries carry a bulb of garlic to ward off evil spirits. Hoá must traverse the hazardous graveyard, avoid falling into open pits, dodge haunting ghosts (including sweeping **Laser Ghosts** and aerial **Wisp Maidens**), evade ancient undead grave ambushes (**Quỷ Nhập Tràng** and **Cương Thi**), collect glowing **Warding Garlic Bulbs (*Củ Tỏi Trừ Tà*)** to maintain stamina, and reach the sanctuary of Hanoi's iconic **Khuê Văn Các** pavilion.

---

## 2. System Architecture & SOLID Principles

The system strictly decouples the physics environment from the machine learning agents:

```
Mario_Agent/
├── core/                   # [Contract] Abstract MDP definitions & interfaces
│   ├── base_env.py         # Action, RawState, GameEvents, BaseEnvironment (ABC)
│   └── __init__.py
├── env/                    # [Physics & Graphics] 2D platformer implementation
│   ├── entities.py         # Player, Obstacle, Pit, Ghost, LaserGhost, WispGhost, GraveMonster, QuyNhapTrang, CuongThi, WarmthItem, GoalFlag
│   ├── game_env.py         # GameEnvironment (Physics simulation, AABB collisions, headless mode, God Mode)
│   ├── game_renderer.py    # GameRenderer (Pygame graphics, camera tracking, HUD, pixel art, overlay banners)
│   ├── map_levels.py       # 7 progressive curriculum stages (LevelConfig & registry)
│   └── __init__.py
├── agent/                  # [RL Agent] Tabular Q-Learning implementation
├── discretizer/            # [State Discretizer] Continuous -> discrete state space mapping
├── rewards/                # [Reward Strategy] Balanced & customized reward shaping
├── tests/                  # [Verification] Headless unit tests (20 automated unit tests)
└── play_human.py           # Interactive keyboard runner for manual playtesting (God Mode & Level Switcher)
```

### Key Architectural Highlights:
- **Dependency Inversion**: Higher-level learning modules (Q-Agent, Trainer, Reward Strategies) depend solely on abstract interfaces in `core/base_env.py`, never on Pygame or concrete entity classes.
- **Single Responsibility (Physics vs. Rendering)**: `GameEnvironment` is strictly dedicated to physics kinematics, AABB collisions, crouch slides, double jumps, and local sensor extraction. All Pygame display logic, camera tracking, and overlay banners are encapsulated in `GameRenderer`.
- **Clean Ghost & Monster Hierarchy**:
  - Base `Ghost` provides patrol and sinusoidal bobbing physics. Subclasses `LaserGhost(Ghost)` (sweeping 360° laser beam) and `WispGhost(Ghost)` (charging and casting parabolic `WillOWisp` projectiles) specialize behaviors cleanly without flag-polluting the base class.
  - Base `GraveMonster` encapsulates dormant-to-active state transitions, ground emergence, pursuit physics, and autonomous pit gap vaulting. Subclasses `QuyNhapTrang` (sprinting corpse demon with accelerating charge) and `CuongThi` (Jiangshi hopping vampire with rhythmic bounds, dynamic draped/fluttering forehead talisman, and prominent red ceremonial hat) specialize behaviors cleanly under the Open/Closed Principle.
- **Headless-First Design**: Stepping headlessly never initializes Pygame or loads graphical surfaces, ensuring blazingly fast training throughput (>100,000 steps/sec).
- **Canvas Scaling**: Internal physics and relative vision sensors operate on a clean `800x400` logical canvas, scaled smoothly to higher-resolution display viewports (`1400x700`) without altering physics equations or Q-table cardinality.

---

## 3. Curriculum Stages (Levels 1–7)

The environment supports a progressive 7-stage curriculum designed for staged reinforcement learning:

1. **Level 1 — Quiet Graveyard**: Introduction to basic jumping over static tombstones (2 tombstones, 0 pits, 0 ghosts, 0 items).
2. **Level 2 — Open Grave Pits**: Introduces lethal gap hazards requiring timed leaps (3 tombstones, 2 pits, 0 ghosts, 0 items).
3. **Level 3 — Whispering Graves**: Introduces hovering sinusoidal ghosts (low jump-over & high walk-under), hellfire erupting pits, and Warding Garlic collectible items (3 tombstones, 2 pits with flame, 2 ghosts, 3 garlic bulbs).
4. **Level 4 — Nightmare Catacombs**: Gauntlet featuring cascading hellfire pits, 5 diverse ghosts including dual counter-rotating laser scissors, fast twitcher demon, and 5 garlic bulbs.
5. **Level 5 — Valley of the Wisp Maidens**: Intermediate-hard vertical gauntlet featuring 18 ghosts: 8 high-sky and mid-air Wisp Maidens flying between $y = 70$ and $y = 155$, 5 ground-level LaserGhosts sweeping 360° rotating beams, 2 ground Wisp Maidens, 3 patrolling base ghosts, 7 tombstones, 5 pits (including 3 hellfire eruptions), and 6 garlic bulbs (`max_steps=2400`).
6. **Level 6 — Awakening of the Corpse Demon**: Introduces grave ambush mechanics featuring **Quỷ Nhập Tràng** rising from tombstones and vaulting over pit gaps in pursuit of Hoá, combined with LaserGhosts, Wisp Maidens, and flame pits (`max_steps=1500`).
7. **Level 7 — Cataclysm of the Undead**: Master challenge stage combining all entities under the **single-monster-per-grave rule**: 8 varied obstacles (Buddhist steles, Catholic granite crosses, and elevated floating stone platforms), 6 pits with staggered Hellfire flame cycles, 4 distinct grave ambushes (**Cương Thi** hopping vampires and **Quỷ Nhập Tràng** corpse demons), 11 diverse ghosts (counter-rotating laser scissors, sky and turf wisp maidens, and twitcher demons), and 8 sacred garlic bulbs (`max_steps=10000`).

---

## 4. Controls & Quick Start

### Prerequisites
- Python 3.9+
- `pygame` (required only for GUI playtesting; headless training runs with zero external dependencies)

### Manual Playtesting
To explore the environment and test level mechanics directly:
```bash
python play_human.py
```

| Key | Action |
| :--- | :--- |
| `A` / `Left Arrow` | Walk Left |
| `D` / `Right Arrow` | Walk Right |
| `Space` / `W` / `Up Arrow` | Jump (tap in mid-air for Double Jump; also skips Game Over/Win overlay) |
| `S` / `Down Arrow` | Crouch & Momentum Slide (Ducks under ghosts, height $32 \to 18\text{px}$) |
| `1` – `7` | Instant Curriculum Level Switch (Levels 1 to 7) |
| `G` | Toggle **God Mode** (Invincibility mode: pits bounce you up, hazards ignored, golden halo over Hoá, HUD badge) |
| `R` / `Enter` | Reset Level / Skip Game Over or Win Overlay |
| `Esc` | Quit Application |

### Automated Headless Tests
Run the 20 automated unit tests verifying deterministic resets, ghost/pit collision physics, item pickups, double jump & ledge recovery, momentum crouch/slide mechanics, grave pit flame hazards, rotating laser collisions, floating platform landings, Grave Monster ambushes and pit vaulting, and headless safety:
```bash
python3 -m unittest discover tests
# or
pytest tests/
```

---

## 5. Upcoming Roadmap & Enhancement Backlog

### Tier 1 — Visual & HUD Polish *(Completed)*
- [x] **Canvas Window Scaling**: Scaled display viewport to `1400x700` while preserving `800x400` physics space.
- [x] **Player Name Tag**: Floating `"Hoá"` label rendered above player hitbox with contrast drop-shadow.
- [x] **Level Name HUD**: Displays current curriculum stage in the top-left corner.
- [x] **Countdown Steps Left & Garlic Counter**: Real-time step countdown (with alert coloring at $\le 150$ steps) and collected garlic bulb counter stacked in the top-left HUD.
- [x] **Diverse Tombstones & Platforms**: Visual variants for Christian cross stone monuments (`style="cross"`), classic arch tombstones engraved with Buddhist sauwastikas (`style="buddhist"` / *chữ Vạn* `卍`), and elevated floating stone platforms (`style="platform"`).
- [x] **Authentic Vietnamese Warding Garlic Bulb (*Củ Tỏi Trừ Tà*)**: Squat 3D bulbous clove lobes, soft lilac/violet crevice shading, pure white central specular gleam, arched sickle hook stalk, compact root base, form-fitting dotted golden aura perimeter, and floating twinkling embers.
- [x] **Dynamic Collection Burst Animation**: Billowing soft lilac/white smoke cloud puffs, radiating 4-pointed golden sparkle stars with white-hot cores, and warm inner radial glow upon collecting garlic.
- [x] **Game Over & Victory Overlay**: Semi-transparent dark banner overlay pause before level reset in human play mode with instant skip on `R`, `Space`, or `Enter`.
- [x] **God Mode for Playtesting**: Invincibility toggle (`[G]`) rendering a floating golden halo over Hoá and HUD indicator.

### Tier 2 — Architecture & Goal Optimization *(Resolved)*
- [x] **SOLID Dynamic Level Architecture**: Handled via `LEVEL_BUILDERS` registry in `env/map_levels.py`, enabling arbitrary curriculum stages without modifying engine physics.
- [x] **Speedrun & Garlic Optimization**: Integrated into **Member 3's `RewardStrategy`** (step penalty `-1` + garlic pickup `+30`), driving the Q-learning agent to naturally optimize path and collection without redundant client code.

### Tier 3 — Dynamic Gameplay & Hazard Mechanics *(Completed & Active)*
- [x] **Mid-air Double Jump & Ledge Recovery (`can_double_jump`)**: Enables Hoá to execute a second jump while airborne to recover from missed pit jumps or leap over high ghosts. Also grants 1 recovery jump if stepping off an edge without jumping. Reuses `Action.JUMP` / `Action.RIGHT_JUMP` without expanding the action space; exposes `can_double_jump: bool` in `RawState`.
- [x] **Player Crouch & Momentum Slide (`Action.CROUCH`)**: Lowers Hoá's hitbox height from $32\text{px} \to 18\text{px}$ to duck under hovering ghosts. Ducks in place when stationary and performs a friction-decelerated slide ($v_{x, t} = v_{x, t-1} \cdot 0.82$) when running. Expands action space to 6 (`Action.CROUCH = 5`, `NUM_ACTIONS = 6`).
- [x] **Pit Flame Hazards (`events.hit_fire`, `pit_fire_state`)**: Periodic flame jets erupting from grave pits on timed cycles (`dormant -> warning embers -> active lethal flame -> cooldown`). Full-pit bounding box `(pit.x_start, floor_y - FIRE_HEIGHT, pit.x_end - pit.x_start, canvas_height - (floor_y - FIRE_HEIGHT))` engulfs the entire pit and reaches $65\text{px}$ above floor level. Configured in Level 4, 5, 6, & 7 with alternating phase offsets.
- [x] **Continuous 360° Rotating Laser Ghosts (`LaserGhost`)**: Specialized ghost wearing a conical straw hat (*Nón Lá*), red Daoist warding talisman (*Bùa Lệnh Trừ Ma*), and Bagua mirror halo (*Kính Bát Quái*), projecting a lethal 360° rotating scarlet laser beam (`laser_length=90.0`, `laser_rotation_speed=0.04`). Emits `events.hit_laser`.
- [x] **Wisp Maidens & Will-o'-the-Wisp Projectiles (`WispGhost`, `WillOWisp`)**: Ethereal white-robed specters with cyan hair (*Bạch Y Ma Nữ*) hovering across sky and ground, periodically charging spirit-flames and casting parabolic will-o'-the-wisp fireballs that arc through the air or roll along the ground with bouncy physics. Emits `events.hit_wisp`.
- [x] **Grave Monster Ambushes & Autonomous Pit Leaping (`GraveMonster`, `QuyNhapTrang`, `CuongThi`)**:
  - **Quỷ Nhập Tràng (Corpse Demon)**: Dormant in tombstones, bursts out when player approaches within $180\text{px}$, accelerates into a frenzied crouched sprint with billowing black hair, flapping white shroud, piercing red eyes, and autonomously leaps across upcoming grave pits.
  - **Cương Thi (Jiangshi / Chinese Hopping Vampire)**: Features 3 distinct states (dormant tomb with yellow talisman & hovering cyan spirit flames; bursting out with upward pink-white energy blast rays and flying talismans; hopping charge with rhythmic leaps, pale cyan limbs, prominent red ceremonial official hat with gold crown jewel, and dynamic forehead talisman that drapes down over the face when standing and flutters back during hops to unveil piercing red eyes).

---

## 6. Team & Contributions
- **Member 1**: Environment & Game Physics Lead (Entities, AABB Collision Engine, Pygame Visuals, Level Curriculum).
- **Member 2**: State Discretization & Tabular Q-Learning Agent.
- **Member 3**: Reward Strategy Shaping & Training Loop.
- **Member 4**: Evaluation, Performance Metrics, and Visualization.
