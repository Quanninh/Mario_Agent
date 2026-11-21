# Hoá's Midnight Escape — Reinforcement Learning Platformer

A 2D midnight graveyard platformer environment engineered for **Tabular Q-Learning** research and benchmark evaluation. Built with a strict **SOLID** architecture, decoupled MDP abstractions, and high-speed headless physics simulation (>100,000 steps/sec).

---

## 1. Story & Gameplay Overview

Hoá wakes up stranded in an eerie midnight graveyard far from home. The cold is creeping in, open graves gape in the dark, and spectral ghosts hover in sinusoidal flight paths. Hoá must traverse the hazardous graveyard, avoid falling into open pits, dodge haunting ghosts, collect glowing warmth items to maintain stamina, and reach the sanctuary of his home.

---

## 2. System Architecture & SOLID Principles

The system strictly decouples the physics environment from the machine learning agents:

```
Mario_Agent/
├── core/                   # [Contract] Abstract MDP definitions & interfaces
│   ├── base_env.py         # Action, RawState, GameEvents, BaseEnvironment (ABC)
│   └── __init__.py
├── env/                    # [Physics & Graphics] 2D platformer implementation
│   ├── entities.py         # Player, Obstacle, Pit, Ghost, WarmthItem, GoalFlag
│   ├── game_env.py         # GameEnvironment (Physics simulation, AABB collisions, headless mode)
│   ├── game_renderer.py    # GameRenderer (Pygame graphics, camera tracking, HUD, overlay banners)
│   ├── map_levels.py       # 5 progressive curriculum stages (LevelConfig & registry)
│   └── __init__.py
├── agent/                  # [RL Agent] Tabular Q-Learning implementation
├── discretizer/            # [State Discretizer] Continuous -> discrete state space mapping
├── rewards/                # [Reward Strategy] Balanced & customized reward shaping
├── tests/                  # [Verification] Headless unit tests (0.02s suite execution)
└── play_human.py           # Interactive keyboard runner for manual playtesting
```

### Key Architectural Highlights:
- **Dependency Inversion**: Higher-level learning modules (Q-Agent, Trainer, Reward Strategies) depend solely on abstract interfaces in `core/base_env.py`, never on Pygame or concrete entity classes.
- **Single Responsibility (Physics vs. Rendering)**: `GameEnvironment` is strictly dedicated to physics kinematics, AABB collisions, crouch slides, double jumps, and local sensor extraction. All Pygame display logic, camera tracking, and overlay banners are encapsulated in `GameRenderer`.
- **Headless-First Design**: Stepping headlessly never initializes Pygame or loads graphical surfaces, ensuring blazingly fast training throughput (>100,000 steps/sec).
- **Canvas Scaling**: Internal physics and relative vision sensors operate on a clean `800x400` logical canvas, scaled smoothly to higher-resolution display viewports (`1400x700`) without altering physics equations or Q-table cardinality.

---

## 3. Curriculum Stages (Levels 1–5)

The environment supports a progressive 5-stage curriculum designed for staged reinforcement learning:

1. **Level 1 — Quiet Graveyard**: Introduction to basic jumping over static tombstones (0 pits, 0 ghosts, 0 items).
2. **Level 2 — Open Grave Pits**: Introduces lethal gap hazards requiring timed leaps.
3. **Level 3 — Cold Trail**: Introduces Warmth Items placed along challenging jump trajectories.
4. **Level 4 — Whispering Graves**: Introduces hovering, sinusoidal ghosts with uniform speeds and patrol zones.
5. **Level 5 — Midnight Escape**: The ultimate gauntlet featuring 4 diverse ghosts with varying velocities, bobbing frequencies, and wave amplitudes alongside pits and items.

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
| `Space` / `W` / `Up Arrow` | Jump (also skips Game Over/Win overlay) |
| `S` / `Down Arrow` | Crouch & Momentum Slide (Ducks under ghosts, height $32 \to 18\text{px}$) |
| `1` – `5` | Instant Curriculum Level Switch |
| `R` / `Enter` | Reset Level / Skip Game Over or Win Overlay |
| `Esc` | Quit Application |

### Automated Headless Tests
Run the 13 automated unit tests verifying deterministic resets, ghost/pit collision physics, item pickups, double jump & ledge recovery, momentum crouch/slide mechanics, and grave pit flame hazards:
```bash
python3 -m unittest discover tests
```

---

## 5. Upcoming Roadmap & Enhancement Backlog

Future enhancements categorized from immediate visual polish to advanced dynamic gameplay mechanics:

### Tier 1 — Visual & HUD Polish *(Completed)*
- [x] **Canvas Window Scaling**: Scaled display viewport to `1400x700` while preserving `800x400` physics space.
- [x] **Player Name Tag**: Floating `"Hoá"` label rendered above player hitbox with contrast drop-shadow.
- [x] **Level Name HUD**: Displays current curriculum stage in the top-left corner.
- [x] **Countdown Steps Left & Warmth Counter**: Real-time step countdown (with alert coloring at $\le 150$ steps) and collected item counter stacked in the top-left HUD.
- [x] **Diverse Tombstones**: Visual variants for Christian cross stone monuments (`style="cross"`) and classic arch tombstones engraved with Buddhist sauwastikas (`style="buddhist"` / *chữ Vạn* `卍`).
- [x] **Game Over & Victory Overlay**: Semi-transparent dark banner overlay pause before level reset in human play mode with instant skip on `R`, `Space`, or `Enter`.

### Tier 2 — Architecture & Goal Optimization *(Resolved)*
- [x] **SOLID Dynamic Level Architecture**: Handled via `LEVEL_BUILDERS` registry in `env/map_levels.py`, enabling arbitrary curriculum stages without modifying engine physics.
- [x] **Speedrun & Warmth Optimization**: Integrated into **Member 3's `RewardStrategy`** (step penalty `-1` + warmth pickup `+30`), driving the Q-learning agent to naturally optimize path and collection without redundant client code.

### Tier 3 — Dynamic Gameplay & Hazard Mechanics *(Active Focus)*
- [x] **Mid-air Double Jump & Ledge Recovery (`can_double_jump`)**: Enables Hoá to execute a second jump while airborne to recover from missed pit jumps or leap over high ghosts. Also grants 1 recovery jump if stepping off an edge without jumping. Reuses `Action.JUMP` / `Action.RIGHT_JUMP` without expanding the action space; exposes `can_double_jump: bool` in `RawState`.
- [x] **Player Crouch & Momentum Slide (`Action.CROUCH`)**: Lowers Hoá's hitbox height from $32\text{px} \to 18\text{px}$ to duck under hovering ghosts. Ducks in place when stationary and performs a friction-decelerated slide ($v_{x, t} = v_{x, t-1} \cdot 0.82$) when running. Expands action space to 6 (`Action.CROUCH = 5`, `NUM_ACTIONS = 6`).
- [x] **Pit Flame Hazards (`events.hit_fire`, `pit_fire_state`)**: Periodic flame jets erupting from grave pits on timed cycles (`dormant -> warning embers -> active lethal flame -> cooldown`). Full-pit bounding box `(pit.x_start, floor_y - FIRE_HEIGHT, pit.x_end - pit.x_start, canvas_height - (floor_y - FIRE_HEIGHT))` engulfs the entire pit and reaches $40\text{px}$ above floor level. Configured in Level 4 & Level 5 with alternating phase offsets.
- [ ] **Rotating Ghost Eye-Lasers**: Ghosts firing lethal rotating laser beams while patrolling.
- [ ] **Rising & Charging Skeleton Ambush**: Special mausoleum tombs that trigger when Hoá approaches (proximity $< 180\text{px}$): a skeleton bursts upward from the grave, leaps forward, and sprints aggressively toward the player (`dx = player.x - skeleton.x`), requiring rapid jump evasion.

---

## 6. Team & Contributions
- **Member 1**: Environment & Game Physics Lead (Entities, AABB Collision Engine, Pygame Visuals, Level Curriculum).
- **Member 2**: State Discretization & Tabular Q-Learning Agent.
- **Member 3**: Reward Strategy Shaping & Training Loop.
- **Member 4**: Evaluation, Performance Metrics, and Visualization.
