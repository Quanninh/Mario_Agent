# 🛠️ Member 1: Environment & Physics Progress Log

> **Owner**: Member 1  
> **Role**: Environment & Game Physics Lead  
> **Scope**: 2D Platformer Engine (Pygame), Physics, Collisions, Levels, and Human Test Mode  

---

## 1. Role Objective
Build a lightweight, headless-compatible 2D Mario platformer environment that implements the `BaseEnvironment` interface. The environment must simulate physical movement, gravity, collisions, and proximity-based state observations for the AI agent.

---

## 2. Key Design Decisions

- **Action Space**: 5 discrete actions (`STAY=0`, `LEFT=1`, `JUMP=2`, `RIGHT=3`, `RIGHT_JUMP=4`).
- **State Representation**: Uses **local vision / relative proximity** (tracking nearest upcoming obstacle, pit, and goal) rather than full-level coordinates, keeping the state space compact for Q-Learning.
- **Velocity Simplification**: Excluded horizontal sliding velocity ($v_x$) since movement strictly mirrors current input. Vertical motion uses `is_grounded` flag to prevent mid-air multi-jumps.
- **SOLID Compliance**: High-level modules (Trainer, Agent) depend on `core.BaseEnvironment`, decoupling physics implementation from RL logic.

---

## 3. Implementation Checklist

- [x] **Step 1: Core Abstraction & Contracts**
  - [x] `core/base_env.py`: Defined `Action`, `RawState`, `GameEvents`, and `BaseEnvironment`.
  - [x] `core/__init__.py`: Package export for clean imports.
- [ ] **Step 2: Game Entities** (`env/entities.py`)
  - [ ] `Mario`: Position, dimensions, vertical velocity, grounded state.
  - [ ] `Obstacle`: Pipe / block bounding box.
  - [ ] `Pit`: Floor gap hazard.
  - [ ] `GoalFlag`: Level destination.
- [ ] **Step 3: Environment Engine** (`env/mario_env.py`)
  - [ ] Gravity, jumping impulse, floor collisions.
  - [ ] Obstacle side-collisions and pit-fall death checks.
  - [ ] `reset()`: Reset to start coordinates.
  - [ ] `step(action)`: Physics tick returning `(RawState, GameEvents, done)`.
  - [ ] `render()`: Pygame visual drawing.
- [ ] **Step 4: Level Layouts** (`env/levels.py`)
  - [ ] Level 1 (Baseline): Flat floor with 1–2 pipes.
  - [ ] Level 2 (Obstacles): Multiple pipes and pits.
  - [ ] Level 3 (Generalization): Shifted obstacle positions for evaluation.
- [ ] **Step 5: Playable Human Mode** (`play_human.py`)
  - [ ] Manual keyboard controls (`A`, `D`, `Space`) to verify game feel.

---

## 4. Work Log

- **Milestone 1**: Defined `Action`, `RawState`, `GameEvents`, and `BaseEnvironment` in `core/base_env.py`. Verified package initialization with `core/__init__.py`.
