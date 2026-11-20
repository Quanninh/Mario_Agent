# Antigravity Implementation Agent Briefing: Project Context & Architecture

> **Role of This Document**: Initial context synchronization for the **Implementation Agent (Gemini 3.8 in Antigravity)**.
>
> **Operating Protocol**: Read this specification carefully and inspect the existing workspace files. **Do not write or modify application code yet** unless explicitly instructed in a subsequent Implementation Order handout. Your first task upon receiving this handout is to verify the repository structure, confirm your understanding of the physics and architectural constraints, and report any discrepancies found in the workspace.

---

## 1. Project Overview & Team Structure

We are building a **2D Platformer Reinforcement Learning System** from scratch in Python using **Tabular Q-Learning** and **Pygame**.

### Team Task Division
* **Member 1 (`env/` — Completed v1)**: Built the 2D Pygame platformer engine, axis-separated AABB collision physics, 3 progressive levels, and manual human play mode (`play_human.py`).
* **Member 2 (`discretizer/` & `agent/` — To Be Implemented)**: State discretization (`RawState` $\to$ compact tuple key), Tabular Q-Table data structure, Bellman update rule, and $\epsilon$-greedy action selection.
* **Member 3 (`rewards/` & `training/` — Primary Focus, To Be Implemented)**: Multi-profile reward shaping (`sparse`, `balanced`, `aggressive`), episode training loops with action repeat, curriculum vs. separate training experiments, and exporting Q-tables + training/evaluation boards (JSON & CSV).
* **Member 4 (`evaluation/` — Downstream Consumer)**: Consumes Member 3's exported JSON and CSV artifacts to generate learning curve graphs, evaluation comparisons, and the technical report.

---

## 2. Current Workspace State (Member 1's v1 Engine)

The repository currently contains Member 1's working **Phase 1 (v1)** codebase:

### 2.1 `core/base_env.py` (Frozen Shared Contract)
* **`Action`**: 5 discrete actions:
  * `STAY = 0`
  * `LEFT = 1`
  * `JUMP = 2`
  * `RIGHT = 3`
  * `RIGHT_JUMP = 4`
  * `NUM_ACTIONS = 5`
* **`RawState` (dataclass)**:
  * `mario_x: float` (horizontal coordinate)
  * `mario_y: float` (vertical coordinate)
  * `velocity_y: float` (vertical velocity; negative is upward, positive is downward)
  * `is_grounded: bool` (`True` when standing on floor or atop an obstacle)
  * `dist_to_obstacle: float` (pixels from `mario.right` to nearest upcoming obstacle's left edge, or $-1.0$ if none ahead)
  * `obstacle_height: float` (height of nearest upcoming obstacle, or $0.0$ if none ahead)
  * `dist_to_pit: float` (pixels from `mario.right` to nearest upcoming pit's `x_start`, or $-1.0$ if none ahead)
  * `dist_to_goal: float` (pixels from `mario.right` to `goal.x`)
* **`GameEvents` (dataclass)**:
  * `dx: float = 0.0` ($\text{mario.x}_t - \text{mario.x}_{t-1}$)
  * `reached_goal: bool = False` (`True` when `mario.right >= goal.x`)
  * `fell_in_pit: bool = False` (`True` when `mario.y > screen_height`)
  * `hit_obstacle: bool = False` (`True` when colliding horizontally with an obstacle side)
* **`BaseEnvironment(ABC)`**: Defines `reset() -> RawState`, `step(action: int) -> Tuple[RawState, GameEvents, bool]`, `render() -> None`, and `close() -> None`.

### 2.2 `env/entities.py`, `env/mario_env.py`, and `env/map_levels.py`
* **Physics Constants**:
  * `GRAVITY = 0.8`
  * `WALK_SPEED = 4.0` (no horizontal sliding inertia; $v_x$ depends purely on current tick's action)
  * `JUMP_FORCE = -13.0`
  * `DEFAULT_FLOOR_Y = 320.0`, `screen_width = 800`, `screen_height = 400`
  * Max jump height is $\approx 105\text{ px}$; max horizontal jump span is $\approx 130\text{ px}$ over $\approx 32\text{ ticks}$.
* **Entity Hitboxes**:
  * `Mario`: $24 \times 32\text{ px}$, spawns at `start_x = 50.0`, `start_y = 288.0`.
  * `Obstacle`: width $32\text{ px}$, heights ranging from $44\text{ px}$ to $68\text{ px}$.
  * `Pit`: floor gap `[x_start, x_end]` with widths of $65\text{ px}$ to $75\text{ px}$. `Pit.contains(x, width)` checks whether Mario's horizontal center $x + \frac{\text{width}}{2}$ falls inside $(x_{\text{start}}, x_{\text{end}})$.
  * `GoalFlag`: $16 \times 96\text{ px}$ finish line.
* **Three Configured Levels (`create_env(level_id)`)**:
  * **Level 1 (Baseline)**: 2 obstacles ($x=240, h=44$; $x=480, h=56$), 0 pits, goal at $x=720$, `max_steps = 800`.
  * **Level 2 (Standard)**: 3 obstacles ($x=220, h=48$; $x=520, h=60$; $x=860, h=68$), 2 pits ($[340, 410]$ and $[680, 755]$), goal at $x=1100$, `max_steps = 1000`.
  * **Level 3 (Generalization Test)**: Pit comes first ($[200, 265]$ and $[750, 825]$) + 3 obstacles ($x=360, h=56$; $x=620, h=44$; $x=930, h=64$), goal at $x=1120$, `max_steps = 1000`.

---

## 3. Non-Negotiable Coding & Architectural Rules (`RULES.md`)

1. **Tabular Q-Learning ONLY**: Do **not** import or use PyTorch, TensorFlow, Stable-Baselines3, or Deep Q-Networks (DQN).
2. **Respect `core/` Contracts (Dependency Inversion)**: High-level modules (`training/`, `evaluation/`) must depend strictly on abstract interfaces in `core/` (`BaseEnvironment`, `BaseAgent`, `BaseDiscretizer`, `BaseRewardStrategy`), never concrete classes.
3. **Do Not Break `core/base_env.py`**: Do not modify `Action`, `RawState`, `GameEvents`, or `BaseEnvironment` in `core/base_env.py`. Place new abstract contracts (`BaseAgent`, `BaseDiscretizer`, `BaseRewardStrategy`) in `core/contracts.py` and re-export them in `core/__init__.py`.
4. **Headless-First Speed**: Never call `pygame.init()` or `env.render()` during training loops so thousands of episodes execute in seconds.
5. **Module Boundaries**:
   * `core/`: Abstract interfaces and shared data contracts.
   * `env/`: Member 1's physics and level files (read-only unless fixing a verified bug).
   * `discretizer/`: State bucketing logic implementing `BaseDiscretizer`.
   * `agent/`: Tabular Q-table and Bellman update logic implementing `BaseAgent`.
   * `rewards/`: Reward calculation logic and reward profiles implementing `BaseRewardStrategy`.
   * `training/`: Episode runner, action-repeat wrapper, curriculum/separate experiment manager, and JSON/CSV exporters.
   * `outputs/`: Destination folder for generated Q-tables, JSON boards, and CSV metrics.

---

## 4. Locked-In Engineering Decisions (v1 Target)

### 4.1 Action Repeat (`action_repeat = 4`) in the Training Loop
* Because `MarioEnvironment.step(action)` advances by a single physics tick ($4.0\text{ px}$ horizontally) and recalculates horizontal velocity $v_x$ on every tick, choosing random actions every single tick causes the agent to stall in mid-air during a $32$-tick jump arc.
* Furthermore, when Mario steps into a pit at $y = 320.0$, `events.fell_in_pit` only triggers once $y > 400.0$ ($\approx 10\text{ ticks}$ later).
* **Decision**: The training loop repeats each selected action $a_t$ for up to $K = 4$ physics ticks (breaking immediately if `done` becomes `True`), summing the step rewards into $R_{\text{macro}}$ before performing the Bellman update on the resulting state $s'_{t}$.

### 4.2 Compact State Discretization ($450$ Max States)
* `StateDiscretizer.discretize(state: RawState)` converts continuous measurements into a 5-integer tuple:
  `(grounded, vy_dir, obs_dist, obs_h, pit_dist)`
* Uses local relative proximity (`dist_to_obstacle`, `obstacle_height`, `dist_to_pit`) rather than absolute `mario_x` so a single Q-table generalizes across Levels 1, 2, and 3.

### 4.3 Multi-Profile Reward Strategy (`sparse`, `balanced`, `aggressive`)
* `BaseRewardStrategy.compute_reward(events, action, done, raw_state)` evaluates:
  1. Rightward progress ($\text{dx} > 0$) vs. leftward backtracking ($\text{dx} < 0$) vs. stalling ($\text{dx} = 0$).
  2. Obstacle side collision penalty (`events.hit_obstacle`).
  3. Unnecessary jump penalty: penalizes selecting `Action.JUMP` or `Action.RIGHT_JUMP` while grounded when no obstacle or pit is within $110\text{ px}$ ahead.
  4. Terminal outcomes: `events.reached_goal` (large bonus), `events.fell_in_pit` (large penalty), or episode timeout (`done` is `True` while neither `reached_goal` nor `fell_in_pit` is `True`).

### 4.4 Curriculum vs. Separate Training & Member 4 Handoff Outputs
* The training engine supports two modes for report comparison:
  * **`curriculum`**: Trains one shared `QLearningAgent` sequentially across Level 1 $\to$ Level 2 $\to$ Level 3, re-warming $\epsilon \leftarrow \max(0.35, \epsilon)$ at the start of Levels 2 and 3.
  * **`separate`**: Trains an independent `QLearningAgent` from scratch on each level.
* Exports three synchronized artifact types into `outputs/` for Member 4:
  1. `q_table_<mode>_<profile>.json` (and `.pkl`)
  2. `training_board_<mode>_<profile>.json` (containing `metadata`, `evaluation_summary` with untrained vs. trained benchmarks, and `episode_history`)
  3. `training_metrics_<mode>_<profile>.csv`

---

## 5. Initial Instructions for the Antigravity Agent

1. Inspect the current workspace directory tree and read `core/base_env.py`, `env/entities.py`, `env/mario_env.py`, `env/map_levels.py`, `play_human.py`, and `RULES.md`.
2. Check whether any directories or files for `discretizer/`, `agent/`, `rewards/`, `training/`, or `evaluation/` already exist in the workspace.
3. Reply with a brief status summary confirming:
   * Current git branch and existing folder layout.
   * Python environment readiness (`pygame`, `numpy` availability).
   * Readiness to receive **Implementation Order #1**.