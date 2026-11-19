# Project Coding Rules & AI Guardrails

> Shared rules for all 4 team members and AI coding assistants working on **Hoá's Midnight Escape**.

---

## 1. Scope & Algorithm Rules
- **Tabular Q-Learning ONLY**: Do **not** add PyTorch, TensorFlow, Stable-Baselines3, or DQN code unless explicitly requested as a post-submission extension.
- **Keep State Space Compact**: Use local vision (nearest upcoming obstacle, pit, ghost, goal). Never pass raw pixel arrays or full lists of all level entities into the Q-table.
- **No Redundant Physics Variables**: Do **not** add `velocity_x` to the player entity or `RawState` (horizontal movement has no sliding inertia).

## 2. SOLID & Clean Code Rules
- **Respect `core/` Contracts**: High-level modules (`training/`, `evaluation/`) must depend on interfaces in `core/` (`BaseEnvironment`, `BaseAgent`, `BaseDiscretizer`, `BaseRewardStrategy`), never concrete classes.
- **Do Not Break Shared Interfaces Silently**: Any change to `RawState`, `GameEvents`, or `Action` in `core/base_env.py` must be documented and synced with the team.
- **Single Responsibility**:
  - `env/entities.py` holds data & hitbox geometry only (no Pygame rendering, no RL logic).
  - `env/game_env.py` (`GameEnvironment`) handles physics, collisions, and rendering only (no reward formulas, no Q-table logic).
  - `env/map_levels.py` holds level definitions via `LEVEL_BUILDERS` registry (Open/Closed Principle).
- **Keep It Simple (KISS)**: Avoid redundant parameters (e.g., use `DEFAULT_FLOOR_Y` constant instead of threading `floor_y` through every builder) and unnecessary state variables.

## 3. Pygame & Simulation Rules
- **Headless-First Environment**: `pygame.init()` and `pygame.display.set_mode()` must remain **lazy-initialized inside `render()`** so training runs at thousands of FPS without opening a window.
- **Always Pump Events**: Any loop branch using `pygame.key.get_pressed()` followed by `continue` must call `env.render()` (or `pygame.event.pump()`) first—otherwise macOS Pygame windows freeze.
- **Axis-Separated Collisions**: Always resolve X-axis collisions before Y-axis collisions, and guard floor landing with `old_bottom <= floor_y` so the player cannot teleport out of pits.

## 4. Team & Git Rules
- **Stay in Your Own Module**: Each member primarily edits their assigned folder (`env/`, `agent/` + `discretizer/`, `rewards/` + `training/`, `evaluation/`) to avoid merge conflicts.
- **Docs Before Big Refactors**: Update  your member log when completing a milestone or changing design plans.
