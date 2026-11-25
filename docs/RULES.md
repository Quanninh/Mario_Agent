# Project Coding Rules & AI Guardrails

> Shared rules for all 4 team members and AI coding assistants working on **Hoá's Midnight Escape**.

---

## 1. Scope & Algorithm Rules
- **Tabular Q-Learning ONLY**: Do **not** add PyTorch, TensorFlow, Stable-Baselines3, or DQN code unless explicitly requested as a post-submission extension.
- **Keep State Space Compact**: Use local vision (nearest upcoming obstacle, pit, fire phase, ghost, garlic item, goal). Never pass raw pixel arrays or full lists of all level entities into the Q-table.
- **Physical Kinematics & Strict Playability Guardrail**:
  - Horizontal movement is discrete input-driven except during **momentum crouch slides** (`Action.CROUCH`), where `velocity_x` undergoes friction deceleration ($v_{x, t} = v_{x, t-1} \cdot 0.82$).
  - **CRITICAL / STRICT GUARDRAIL — DO NOT MODIFY JUMP IMPULSE IN GOD MODE**:
    Never uncomment or apply jump boost multiplier in `God Mode`:
    ```python
    # DO NOT UNCOMMENT THIS:
    # jump_impulse = self.JUMP_FORCE * 1.35 if getattr(self, "god_mode", False) else self.JUMP_FORCE
    ```
    `God Mode` is strictly used by human testers to inspect level geometry and test **human playability/reachability**. Artificially increasing jump force distorts physics and causes false-positive playability passes. In `God Mode`, `JUMP_FORCE` must strictly remain `-13.0` (infinite mid-air flutter leaps are allowed, but vertical impulse per leap must remain unmodified).

## 2. SOLID & Clean Code Rules
- **Respect `core/` Contracts**: High-level modules (`training/`, `evaluation/`) must depend on interfaces in `core/` (`BaseEnvironment`, `BaseAgent`, `BaseDiscretizer`, `BaseRewardStrategy`), never concrete classes.
- **Do Not Break Shared Interfaces Silently**: Any change to `RawState`, `GameEvents`, or `Action` in `core/base_env.py` must be documented and synced with the team.
- **Clean Inheritance Over Dummy Flags**: Never add unused dummy boolean flags (e.g. `has_laser`) to base classes; use proper object-oriented inheritance (`LaserGhost`, `WispGhost`) and type checks (`isinstance()`).
- **Single Responsibility**:
  - `env/entities.py` holds data & hitbox geometry only (no Pygame rendering, no RL logic).
  - `env/game_env.py` (`GameEnvironment`) handles physics, collisions, and state extraction only (no reward formulas, no Q-table logic, no surface rendering).
  - `env/game_renderer.py` (`GameRenderer`) encapsulates all Pygame graphics, camera tracking, HUD, and banners.
  - `env/map_levels.py` holds level definitions via `LEVEL_BUILDERS` registry (Open/Closed Principle).
- **Keep It Simple (KISS)**: Avoid redundant parameters (e.g., use `DEFAULT_FLOOR_Y` constant instead of threading `floor_y` through every builder) and unnecessary state variables.

## 3. Pygame & Simulation Rules
- **Headless-First Environment**: `pygame.init()` and `pygame.display.set_mode()` must remain **lazy-initialized inside `render()`** so training runs at thousands of FPS without opening a window.
- **Always Pump Events**: Any loop branch using `pygame.key.get_pressed()` followed by `continue` must call `env.render()` (or `pygame.event.pump()`) first—otherwise macOS Pygame windows freeze.
- **Axis-Separated Collisions**: Always resolve X-axis collisions before Y-axis collisions, and guard floor landing with `old_bottom <= floor_y` so the player cannot teleport out of pits.

## 4. Team & Git Rules
- **Stay in Your Own Module**: Each member primarily edits their assigned folder (`env/`, `agent/` + `discretizer/`, `rewards/` + `training/`, `evaluation/`) to avoid merge conflicts.
- **Docs Before Big Refactors**: Update  your member log when completing a milestone or changing design plans.
