# Phased Implementation Tasks

## Phase 1: Specs Setup, Core Interfaces, Discretizer & Tabular Agent (Order #1)
- [x] **Task 1.1**: Create `specs/PRD.md`, `specs/ARCHITECTURE.md`, `specs/TASKS.md`, `specs/PROGRESS.md`.
- [x] **Task 1.2**: Ensure `core/interfaces.py` is in place and exported in `core/__init__.py` without breaking `core/base_env.py`.
- [x] **Task 1.3**: Ensure `discretizer/state_discretizer.py` (`MarioStateDiscretizer`) is in place and exported in `discretizer/__init__.py`.
- [x] **Task 1.4**: Ensure `agent/tabular_q_agent.py` (`TabularQAgent`) is in place and exported in `agent/__init__.py`.
- [x] **Task 1.5**: Run a headless Python verification command testing `create_env(1)`, `MarioStateDiscretizer`, and `TabularQAgent` for 10 steps and save/load roundtrip. Update `specs/PROGRESS.md`.

## Phase 2: v2 Sync, 7-Tuple Discretizer, Rewards & Training Engine (Order #2)
- [x] **Task 2.1**: Update `core/interfaces.py` to 7-tuple `DiscreteState` and upgrade `discretizer/state_discretizer.py` to 864-state `MarioStateDiscretizer` with above-only ghost clearance check and `PlayerStateDiscretizer` alias.
- [x] **Task 2.2**: Upgrade `rewards/` (`balanced_reward.py` with jump/stall/timeout penalties, `dense_reward.py` with timeout penalty, `sparse_reward.py`, `__init__.py` with `create_reward_strategy`).
- [x] **Task 2.3**: Implement `training/trainer.py` (`Trainer` with curriculum/separate modes, greedy-eval best checkpointing Option B, zero-shot eval on levels 3 & 5, and Member 4 JSON/CSV board export).
- [x] **Task 2.4**: Implement `training/__init__.py` exporting `Trainer`.
- [x] **Task 2.5**: Run Phase 2 integration smoke test across all 5 levels, verify 7-tuple states, checkpoints, and Member 4 artifacts. Update `specs/PROGRESS.md`.

## Phase 3: CLI Training Runner & Visual Agent Player (Order #3)
- [x] **Task 3.1**: Implement `train.py` with `--smoke-test`, zero-dependency progress tracker (`sys.stdout.isatty` inline and newline fallback), deterministic seeding, and full comparison table.
- [x] **Task 3.2**: Implement `main.py` as a thin CLI wrapper around `train.py`.
- [x] **Task 3.3**: Implement/upgrade `play_agent.py` for 60 FPS Pygame visual playback across levels 1..5, fixing event swallowing, removing duplicate FPS ticking, adding hotkeys (`1..5`, `F1..F3`, `T`, `Space` pause/resume, `R`, `Esc`), and title-bar telemetry.
- [x] **Task 3.4**: Generate comprehensive `MEMBER_3_HANDOFF_REPORT.md` documenting exact v2 output schemas, 864-state 7-tuple, and JSON/CSV artifacts.
- [x] **Task 3.5**: Run end-to-end smoke test across all 6 experiments (`curriculum`/`separate` × `balanced`/`sparse`/`dense`), verify 12 output files and checkpoints, and export summary to markdown.

## Phase 4: Pre-Server Telemetry Fixes, 3-Tuple Best Checkpointing & 2000-Episode Defaults (Order #4)
- [x] **Task 4.1**: Reset `rolling_win_rate_50` per level, fix `progress_pct` for player hitbox width (100.0% on win), and round all logged floats cleanly.
- [x] **Task 4.2**: Implement 3-tuple ranking `(num_unlocked_levels, avg_win_rate, avg_eval_reward)` for `q_table_best` in curriculum mode with per-stage backups.
- [x] **Task 4.3**: Add `states_discovered_per_level` to `metadata` in `training_board_<mode>_<reward>.json`.
- [x] **Task 4.4**: Update server production defaults in `train.py` (`episodes=2000`, `epsilon_decay=0.997`, `checkpoint_interval=400`).
- [x] **Task 4.5**: Sync `MEMBER_3_HANDOFF_REPORT.md` and execute verification smoke test validating all 4 Order #4 criteria.
