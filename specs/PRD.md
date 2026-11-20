# Product Requirements Document (PRD) — Mario RL Training & Evaluation Backend (v2)

## 1. Purpose & Scope
We are building the Reinforcement Learning backend (`core/interfaces.py`, `discretizer/`, `agent/`, `rewards/`, `training/`, `train.py`, `main.py`, `play_agent.py`) for **Hoá's Midnight Escape**, a 5-level progressive 2D platformer (`env/` built by Member 1) featuring tombstones, grave pits, `Ghost` hazards, and `WarmthItem` collectibles.
- **Level 1**: Quiet Graveyard (Baseline: flat ground, 2 tombstones, 0 pits, 0 ghosts, 0 items)
- **Level 2**: Open Graves Pits (3 tombstones, 2 grave pits, 0 ghosts, 0 items)
- **Level 3**: Cold Trail (Pit comes first, 3 tombstones, 2 pits, 3 warmth items)
- **Level 4**: Whispering Graves (Uniform ghosts, 3 tombstones, 2 pits, 2 ghosts, 3 warmth items)
- **Level 5**: Midnight Escape (Diverse ghosts, 4 tombstones, 2 pits, 4 diverse ghosts, 4 warmth items)

## 2. Team Role Integration
- **Member 1 (`env/`)**: Continuous 2D Pygame environment (`GameEnvironment`, `create_env(level_id)`, `get_level(level_id)`). Read-only.
- **Member 2 (`core/interfaces.py`, `discretizer/`, `agent/`)**: `DiscreteState` (864-state 7-tuple), `MarioStateDiscretizer` (with `PlayerStateDiscretizer` alias), and `TabularQAgent`.
- **Member 3 (`rewards/`, `training/`, `train.py`, `main.py`, `play_agent.py`)**: Reward strategies (`BalancedRewardStrategy`, `SparseRewardStrategy`, `DenseRewardStrategy`), episode training harness (`curriculum` and `separate` modes), checkpointing, and metrics export.
- **Member 4 (`evaluation/`)**: Consumes Member 3's exported `checkpoints/` (`.pkl`, `.json`) and `outputs/` (`training_board_*.json`, `training_metrics_*.csv`) for plots, comparative evaluation, and technical reporting.

## 3. Core Deliverables
1. **864-State Discretizer & Tabular Q-Agent**:
   - `DiscreteState`: 7-tuple `(is_grounded, vy_sign, obstacle_dist, obstacle_height, pit_dist, ghost_dist, ghost_alt)` with max cardinality $2 \times 3 \times 4 \times 2 \times 3 \times 3 \times 2 = 864$ states (144 baseline states when ghosts are absent).
   - Cleanly integrated with `core/base_env.py` and `TabularQAgent`.
2. **3 Modular Reward Strategies**:
   - `balanced`: Production default with asymmetric progress, goal bonus, grave pit penalty, ghost death penalty, obstacle bump penalty, warmth item bonus, step penalty, unnecessary jump discipline, stall penalty, and timeout penalty.
   - `sparse`: Baseline unshaped signal (goal bonus, pit/ghost death penalty, warmth item bonus).
   - `dense`: Ablation study using goal distance reduction, idle penalty, death penalties, and timeout penalty.
3. **Multi-Mode Training Engine (`training/trainer.py`)**:
   - Supports **Curriculum Mode** (1 shared Q-table trained sequentially across Levels 1 -> 2 -> 3 -> 4 -> 5 with partial epsilon re-warming `rewarm_epsilon = 0.30`, plus zero-shot evaluations `level_3_zero_shot` and `level_5_zero_shot`).
   - Supports **Separate Mode** (independent Q-table per level 1..5).
   - Uses `action_repeat = 1` by default so narrow `<=15px` obstacle bins and lethal hazards are never skipped.
   - Default **2000 episodes per level** (with `epsilon_decay = 0.997` and checkpoint interval `400`).
   - **Greedy-Eval Best Checkpointing (3-Tuple Ranking)**: Evaluates periodically with `set_mode(False)` across all unlocked levels, ranking candidates by `(num_unlocked_levels, avg_win_rate, avg_eval_reward)` to guarantee the global best model represents multi-level mastery. Per-stage backups saved as `q_table_lvl{N}_best.{pkl,json}`.
   - Periodic checkpoints saved every 400 episodes (`q_table_ep{N}.pkl`).
   - Final checkpoints saved as `q_table_final.pkl` and `q_table_final.json`.
   - In production curriculum run, also writes canonical files directly to root `checkpoints/`.
   - Exports Member 4 data boards (`outputs/training_board_<mode>_<reward>.json` and `outputs/training_metrics_<mode>_<reward>.csv`) tracking `items_collected`, `items_total`, `avg_items_collected`, `states_discovered_per_level`, and 4 discrete episode outcomes: `"win"`, `"pit_death"`, `"ghost_death"`, `"timeout"`. Every winning episode strictly records `progress_pct == 100.0` accounting for player hitbox width.
4. **CLI Entry Points**:
   - `train.py`: Headless training runner with `--smoke-test` and full server training flags (`episodes=2000`, `epsilon_decay=0.997`, `checkpoint_interval=400`).
   - `main.py`: Thin CLI wrapper delegating to `train.py`.
   - `play_agent.py`: 60 FPS Pygame visual replay runner with keyboard switching for levels 1..5, reward live swapping (F1..F3), variant toggle (T), pause (Space), and title-bar telemetry.
