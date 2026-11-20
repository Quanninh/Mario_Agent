# System Architecture & Data Flow (v2)

## 1. Tech Stack
- **Language**: Python 3.10+
- **Libraries**: `numpy`, `pygame` (lazy-initialized only in `render()`), standard library (`json`, `csv`, `pickle`, `pathlib`, `dataclasses`, `abc`, `argparse`, `typing`, `collections`).
- **Forbidden**: PyTorch, TensorFlow, Stable-Baselines3, Gymnasium.

## 2. Directory Structure
```text
project_root/
├── specs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── TASKS.md
│   └── PROGRESS.md
├── core/
│   ├── __init__.py
│   ├── base_env.py              # Member 1 (DO NOT MODIFY: Action, RawState, GameEvents, BaseEnvironment)
│   └── interfaces.py            # DiscreteState (7-tuple), BaseDiscretizer, BaseRewardStrategy, BaseAgent
├── env/                         # Member 1 (DO NOT MODIFY)
│   ├── __init__.py
│   ├── entities.py              # Player, Obstacle, Pit, Ghost, WarmthItem, GoalFlag
│   ├── game_env.py              # GameEnvironment
│   └── map_levels.py            # Level 1..5 builders, create_env(level_id)
├── discretizer/
│   ├── __init__.py
│   └── state_discretizer.py     # MarioStateDiscretizer (864 states, PlayerStateDiscretizer alias)
├── agent/
│   ├── __init__.py
│   └── tabular_q_agent.py       # TabularQAgent (supports 7-tuple state serialization)
├── rewards/
│   ├── __init__.py              # Exports strategies + create_reward_strategy helper
│   ├── base_reward.py
│   ├── balanced_reward.py       # BalancedRewardStrategy (production default)
│   ├── sparse_reward.py         # SparseRewardStrategy (evaluation baseline)
│   └── dense_reward.py          # DenseRewardStrategy (ablation baseline)
├── training/
│   ├── __init__.py
│   └── trainer.py               # Trainer class (curriculum & separate modes, greedy-eval best checkpointing)
├── checkpoints/                 # Generated .pkl and .json Q-tables
│   ├── <mode>_<reward>/         # Run-specific checkpoints (q_table_best.*, q_table_ep{N}.pkl, q_table_final.*)
│   ├── q_table_best.pkl         # Canonical production best Q-table
│   ├── q_table_best.json        # Canonical production best JSON
│   ├── q_table_final.pkl        # Canonical production final Q-table
│   └── q_table_final.json       # Canonical production final JSON
├── outputs/                     # Generated training_board_*.json and training_metrics_*.csv
├── play_human.py                # Member 1 (DO NOT MODIFY)
├── train.py                     # CLI training & evaluation harness (Order #3)
├── main.py                      # Thin CLI wrapper around train.py (Order #3)
└── play_agent.py                # 60 FPS visual agent demo with level 1..5 switching (Order #3)
```

## 3. Data Flow Per Physics Tick (`action_repeat = 1`)
1. `raw_state = env.reset()` -> `state = discretizer.discretize(raw_state)` (7-tuple in 864-state space).
2. `action = agent.select_action(state)` (`0..4`: `STAY`, `LEFT`, `JUMP`, `RIGHT`, `RIGHT_JUMP`).
3. `next_raw_state, events, done = env.step(action)` (repeated up to `action_repeat` times; default `1`).
4. `next_state = discretizer.discretize(next_raw_state)`.
5. `reward = reward_strategy.calculate_reward(raw_state, action, next_raw_state, events, done)`.
6. `td_error = agent.update(state, action, reward, next_state, done)` (in RAM).
7. Advance: `raw_state = next_raw_state`, `state = next_state`.
8. At episode completion:
   - `eps = agent.decay_epsilon()`.
   - Update `episode_history` and rolling metrics.
   - Periodic greedy evaluation every `eval_interval` episodes: if `(avg_win_rate, avg_reward)` beats `best_eval_score`, persist `q_table_best.pkl` and `q_table_best.json`.
   - Periodic checkpoint every `checkpoint_interval` episodes (`q_table_ep{N}.pkl`).
   - Export Member 4 JSON and CSV data boards at end of experiment.
