# Complete Training & Visual Demo Guide for Hoá's Midnight Escape

This guide provides step-by-step instructions for training reinforcement learning agents (`train.py`) and launching the interactive 60 FPS visual playback demonstration (`play_agent.py`). It is designed to work on any teammate's machine across Windows, macOS, or Linux.

---

## 1. Environment & Setup

### 1.1 Prerequisites
- **Python Version:** Python 3.10 or higher
- **Required Packages:** `pygame` ($\ge 2.6.0$) and `numpy` ($\ge 1.24.0$)

Install the dependencies into your active Python environment (virtualenv or conda):

```bash
pip install pygame numpy
```

*(Alternatively, if using Conda: `conda install pygame numpy`)*

### 1.2 Execution
Ensure your target environment is activated, then run the scripts from the repository root:

```bash
python train.py [OPTIONS]
python play_agent.py [OPTIONS]
```

---

## 2. Training Guide (`train.py`)

The training harness runs headless RL experiments without rendering overhead, processing thousands of simulation steps per second.

### 2.1 Available Agent Architectures & Modes

| Concept | Choices | Description |
| :--- | :--- | :--- |
| **`--agent`** | `tabular`, `hybrid`, `all` | `tabular` (Pure Q-Learning), `hybrid` (Hybrid SARSA/Q-Learning), or `all` (both). |
| **`--hybrid-mode`** | `expected`, `linear`, `hazard_gated` | Target formulation for hybrid agent (default: `expected`). |
| **`--mode`** | `curriculum`, `separate`, `all` | `curriculum` (sequential transfer L1 $\to$ L5 with Option B replay) or `separate` (fresh agent per level). |
| **`--reward`** | `balanced`, `sparse`, `dense`, `all` | `balanced` (production shaped), `sparse` (unshaped goal/death), or `dense` (distance-based ablation). |

---

### 2.2 Standard Training Workflows

#### Workflow A: Recommended 4-Run Balanced Benchmark (2,000 Episodes / Level)
Trains both `TabularQAgent` and `HybridQSARSAAgent` across `curriculum` and `separate` modes on the production `balanced` reward:
```bash
python train.py --agent all --mode all --reward balanced --episodes 2000
```

#### Workflow B: Canonical Single Production Run (Tabular Curriculum Balanced)
Trains the primary baseline agent across Levels 1 $\to$ 5 in curriculum mode. Saves the global best model directly to root [`checkpoints/q_table_best.json`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/checkpoints/q_table_best.json):
```bash
python train.py --agent tabular --mode curriculum --reward balanced --episodes 2000
```

#### Workflow C: Fast Smoke Test (Verification in ~10 Seconds)
Runs 5 episodes per level with small evaluation windows to verify code changes, checkpoint saving, and metrics generation without waiting:
```bash
python train.py --smoke-test --agent all --reward balanced
```

#### Workflow D: Full 12-Run Experimental Matrix (All Rewards)
Executes all 2 Agents $\times$ 2 Modes $\times$ 3 Rewards for research reports and Member 4 evaluation boards:
```bash
python train.py --agent all --mode all --reward all --episodes 2000 --no-progress-bar
```

#### Workflow E: Single Level Training (Separate Mode)
To train an independent agent exclusively on Level 4 or Level 5:
```bash
python train.py --agent tabular --mode separate --reward balanced --levels 4 --episodes 2000
```

---

### 2.3 CLI Flags Reference

| Flag | Default | Description |
| :--- | :---: | :--- |
| **`--agent`** | `"all"` | Agent algorithm: `tabular`, `hybrid`, or `all`. |
| **`--hybrid-mode`**| `"expected"` | Target formulation: `expected`, `linear`, or `hazard_gated`. |
| **`--mode`** | `"all"` | Mode: `curriculum`, `separate`, or `all`. |
| **`--reward`** | `"all"` | Reward strategy: `balanced`, `sparse`, `dense`, or `all`. |
| **`--levels`** | `1 2 3 4 5` | Levels to include in the run (space-separated). |
| **`--episodes`** | `2000` | Number of training episodes per level. |
| **`--seed`** | `42` | Random seed for deterministic reproducibility. |
| **`--rewarm-epsilon`** | `0.30` | Exploration rate when transitioning to Levels 2–3 in curriculum mode. |
| **`--rewarm-epsilon-ghost`**| `0.35` | Exploration rate when transitioning to ghost levels (Levels 4–5). |
| **`--replay-ratio`** | `0.35` | Probability of background replay episodes from earlier levels in curriculum mode. |
| **`--no-progress-bar`** | `False` | Disables in-place carriage returns (`\r`) in favor of clean newline log lines (recommended when logging to files or Slurm). |
| **`--smoke-test`** | `False` | Overrides parameters for fast 5-episode smoke test verification. |

---

### 2.4 Training Output Artifacts

Training automatically saves models and telemetry across two main folders:

1. **Checkpoints ([`checkpoints/`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/checkpoints)):**
   - `checkpoints/tabular/curriculum_balanced/` $\to$ `q_table_best.{json,pkl}`, `q_table_final.{json,pkl}`, per-stage `q_table_lvl{1..5}_best.{json,pkl}`.
   - `checkpoints/tabular/separate_balanced/` $\to$ `q_table_lvl{1..5}_best.{json,pkl}`, `q_table_lvl{1..5}_final.{json,pkl}`.
   - `checkpoints/hybrid_expected/curriculum_balanced/` $\to$ matching layout for hybrid agent.
   - `checkpoints/q_table_best.{json,pkl}` $\to$ canonical production mirror.
2. **Metrics & Data Boards ([`outputs/`](file:///d:/Study%20Resources/Year%204/Elective%20Compulsory%201/Mario_Agent/outputs)):**
   - `outputs/training_board_<agent_tag>_<mode>_<reward>.json` $\to$ complete training metadata, evaluation summary, and episode history.
   - `outputs/training_metrics_<agent_tag>_<mode>_<reward>.csv` $\to$ per-episode CSV data for pandas/matplotlib analysis.

---

## 3. Visual Demo & Playback Guide (`play_agent.py`)

The visual player renders Hoá in real time at **60 FPS** using Pygame, displaying policy decisions, obstacle clearance, ghost avoidance, and warmth item collection.

### 3.1 Launching the Demo

#### Quick Launch (Zero Configuration)
Loads the canonical production best policy on Level 1:
```bash
python play_agent.py
```

#### Launching Specific Levels
```bash
# Play Level 4 with trained Tabular Q-Agent
python play_agent.py --agent tabular --level 4

# Play Level 5 with trained Hybrid Agent (Expected SARSA)
python play_agent.py --agent hybrid --hybrid-mode expected --level 5
```

#### Launching with Specific Training Modes or Checkpoints
```bash
# Play Level 4 using a model trained in separate mode
python play_agent.py --mode separate --reward balanced --level 4

# Load an explicit checkpoint file directly
python play_agent.py --checkpoint checkpoints/tabular/curriculum_balanced/q_table_best.json --level 4
```

#### Slow-Motion Observation (20 FPS)
Useful for inspecting exact jump take-off margins and ghost clearance:
```bash
python play_agent.py --level 4 --fps 20
```

---

### 3.2 In-Game Interactive Hotkeys

While the Pygame game window is open and focused, you can control playback in real time without restarting the script:

| Key | Action | What It Does |
| :---: | :--- | :--- |
| **`[H / A]`** | **Toggle Agent Architecture** | Switches live between **`TabularQAgent`** and **`HybridAgent`**, immediately reloading that agent family's matching checkpoint and restarting the episode. |
| **`[1 / 2 / 3 / 4 / 5]`** | **Switch Level** | Instantly switches the active graveyard level ($1 \dots 5$). In separate mode, it automatically loads that level's dedicated Q-table. |
| **`[F1 / F2 / F3]`** | **Swap Reward Strategy** | Hot-swaps policy between **Balanced** (`F1`), **Sparse** (`F2`), and **Dense** (`F3`). |
| **`[T]`** | **Toggle Model Variant** | Toggles between the **`best`** checkpoint (peak evaluation win rate) and **`final`** checkpoint (latest episode). |
| **`[Space]`** | **Pause / Resume** | Freezes simulation physics; updates window title to `[PAUSED]`. |
| **`[R]`** | **Restart Episode** | Resets Hoá, items, and ghosts to their initial starting positions. |
| **`[Esc]`** | **Quit** | Closes the window and cleanly exits. |

---

### 3.3 Window Title Telemetry Explained

The Pygame window title bar updates dynamically every frame:

```text
Hoá's Midnight Escape | Lvl 4 | CURRICULUM-BALANCED [BEST] | Agent: TABULAR | Step: 142 | Items: 2/3 | Act: RIGHT_JUMP
```

- **`Lvl 4`**: Current active level.
- **`CURRICULUM-BALANCED [BEST]`**: Checkpoint mode, reward strategy, and variant currently loaded.
- **`Agent: TABULAR`** (or `HYBRID (expected)`): Active agent algorithm.
- **`Step: 142`**: Simulation ticks elapsed in the current episode.
- **`Items: 2/3`**: Warmth items collected vs. total placed in the level.
- **`Act: RIGHT_JUMP`**: Action chosen by the policy for the current frame.
- **`[PAUSED]`**: Displayed whenever physics are suspended via `[Space]`.

---

## 4. End-to-End Walkthrough Tutorial

Follow these 4 steps to run a complete experiment from scratch and watch the agent navigate:

### Step 1: Run Training
Execute the 4-run balanced benchmark:
```bash
python train.py --agent all --mode all --reward balanced --episodes 2000
```
Wait for the run to complete. You will see the final evaluation comparison table printed to the terminal.

### Step 2: Confirm Saved Checkpoint Files
Check that your models are saved (Linux/macOS `ls` or Windows `dir`):
```bash
ls checkpoints/tabular/curriculum_balanced
ls checkpoints/hybrid_expected/curriculum_balanced
```

### Step 3: Launch Visual Playback on Level 4
Launch the visual demo on Level 4:
```bash
python play_agent.py --agent hybrid --level 4
```

### Step 4: Interact Live in the Window
1. Watch Hoá walk underneath High Ghost 2 and collect the ground warmth item.
2. Press **`[H]`** to switch live to `TabularQAgent` and compare behavior.
3. Press **`[5]`** to switch to Level 5 and spectate the diverse ghost patrols.
4. Press **`[Space]`** to pause whenever you want to inspect a jump clearance.
5. Press **`[Esc]`** to exit when finished.

---

## 5. Troubleshooting & FAQs

### Q: Why does the terminal show warning: `Checkpoint '...' not found. Policy is uninitialized`?
- **A:** The specified checkpoint has not been trained yet. `play_agent.py` will still run using random tie-breaking. Run `train.py` first to generate trained weights.

### Q: Why is the progress bar printing on multiple lines instead of updating in-place?
- **A:** Some terminals or embedded IDE consoles do not support carriage returns (`\r`). Pass `--no-progress-bar` to `train.py` for clean newline logging every 100 episodes.

### Q: Can I run human manual play to compare against the agent?
- **A:** Yes! Run Member 1's human player script:
  ```bash
  python play_human.py --level 4
  ```
  Controls: Left/Right arrows to walk, Space to jump.
