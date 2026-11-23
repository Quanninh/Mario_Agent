# Autonomous Agent Playback Guide (`play_agent.py`)

This guide explains how to execute, configure, and spectate trained reinforcement learning agents in **Hoá's Midnight Escape** using `play_agent.py`.

---

## 1. Overview & Purpose

`play_agent.py` provides a real-time, 60 FPS interactive Pygame visualizer that renders Hoá navigating through graveyard levels under the control of a learned policy.

### Key Capabilities
- **Dual-Architecture Choice**: Choose between **Pure Tabular Q-Learning (`TabularQAgent`)** and **Hybrid SARSA/Q-Learning (`HybridAgent` / `HybridQSARSAAgent`)**.
- **Hybrid Target Formulations**: Configure hybrid updating via `--hybrid-mode` (`expected`, `linear`, `hazard_gated`).
- **Clean Folder Resolution**: Seamlessly loads from the structured hierarchy `checkpoints/<agent_tag>/<mode>_<reward>/` with automatic fallback to legacy paths.
- **Interactive Live Switching**: Switch agent architectures on the fly during gameplay using **`[H]`** or **`[A]`**.
- **Intelligent Auto-Detection**: Inspects checkpoint filenames, directory paths, and serialized metadata (`beta`, `mode`) to automatically instantiate the matching agent class.
- **Evaluation Mode (Pure Greedy)**: Fixes exploration rate $\epsilon = 0$, deterministically executing the maximal $Q(s, a)$ policy with uniform random tie-breaking.
- **State-Discretizer Integration**: Employs `MarioStateDiscretizer` (1,152 states) to convert continuous physics into hashable discrete states.

---

## 2. Choosing Agent Architecture & Hybrid Mode

You can choose between `TabularQAgent` and `HybridQSARSAAgent` through two methods:

### Method A: Command-Line Flags (`--agent` & `--hybrid-mode`)
When launching `play_agent.py`, specify the `--agent` parameter:
- **`--agent tabular`** (or `tabular_q`): Instantiates **`TabularQAgent`** (Pure Off-Policy Q-Learning).
- **`--agent hybrid`** (or `hybrid_agent`): Instantiates **`HybridQSARSAAgent`** (Unified SARSA + Q-Learning).
- **`--agent auto`** (default): Automatically detects whether the checkpoint is hybrid or tabular based on metadata and naming.

When running with hybrid agents, specify the target calculation mode:
- **`--hybrid-mode expected`** (default): Expected SARSA target blending off-policy $\max Q$ and policy expectation $\sum_b \pi(b|s') Q(s', b)$.
- **`--hybrid-mode linear`** (or $\beta$-blending): Blends off-policy $\max Q$ with on-policy $Q(s', a')$ using blending factor $\beta$.
- **`--hybrid-mode hazard_gated`**: Dynamically activates SARSA on-policy targets only in hazard zones (close pits or ghosts), using Q-learning elsewhere.

### Method B: Live In-Game Hotkey (`[H]` / `[A]`)
Press **`[H]`** or **`[A]`** at any time while the game is running to instantly switch between `TabularQAgent` and `HybridQSARSAAgent`.
- The agent policy is automatically reloaded from the corresponding agent tag checkpoint directory.
- The episode resets smoothly to test the new architecture from the starting position.
- The window caption updates immediately (`Agent: TABULAR` <-> `Agent: HYBRID (expected)`).

---

## 3. Interactive Window Controls

While the Pygame game window is active, you can interact with the visualizer in real-time:

| Key | Action | Behavior |
| :--- | :--- | :--- |
| **`[H / A]`** | **Toggle Agent Architecture** | Swaps live between **`TabularQAgent`** and **`HybridQSARSAAgent`**, reloading matching checkpoints and resetting the episode. |
| **`[1 / 2 / 3 / 4 / 5]`** | **Switch Level** | Dynamically changes the active graveyard level (1 to 5) without restarting the script. |
| **`[F1 / F2 / F3]`** | **Swap Reward Strategy** | Hot-swaps Q-table policy between `balanced` (`F1`), `sparse` (`F2`), and `dense` (`F3`). |
| **`[T]`** | **Toggle Variant** | Switches between `best` (highest evaluation score) and `final` (latest training episode) checkpoints. |
| **`[Space]`** | **Pause / Resume** | Pauses simulation and physics steps; updates window title with `[PAUSED]`. |
| **`[R]`** | **Restart Episode** | Resets Hoá, items, and all ghosts/hazards to their initial starting positions. |
| **`[Esc]`** | **Quit** | Closes the Pygame display window and terminates cleanly. |
| **`[X]` (Window Close)** | **Exit** | Standard OS window close button handles cleanup cleanly. |

---

## 4. Command-Line Interface (CLI)

You can launch `play_agent.py` directly from your terminal using Python.

### Syntax
```bash
python play_agent.py [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`--agent`** | `str` | `"auto"` | Agent architecture to instantiate (`tabular`, `hybrid`, or `auto`). |
| **`--hybrid-mode`**| `str` | `"expected"` | Target formulation for hybrid agent (`expected`, `linear`, or `hazard_gated`). |
| **`--level`** | `int` | `1` | Initial graveyard level to load (`1` to `5`). |
| **`--mode`** | `str` | `"curriculum"` | Training mode checkpoint hierarchy: `curriculum` or `separate`. |
| **`--reward`** | `str` | `"balanced"` | Reward strategy to load (`balanced`, `sparse`, or `dense`). |
| **`--variant`** | `str` | `"best"` | Model variant to load (`best` or `final`). |
| **`--checkpoint`** | `str` | `None` | Explicit path override to a `.json` or `.pkl` checkpoint file. |
| **`--fps`** | `int` | `60` | Target display frame rate limit (ticks per second). |

---

## 5. Usage Examples

### 5.1 Run with TabularQAgent Explicitly
```bash
python play_agent.py --agent tabular --level 1
```

### 5.2 Run with HybridQSARSAAgent (Expected SARSA Target)
```bash
python play_agent.py --agent hybrid --hybrid-mode expected --level 1
```

### 5.3 Run with Hazard-Gated Hybrid Agent on Level 4
```bash
python play_agent.py --agent hybrid --hybrid-mode hazard_gated --level 4
```

### 5.4 Spectating Level 4 with Separate Balanced Policy
```bash
python play_agent.py --mode separate --reward balanced --level 4
```

### 5.5 Spectating Level 5 with Explicit Checkpoint
```bash
python play_agent.py --checkpoint checkpoints/tabular/curriculum_balanced/q_table_best.json --level 5
```

### 5.6 Slow-Motion Analysis (20 FPS)
Useful for inspecting jump takeoff margins and obstacle clearance:
```bash
python play_agent.py --level 4 --fps 20
```

### 5.7 Running with an Uninitialized Policy (Random Walk Baseline)
If the specified checkpoint file does not exist or has 0 states, `play_agent.py` alerts the user and runs with uniform random tie-breaking:
```bash
python play_agent.py --checkpoint nonexistent.json --level 2
```

---

## 6. Checkpoint Directory Layout

Checkpoints are organized cleanly by agent tag, training mode, and reward profile:

```
checkpoints/
├── tabular/
│   ├── curriculum_balanced/
│   │   ├── q_table_lvl1_best.{pkl,json}
│   │   ├── q_table_lvl2_best.{pkl,json}
│   │   ├── q_table_lvl3_best.{pkl,json}
│   │   ├── q_table_lvl4_best.{pkl,json}
│   │   ├── q_table_lvl5_best.{pkl,json}
│   │   ├── q_table_best.{pkl,json}
│   │   └── q_table_final.{pkl,json}
│   └── separate_balanced/
│       ├── q_table_lvl1_best.{pkl,json}
│       ├── q_table_lvl1_final.{pkl,json}
│       └── ...
└── hybrid_expected/
    ├── curriculum_balanced/
    │   ├── q_table_best.{pkl,json}
    │   └── ...
    └── separate_balanced/
        └── ...
```

Canonical production runs (`tabular` + `curriculum` + `balanced`) also mirror `q_table_best.{pkl,json}` and `q_table_final.{pkl,json}` to the root `checkpoints/` directory for zero-configuration playback.

---

## 7. Programmatic API

You can also invoke the autonomous playback loop directly within Python scripts:

```python
from play_agent import play_autonomous

# 1. Play Level 4 using TabularQAgent
play_autonomous(
    level_id=4,
    agent_type="tabular",
    mode="separate",
    reward="balanced",
    variant="best",
    fps=60,
)

# 2. Play Level 5 using HybridQSARSAAgent
play_autonomous(
    level_id=5,
    agent_type="hybrid",
    hybrid_mode="hazard_gated",
    mode="curriculum",
    reward="balanced",
    variant="best",
    fps=60,
)
```

---

## 8. Supported Checkpoint Formats

Both agents support loading from either serialization format:

| Format | Extension | Characteristics | Best For |
| :--- | :---: | :--- | :--- |
| **JSON** | `.json` | Human-readable, portable across Python versions. State keys are serialized as comma-delimited strings (e.g. `"1,0,2,0,1,0,0"`). | Inspection, debugging, cross-environment evaluation. |
| **Pickle** | `.pkl` | Native Python binary format preserving NumPy `float32` arrays directly. Fast serialization. | Local checkpointing, exact precision recovery. |

---

## 9. Troubleshooting

### Issue: Warning: Loaded checkpoint but Q-table has 0 states
- **Cause**: The checkpoint file exists on disk but was saved empty or serialized before learning occurred.
- **Solution**: Check that the training run completed and populated the Q-table, or train the agent using `train.py`.

### Issue: Agent stands still or falls immediately
- **Cause**: The checkpoint file path was not found or was uninitialized.
- **Solution**: Verify that training has completed and check that the checkpoint path exists under `checkpoints/<agent_tag>/<mode>_<reward>/`.
