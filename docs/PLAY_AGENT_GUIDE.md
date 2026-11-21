# Autonomous Agent Playback Guide (`play_agent.py`)

This guide explains how to execute, configure, and spectate trained reinforcement learning agents in **Hoá's Midnight Escape** using `play_agent.py`.

---

## 1. Overview & Purpose

`play_agent.py` provides a real-time, 60 FPS interactive Pygame visualizer that renders Hoá navigating through graveyard levels under the control of a learned policy.

### Key Capabilities
- **Dual-Architecture Choice**: Choose between **Pure Tabular Q-Learning (`TabularQAgent`)** and **Hybrid SARSA/Q-Learning (`HybridAgent` / `HybridQSARSAAgent`)**.
- **Interactive Live Switching**: Switch agent architectures on the fly during gameplay using **`[H]`** or **`[A]`**.
- **Intelligent Auto-Detection**: Inspects checkpoint filenames and serialized metadata (`beta`, `mode`) to automatically instantiate the matching agent class.
- **Evaluation Mode (Pure Greedy)**: Fixes exploration rate $\epsilon = 0$, deterministically executing the maximal $Q(s, a)$ policy with uniform random tie-breaking.
- **State-Discretizer Integration**: Employs `MarioStateDiscretizer` to convert continuous physics into hashable discrete states.

---

## 2. Choosing Agent Architecture

You can choose between `TabularQAgent` and `HybridAgent` through two methods:

### Method A: Command-Line Flag (`--agent`)
When launching `play_agent.py`, specify the `--agent` parameter:
- **`--agent tabular`** (or `tabular_q`): Instantiates **`TabularQAgent`** (Pure Off-Policy Q-Learning).
- **`--agent hybrid`** (or `hybrid_agent`): Instantiates **`HybridAgent`** / `HybridQSARSAAgent` (Unified SARSA + Q-Learning).
- **`--agent auto`** (default): Automatically detects whether the checkpoint is hybrid or tabular based on metadata and naming.

### Method B: Live In-Game Hotkey (`[H]` / `[A]`)
Press **`[H]`** or **`[A]`** at any time while the game is running to instantly switch between `TabularQAgent` and `HybridAgent`.
- The agent policy is automatically reloaded.
- The episode resets smoothly to test the new architecture from the starting position.
- The window caption updates immediately (`Agent: TABULAR` <-> `Agent: HYBRID`).

---

## 3. Interactive Window Controls

While the Pygame game window is active, you can interact with the visualizer in real-time:

| Key | Action | Behavior |
| :--- | :--- | :--- |
| **`[H / A]`** | **Toggle Agent Architecture** | Swaps live between **`TabularQAgent`** and **`HybridAgent`**, resetting the episode with the active checkpoint. |
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

### 5.2 Run with HybridAgent Explicitly
```bash
python play_agent.py --agent hybrid --level 1
```

### 5.3 Spectating Level 4 with Separate Balanced Policy
```bash
python play_agent.py --mode separate --reward balanced --level 4
```

### 5.4 Spectating Level 5
```bash
python play_agent.py --checkpoint checkpoints/q_table_level_5_best.json --level 5
```

### 5.5 High-Speed Playback (120 FPS)
```bash
python play_agent.py --level 5 --fps 120
```

### 5.6 Slow-Motion Analysis (20 FPS)
Useful for inspecting jump takeoff margins and obstacle clearance:
```bash
python play_agent.py --level 4 --fps 20
```

### 5.7 Running with an Uninitialized Policy (Random Walk Baseline)
If the specified checkpoint file does not exist, `play_agent.py` alerts the user and runs with uniform random tie-breaking:
```bash
python play_agent.py --checkpoint nonexistent.json --level 2
```

---

## 6. Programmatic API

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

# 2. Play Level 5 using HybridAgent
play_autonomous(
    checkpoint_path="checkpoints/hybrid_q_sarsa.pkl",
    level_id=5,
    agent_type="hybrid",
    fps=60,
)
```

---

## 7. Supported Checkpoint Formats

Both agents support loading from either serialization format:

| Format | Extension | Characteristics | Best For |
| :--- | :---: | :--- | :--- |
| **JSON** | `.json` | Human-readable, portable across Python versions. State keys are serialized as comma-delimited strings (e.g. `"1,0,2,0,1,0,0"`). | Inspection, debugging, cross-environment evaluation. |
| **Pickle** | `.pkl` | Native Python binary format preserving NumPy `float32` arrays directly. Fast serialization. | Local checkpointing, exact precision recovery. |

---

## 8. Troubleshooting

### Issue: Window freezes or shows spinning wheel on macOS
- **Cause**: Pygame event pump is starved when window events are unhandled.
- **Solution**: `play_agent.py` automatically processes `pygame.event.get()` every frame. Ensure the window is not minimized or suspended.

### Issue: Agent stands still or falls immediately
- **Cause**: The checkpoint file path was not found or was uninitialized.
- **Solution**: Verify the checkpoint path exists, or train the agent first using the training scripts.
