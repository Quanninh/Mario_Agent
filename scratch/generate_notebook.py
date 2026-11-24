import json
from pathlib import Path


def create_dqn_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # -------------------------------------------------------------
    # CELL 1: Header & Introduction
    # -------------------------------------------------------------
    add_markdown("""
# 👻 Hoá's Midnight Escape: Dueling Double Deep Q-Network (D3QN)
### Complete End-to-End Reinforcement Learning on Google Colab & Kaggle

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Quanninh/Mario_Agent/blob/DQN/notebooks/mario_dqn_colab_kaggle.ipynb)

This notebook implements and trains a **Dueling Double Deep Q-Network (D3QN)** to master all 8 progressive platformer levels of **Hoá's Midnight Escape**.

---

### Key Technical Components:
1. **Continuous Vector Feature Extractor**: Replaces the discrete tabular binning with a normalized 22-dimensional continuous state representation.
2. **Dueling Network Architecture**: Decomposes $Q(s, a)$ into a state-value stream $V(s)$ and advantage stream $A(s, a)$ to stabilize value estimation.
3. **Double Q-Target Optimization**: Decouples action selection from action evaluation to eliminate positive maximization bias.
4. **Experience Replay Buffer**: Breaks temporal correlation with uniform i.i.d. batch sampling.
5. **Polyak Soft Target Updates ($\tau = 0.005$)**: Ensures smooth target network tracking.
6. **Curriculum Transfer (Levels 1 $\\to$ 8)**: Incrementally trains the neural policy through progressive graveyard challenges, culminating in **Level 8: The Celestial Purgatory** (7,150 px marathon).
7. **In-Notebook Gameplay GIF Recorder**: Off-screen Pygame frame capture to watch your trained agent play directly inside the notebook.
""")

    # -------------------------------------------------------------
    # CELL 2: Environment Setup & Clone
    # -------------------------------------------------------------
    add_markdown("""
## 1. Environment Setup & Dependency Installation
Configure headless Pygame (using the `"dummy"` video driver for Linux servers) and import/clone the environment repository.
""")

    add_code("""
import os
import sys
from pathlib import Path

# 1. Configure headless Pygame driver (crucial for Colab / Kaggle without X11 display)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 2. Detect runtime environment (Google Colab vs. Kaggle vs. Local)
is_colab = "google.colab" in sys.modules
is_kaggle = os.path.exists("/kaggle/working")

print(f"Runtime Environment: Colab={is_colab}, Kaggle={is_kaggle}, Local={not is_colab and not is_kaggle}")

# 3. Clone repository if not present locally
repo_dir = Path("Mario_Agent")
if not Path("env").exists() and not (repo_dir / "env").exists():
    print("[INFO] Cloning repository from GitHub...")
    !git clone https://github.com/Quanninh/Mario_Agent.git
    sys.path.append(str(Path("Mario_Agent").resolve()))
elif (repo_dir / "env").exists():
    sys.path.append(str(repo_dir.resolve()))
else:
    sys.path.append(str(Path(".").resolve()))

# 4. Install required packages
!pip install -q pygame torch matplotlib imageio pandas tqdm

import pygame
pygame.init()
print("[OK] Pygame initialized in headless mode successfully!")
""")

    # -------------------------------------------------------------
    # CELL 3: Hardware Verification
    # -------------------------------------------------------------
    add_markdown("""
## 2. Hardware Acceleration Check
Detect and configure CUDA GPU, Apple Silicon MPS, or CPU.
""")

    add_code("""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[ACCELERATION] GPU Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[ACCELERATION] Apple Silicon MPS Detected!")
else:
    device = torch.device("cpu")
    print("[ACCELERATION] Running on CPU.")

print(f"PyTorch Version: {torch.__version__} | Active Device: {device}")
""")

    # -------------------------------------------------------------
    # CELL 4: Continuous State Extractor
    # -------------------------------------------------------------
    add_markdown("""
## 3. Continuous Feature State Extractor (22-Dimensional Vector)
Instead of coarse tabular discrete bins (which suffer from perceptual aliasing and state space explosion), we extract a continuous normalized 22-dimensional feature vector $\\mathbf{s} \\in \\mathbb{R}^{22}$.
""")

    add_code("""
from typing import Optional
import numpy as np
from core.base_env import RawState, Action

class ContinuousStateExtractor:
    \"\"\"
    Extracts a normalized, stationary continuous feature vector from RawState.
    Feature vector dimension: 22
    \"\"\"
    STATE_DIM: int = 22

    def __init__(
        self,
        canvas_height: float = 400.0,
        max_threat_distance: float = 200.0,
        max_level_length: float = 8000.0,
    ) -> None:
        self.canvas_height = canvas_height
        self.max_threat_distance = max_threat_distance
        self.max_level_length = max_level_length

    def extract(self, raw: RawState) -> np.ndarray:
        player_y = getattr(raw, "player_y", 288.0)
        vy = getattr(raw, "velocity_y", 0.0)
        vx = getattr(raw, "velocity_x", 0.0)

        # 1. Kinematics & Posture (6 features)
        norm_y = np.clip(player_y / self.canvas_height, 0.0, 1.0)
        norm_vx = np.clip(vx / 7.0, -1.0, 1.0)
        norm_vy = np.clip(vy / 15.0, -1.0, 1.0)
        grounded = 1.0 if getattr(raw, "is_grounded", False) else 0.0
        double_jump = 1.0 if getattr(raw, "can_double_jump", False) else 0.0
        crouch = 1.0 if getattr(raw, "is_crouching", False) else 0.0

        # 2. Tombstone Obstacles (2 features)
        dist_obs = getattr(raw, "dist_to_obstacle", -1.0)
        norm_dist_obs = (
            np.clip(dist_obs / self.max_threat_distance, 0.0, 1.0)
            if dist_obs >= 0.0
            else 1.0
        )
        norm_height_obs = np.clip(getattr(raw, "obstacle_height", 0.0) / 100.0, 0.0, 1.0)

        # 3. Grave Pits & Fire Hazard (2 features)
        dist_pit = getattr(raw, "dist_to_pit", -1.0)
        norm_dist_pit = (
            np.clip(dist_pit / self.max_threat_distance, 0.0, 1.0)
            if dist_pit >= 0.0
            else 1.0
        )
        norm_fire_state = np.clip(getattr(raw, "pit_fire_state", 0) / 2.0, 0.0, 1.0)

        # 4. Ghost Proximity & Kinematics (3 features)
        dist_ghost = getattr(raw, "dist_to_ghost", -1.0)
        norm_dist_ghost = (
            np.clip(dist_ghost / self.max_threat_distance, 0.0, 1.0)
            if dist_ghost >= 0.0
            else 1.0
        )
        ghost_y = getattr(raw, "ghost_y", 0.0)
        rel_ghost_y = (
            np.clip((ghost_y - player_y) / self.max_threat_distance, -1.0, 1.0)
            if dist_ghost >= 0.0 and ghost_y > 0.0
            else 0.0
        )
        norm_ghost_vx = np.clip(getattr(raw, "ghost_vx", 0.0) / 5.0, -1.0, 1.0)

        # 5. Rotating Laser Threat (2 features)
        dist_laser = getattr(raw, "dist_to_laser", -1.0)
        norm_dist_laser = (
            np.clip(dist_laser / self.max_threat_distance, 0.0, 1.0)
            if dist_laser >= 0.0
            else 1.0
        )
        norm_laser_state = np.clip(getattr(raw, "laser_state", 0) / 2.0, 0.0, 1.0)

        # 6. Will-o'-the-Wisp Projectile (2 features)
        dist_wisp = getattr(raw, "dist_to_wisp", -1.0)
        norm_dist_wisp = (
            np.clip(dist_wisp / self.max_threat_distance, 0.0, 1.0)
            if dist_wisp >= 0.0
            else 1.0
        )
        norm_wisp_dir = float(np.clip(getattr(raw, "wisp_dir", 0), -1, 1))

        # 7. Grave Monsters (Quỷ Nhập Tràng & Cương Thi) (3 features)
        dist_monster = getattr(raw, "dist_to_monster", -1.0)
        norm_dist_monster = (
            np.clip(dist_monster / self.max_threat_distance, 0.0, 1.0)
            if dist_monster >= 0.0
            else 1.0
        )
        monster_y = getattr(raw, "monster_y", 0.0)
        rel_monster_y = (
            np.clip((monster_y - player_y) / self.max_threat_distance, -1.0, 1.0)
            if dist_monster >= 0.0 and monster_y > 0.0
            else 0.0
        )
        monster_active = 1.0 if dist_monster >= 0.0 else 0.0

        # 8. Warmth Garlic Item & Level Goal (2 features)
        dist_item = getattr(raw, "dist_to_item", -1.0)
        norm_dist_item = (
            np.clip(dist_item / self.max_threat_distance, 0.0, 1.0)
            if dist_item >= 0.0
            else 1.0
        )
        dist_goal = getattr(raw, "dist_to_goal", self.max_level_length)
        norm_dist_goal = np.clip(dist_goal / self.max_level_length, 0.0, 1.0)

        return np.array(
            [
                norm_y, norm_vx, norm_vy, grounded, double_jump, crouch,
                norm_dist_obs, norm_height_obs, norm_dist_pit, norm_fire_state,
                norm_dist_ghost, rel_ghost_y, norm_ghost_vx, norm_dist_laser,
                norm_laser_state, norm_dist_wisp, norm_wisp_dir, norm_dist_monster,
                rel_monster_y, monster_active, norm_dist_item, norm_dist_goal,
            ],
            dtype=np.float32,
        )

print(f"[OK] ContinuousStateExtractor initialized. State dimension: {ContinuousStateExtractor.STATE_DIM}")
""")

    # -------------------------------------------------------------
    # CELL 5: Dueling DQN Neural Network Architecture
    # -------------------------------------------------------------
    add_markdown("""
## 4. Dueling Double Deep Q-Network (D3QN) Architecture
The network decouples into a state-value stream $V(s)$ and an advantage stream $A(s, a)$:

$$Q(s, a) = V(s) + \\left( A(s, a) - \\frac{1}{|\\mathcal{A}|} \\sum_{a' \\in \\mathcal{A}} A(s, a') \\right)$$
""")

    add_code("""
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    \"\"\"
    Dueling Deep Q-Network with Layer Normalization and Separated V(s) / A(s, a) Streams.
    \"\"\"
    def __init__(self, state_dim: int = 22, num_actions: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared feature trunk
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Aggregation with zero-mean advantage centering
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))

sample_net = DuelingDQN().to(device)
dummy_input = torch.randn(2, 22).to(device)
out = sample_net(dummy_input)
print(f"[OK] DuelingDQN built successfully! Test output shape: {out.shape}")
""")

    # -------------------------------------------------------------
    # CELL 6: Replay Buffer & DQNAgent Class
    # -------------------------------------------------------------
    add_markdown("""
## 5. Replay Buffer & DQNAgent
Includes Double Q-learning target computation, soft Polyak target update, and Huber Loss.
""")

    add_code("""
from collections import deque
import random
from typing import Tuple
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(actions), dtype=torch.int64, device=device).unsqueeze(1),
            torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 22,
        num_actions: int = 6,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        device: torch.device = device,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device = device
        self.training = True

        self.policy_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        self.replay_buffer.push(state, action, reward, next_state, done)

        if not self.training or len(self.replay_buffer) < self.warmup_steps:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        # Double DQN: Policy net selects best action, Target net evaluates value
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        current_q = self.policy_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak soft update
        with torch.no_grad():
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)

print("[OK] ReplayBuffer and DQNAgent defined successfully!")
""")

    # -------------------------------------------------------------
    # CELL 7: Curriculum Training Function
    # -------------------------------------------------------------
    add_markdown("""
## 6. Curriculum Training Across Levels 1 to 8
Now we train the DQN agent. The curriculum trains through Level 1 to Level 8 progressively.
Checkpoint weights are inherited between stages, enabling smooth skill transfer.
""")

    add_code("""
import time
import pandas as pd
from env import create_env
from rewards.balanced_reward import BalancedRewardStrategy

def train_curriculum(
    levels=[1, 2, 3, 4, 5, 6, 7, 8],
    episodes_per_level=150,
    lr=3e-4,
    batch_size=64,
    frame_skip=1,
    checkpoint_dir="checkpoints/dqn_colab",
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extractor = ContinuousStateExtractor()
    reward_fn = BalancedRewardStrategy()
    agent = DQNAgent(
        state_dim=ContinuousStateExtractor.STATE_DIM,
        num_actions=6,
        learning_rate=lr,
        batch_size=batch_size,
        device=device,
    )

    history = []
    global_ep = 0
    total_episodes = len(levels) * episodes_per_level
    start_time = time.time()

    print(f"================== STARTING DQN CURRICULUM ({len(levels)} Levels) ==================")

    for lvl in levels:
        env = create_env(level_id=lvl)
        lvl_wins = []
        best_win_rate = 0.0

        # Maintain minimum exploration when entering harder levels
        if lvl > levels[0] and agent.epsilon < 0.20:
            agent.epsilon = 0.20

        print(f"\\n---> [LEVEL {lvl}] Starting {episodes_per_level} episodes (eps={agent.epsilon:.2f})")

        for ep in range(1, episodes_per_level + 1):
            global_ep += 1
            raw_state = env.reset()
            state_vec = extractor.extract(raw_state)
            ep_reward = 0.0
            ep_loss = 0.0
            steps = 0
            loss_count = 0
            done = False

            while not done:
                action = agent.select_action(state_vec)

                # Frame skipping
                skip_reward = 0.0
                for _ in range(frame_skip):
                    next_raw, events, done = env.step(action)
                    step_r = reward_fn.calculate_reward(raw_state, action, next_raw, events, done)
                    skip_reward += step_r
                    raw_state = next_raw
                    steps += 1
                    if done:
                        break

                next_vec = extractor.extract(next_raw)
                loss = agent.update(state_vec, action, skip_reward, next_vec, done)
                if loss > 0.0:
                    ep_loss += loss
                    loss_count += 1

                ep_reward += skip_reward
                state_vec = next_vec

            agent.decay_epsilon()
            win = 1 if getattr(events, "reached_goal", False) else 0
            lvl_wins.append(win)
            rolling_50 = float(np.mean(lvl_wins[-50:])) * 100.0
            avg_loss = ep_loss / max(1, loss_count)

            history.append({
                "global_episode": global_ep,
                "level_id": lvl,
                "level_episode": ep,
                "reward": ep_reward,
                "steps": steps,
                "win": win,
                "rolling_win_rate_50": rolling_50,
                "loss": avg_loss,
                "epsilon": agent.epsilon,
            })

            if len(lvl_wins) >= 20 and rolling_50 > best_win_rate:
                best_win_rate = rolling_50
                agent.save(str(ckpt_dir / f"dqn_level_{lvl}_best.pt"))

            if ep == 1 or ep % 25 == 0 or ep == episodes_per_level:
                elapsed = time.time() - start_time
                print(
                    f"[{global_ep:4d}/{total_episodes:4d}] Lvl {lvl} (Ep {ep:3d}/{episodes_per_level:3d}) | "
                    f"Win50: {rolling_50:5.1f}% | R: {ep_reward:+7.1f} | Loss: {avg_loss:.4f} | "
                    f"eps: {agent.epsilon:.3f} | Elapsed: {int(elapsed)}s"
                )

        env.close()

    agent.save(str(ckpt_dir / "dqn_final.pt"))
    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(ckpt_dir / "training_metrics.csv", index=False)
    print(f"\\n[COMPLETE] Training finished! Model and metrics saved to {ckpt_dir}")
    return agent, df_metrics
""")

    # -------------------------------------------------------------
    # CELL 8: Run Training
    # -------------------------------------------------------------
    add_markdown("""
## 7. Run DQN Training
*Tip*: In Colab / Kaggle, training Levels 1–4 takes ~2–3 minutes on GPU. To train all 8 levels, set `levels=[1, 2, 3, 4, 5, 6, 7, 8]` and `episodes_per_level=150`.
""")

    add_code("""
# Configure and run training (Adjust levels and episodes as needed)
TRAIN_LEVELS = [1, 2, 3, 4]  # Expand to [1, 2, 3, 4, 5, 6, 7, 8] for the full marathon!
EPISODES_PER_LEVEL = 100

trained_agent, df_metrics = train_curriculum(
    levels=TRAIN_LEVELS,
    episodes_per_level=EPISODES_PER_LEVEL,
    lr=3e-4,
    batch_size=64,
    frame_skip=1,
)
""")

    # -------------------------------------------------------------
    # CELL 9: Plot Training Curves
    # -------------------------------------------------------------
    add_markdown("""
## 8. Training Telemetry & Learning Curves
Plot the training metrics: Episodic Return, Rolling Win Rate %, TD Loss, and Steps per Episode.
""")

    add_code("""
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Episodic Reward
axes[0, 0].plot(df_metrics["global_episode"], df_metrics["reward"], alpha=0.3, color="royalblue", label="Raw Reward")
axes[0, 0].plot(df_metrics["global_episode"], df_metrics["reward"].rolling(20, min_periods=1).mean(), color="navy", lw=2, label="Moving Avg (20)")
axes[0, 0].set_title("Episodic Return", fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("Global Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].legend()

# 2. Rolling Win Rate %
axes[0, 1].plot(df_metrics["global_episode"], df_metrics["rolling_win_rate_50"], color="forestgreen", lw=2.5)
axes[0, 1].set_title("50-Episode Rolling Win Rate (%)", fontsize=13, fontweight="bold")
axes[0, 1].set_xlabel("Global Episode")
axes[0, 1].set_ylabel("Win Rate (%)")
axes[0, 1].set_ylim(-5, 105)

# 3. TD Loss
axes[1, 0].plot(df_metrics["global_episode"], df_metrics["loss"], color="crimson", alpha=0.7)
axes[1, 0].set_title("Temporal Difference Huber Loss", fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Global Episode")
axes[1, 0].set_ylabel("Loss")

# 4. Steps per Episode
axes[1, 1].plot(df_metrics["global_episode"], df_metrics["steps"], color="darkorange", alpha=0.6)
axes[1, 1].set_title("Steps per Episode", fontsize=13, fontweight="bold")
axes[1, 1].set_xlabel("Global Episode")
axes[1, 1].set_ylabel("Steps")

plt.tight_layout()
plt.savefig("checkpoints/dqn_colab/learning_curves.png", dpi=150)
plt.show()
print("[OK] Plots generated and saved to checkpoints/dqn_colab/learning_curves.png")
""")

    # -------------------------------------------------------------
    # CELL 10: Greedy Evaluation Matrix
    # -------------------------------------------------------------
    add_markdown("""
## 9. Greedy Evaluation Benchmark across Levels
Evaluate the trained DQN policy under pure exploitation ($\\epsilon = 0.0$) across evaluated levels.
""")

    add_code("""
def evaluate_agent(agent, levels, num_trials=10):
    agent.training = False
    agent.epsilon = 0.0
    extractor = ContinuousStateExtractor()
    reward_fn = BalancedRewardStrategy()
    eval_results = []

    print(f"================ EVALUATION MATRIX (Epsilon = 0.0) ================")
    print(f"{'Level':<8} {'Win Rate':<12} {'Avg Reward':<14} {'Avg Steps':<12} {'Status'}")
    print("-" * 55)

    for lvl in levels:
        env = create_env(level_id=lvl)
        wins = 0
        rewards = []
        steps_list = []

        for _ in range(num_trials):
            raw_state = env.reset()
            state_vec = extractor.extract(raw_state)
            tot_r = 0.0
            steps = 0
            done = False

            while not done:
                action = agent.select_action(state_vec)
                next_raw, events, done = env.step(action)
                r = reward_fn.calculate_reward(raw_state, action, next_raw, events, done)
                tot_r += r
                steps += 1
                state_vec = extractor.extract(next_raw)
                raw_state = next_raw

            if getattr(events, "reached_goal", False):
                wins += 1
            rewards.append(tot_r)
            steps_list.append(steps)

        env.close()
        win_pct = (wins / num_trials) * 100.0
        avg_r = np.mean(rewards)
        avg_s = np.mean(steps_list)
        status = "PASSED" if win_pct >= 80.0 else "IN PROGRESS"

        eval_results.append({
            "Level": lvl,
            "Win_Rate_Pct": win_pct,
            "Avg_Reward": avg_r,
            "Avg_Steps": avg_s,
            "Status": status,
        })
        print(f"Level {lvl:<3} {win_pct:6.1f}%       {avg_r:+8.1f}       {avg_s:6.1f}       {status}")

    df_eval = pd.DataFrame(eval_results)
    return df_eval

df_eval = evaluate_agent(trained_agent, levels=TRAIN_LEVELS, num_trials=10)
""")

    # -------------------------------------------------------------
    # CELL 11: Gameplay Video / Animated GIF Generator
    # -------------------------------------------------------------
    add_markdown("""
## 10. Watch the Agent Play (Animated GIF Generator)
Since headless cloud environments have no active window display, we render Pygame surfaces directly to an off-screen buffer and save an animated GIF to watch inside the notebook!
""")

    add_code("""
import imageio
from IPython.display import Image, display

def record_gameplay_gif(agent, level_id=1, output_path="gameplay.gif", max_frames=600):
    \"\"\"
    Simulates one episode of the trained agent and renders an animated GIF.
    \"\"\"
    env = create_env(level_id=level_id)
    extractor = ContinuousStateExtractor()
    frames = []

    raw_state = env.reset()
    state_vec = extractor.extract(raw_state)
    done = False
    frame_count = 0

    agent.training = False
    agent.epsilon = 0.0

    print(f"[RECORDING] Capturing Level {level_id} gameplay frames...")

    while not done and frame_count < max_frames:
        action = agent.select_action(state_vec)
        next_raw, events, done = env.step(action)
        state_vec = extractor.extract(next_raw)

        # Render off-screen
        env.render()
        if env._renderer is not None and env._renderer._screen is not None:
            # Capture RGB surface array from Pygame
            surf = env._renderer._screen
            img_data = pygame.surfarray.array3d(surf)
            # Transpose axes from (W, H, C) to (H, W, C)
            img_data = np.transpose(img_data, (1, 0, 2))
            frames.append(img_data)

        frame_count += 1

    env.close()

    # Downsample slightly for fast rendering and small GIF size
    step_skip = 2
    sampled_frames = frames[::step_skip]
    imageio.mimsave(output_path, sampled_frames, fps=25)
    print(f"[OK] Gameplay GIF saved to {output_path} ({len(sampled_frames)} frames)")
    return output_path

gif_path = record_gameplay_gif(trained_agent, level_id=1, output_path="gameplay_level_1.gif")
display(Image(filename=gif_path))
""")

    # -------------------------------------------------------------
    # CELL 12: Checkpoint Export / Download
    # -------------------------------------------------------------
    add_markdown("""
## 11. Download Trained Models & Artifacts
Compress checkpoints and metrics into a zip archive and download to your machine.
""")

    add_code("""
import shutil

# Create a zip bundle of trained models and plots
shutil.make_archive("mario_dqn_checkpoints", "zip", "checkpoints/dqn_colab")
print("[OK] Created mario_dqn_checkpoints.zip")

# Trigger automatic browser download if in Google Colab
if is_colab:
    try:
        from google.colab import files
        files.download("mario_dqn_checkpoints.zip")
        print("[INFO] Download triggered in Colab!")
    except Exception as e:
        print(f"[NOTE] Manual download available: {e}")
elif is_kaggle:
    print("[INFO] Checkpoints available in Kaggle Output directory: /kaggle/working/mario_dqn_checkpoints.zip")
""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path("notebooks/mario_dqn_colab_kaggle.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    # Also place at root for single-click drag-and-drop
    root_path = Path("mario_dqn_colab_kaggle.ipynb")
    with open(root_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[OK] Notebook created at {out_path} and {root_path}")


if __name__ == "__main__":
    create_dqn_notebook()
