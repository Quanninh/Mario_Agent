from pathlib import Path
import random
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_env import Action
from core.interfaces import BaseAgent
from .dqn_network import DuelingDQN
from .replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Dueling Double Deep Q-Network (D3QN) Agent.

    Features:
      - Dueling network architecture separating V(s) and A(s, a)
      - Double DQN target calculation to prevent value overestimation
      - Experience replay buffer for decorrelated i.i.d. gradient updates
      - Polyak soft target network tracking: theta_target = tau * theta + (1 - tau) * theta_target
      - Huber Loss (SmoothL1Loss) for robust gradients against outlier penalties
      - Fully compatible with BaseAgent interface
    """

    def __init__(
        self,
        state_dim: int = 32,
        num_actions: int = Action.NUM_ACTIONS,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        warmup_steps: int = 500,
        training: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.training = training
        self.step_count = 0

        # Auto-detect device (CUDA -> MPS -> CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Policy and Target Networks
        self.policy_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def _ensure_tensor_state(self, state: Any) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.to(self.device).float()
        if isinstance(state, (list, tuple)):
            state = np.array(state, dtype=np.float32)
        return torch.as_tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state: Any) -> int:
        """
        Selects an action using epsilon-greedy exploration in training mode,
        or greedy argmax evaluation in testing mode.
        """
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = self._ensure_tensor_state(state)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """
        Records transition in replay buffer and performs a Double DQN update step.
        Returns the scalar loss of the update (or 0.0 during warmup).
        """
        s_arr = np.array(state, dtype=np.float32) if not isinstance(state, np.ndarray) else state
        ns_arr = np.array(next_state, dtype=np.float32) if not isinstance(next_state, np.ndarray) else next_state

        self.replay_buffer.push(s_arr, action, reward, ns_arr, done)
        self.step_count += 1

        if not self.training or len(self.replay_buffer) < self.warmup_steps:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Double DQN Target:
        # a* = argmax_a Q(s', a; theta_policy)
        # Target = r + (1 - d) * gamma * Q(s', a*; theta_target)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        current_q = self.policy_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak Soft Target Network Update
        with torch.no_grad():
            for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def set_mode(self, training: bool) -> None:
        self.training = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "tau": self.tau,
            "step_count": self.step_count,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }
        torch.save(checkpoint, path)

    def load(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.step_count = checkpoint.get("step_count", 0)
