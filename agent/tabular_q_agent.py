import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import json

from core.base_env import Action
from core.interfaces import BaseAgent, DiscreteState


class TabularQAgent(BaseAgent):
    """
    Q-Learning Agent maintaining an action-value table Q: (s, a) -> float.
    """

    def __init__(
        self,
        num_actions: int = Action.NUM_ACTIONS,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        training: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.training = training

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Q-Table mapping DiscreteState tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[DiscreteState, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def select_action(self, state: DiscreteState) -> int:
        """
        Selects an action using epsilon-greedy exploration in training mode,
        or pure greedy exploitation in evaluation mode.
        Performs uniform random tie-breaking among maximal action values.
        """
        # Epsilon-exploration during training
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Greedy exploitation with tie-breaking
        q_values = self.q_table[state]
        max_val = np.max(q_values)
        candidates = np.flatnonzero(q_values == max_val)

        return int(np.random.choice(candidates))

    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        done: bool,
    ) -> float:
        """
        Applies Temporal Difference (TD) control update:
            TD Target = r                                (if terminal)
            TD Target = r + gamma * max_a' Q(s', a')     (if non-terminal)
            TD Error  = TD Target - Q(s, a)
            Q(s, a)  <- Q(s, a) + alpha * TD Error
        """
        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            next_q_values = self.q_table[next_state]
            td_target = reward + self.gamma * np.max(next_q_values)

        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decays epsilon by decay factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def set_mode(self, training: bool) -> None:
        """Toggles between training and evaluation mode."""
        self.training = training

    def save(self, filepath: str) -> None:
        """
        Saves the Q-table to disk.
        Supports both .pkl (binary) and .json (human-readable) formats.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            # JSON keys must be strings, so convert state tuples: (1, 0, 2, 0, 1) -> "1,0,2,0,1"
            serializable_q = {
                ",".join(map(str, state)): q_vals.tolist()
                for state, q_vals in self.q_table.items()
            }
            metadata = {
                "num_states_discovered": len(self.q_table),
                "num_actions": self.num_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "q_table": serializable_q,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        else:
            # Default to Pickle (.pkl)
            checkpoint = {
                "num_actions": self.num_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "q_table": dict(self.q_table),  # Convert defaultdict to normal dict
            }
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Agent] Saved Q-table ({len(self.q_table)} states) to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Loads the Q-table and configuration from disk into the agent.
        Supports both .pkl and .json formats.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # Re-initialize clean table
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for state_str, q_vals in data.get("q_table", {}).items():
                state_tuple = tuple(map(int, state_str.split(",")))
                self.q_table[state_tuple] = np.array(q_vals, dtype=np.float32)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for state, q_vals in data.get("q_table", {}).items():
                self.q_table[state] = np.array(q_vals, dtype=np.float32)

        # Restore hyperparameters if available in checkpoint
        for attr in ("num_actions", "alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay"):
            if attr in data:
                setattr(self, attr, data[attr])

        print(f"[Agent] Loaded Q-table ({len(self.q_table)} states) from {filepath}")