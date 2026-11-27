import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

from core.base_env import Action
from core.interfaces import BaseAgent, DiscreteState


class TabularQAgent(BaseAgent):
    """
    Q-Learning Agent maintaining an action-value table Q: (s, a) -> float.
    Supports optional Watkins's Q(lambda) replacing eligibility traces when trace_decay > 0.
    """

    def __init__(
        self,
        num_actions: int = Action.NUM_ACTIONS,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        trace_decay: float = 0.0,
        trace_threshold: float = 1e-3,
        training: bool = True,
        seed: Optional[int] = None,
    ):
        if not (0.0 <= trace_decay <= 1.0):
            raise ValueError(f"trace_decay must be in [0.0, 1.0], got {trace_decay}")
        if trace_threshold < 0.0:
            raise ValueError(f"trace_threshold must be >= 0.0, got {trace_threshold}")

        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.trace_decay = float(trace_decay)
        self.trace_threshold = float(trace_threshold)
        self.training = training

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Q-Table mapping DiscreteState tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[DiscreteState, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )
        self._traces: Dict[Tuple[DiscreteState, int], float] = {}
        self._last_action_exploratory: bool = False

    def reset_traces(self) -> None:
        """Clears eligibility traces and exploration state between episodes."""
        self._traces.clear()
        self._last_action_exploratory = False

    def select_action(self, state: DiscreteState) -> int:
        """
        Selects an action using epsilon-greedy exploration in training mode,
        or pure greedy exploitation in evaluation mode (without mutating q_table).
        Performs uniform random tie-breaking among maximal action values.
        """
        # Epsilon-exploration during training
        if self.training and random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            if self.trace_decay > 0.0:
                q_values = self.q_table.get(state)
                self._last_action_exploratory = (
                    q_values is not None and float(q_values[action]) < float(np.max(q_values))
                )
            else:
                self._last_action_exploratory = False
            return action

        self._last_action_exploratory = False

        # Read-only lookup so evaluation never inserts unseen states into q_table
        q_values = self.q_table.get(state)
        if q_values is None:
            return random.randrange(self.num_actions)

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
        Applies Temporal Difference (TD) control update (or Watkins's Q(lambda) if trace_decay > 0):
            TD Target = r                                (if terminal)
            TD Target = r + gamma * max_a' Q(s', a')     (if non-terminal)
            TD Error  = TD Target - Q(s, a)
            Q(s, a)  <- Q(s, a) + alpha * TD Error
        """
        is_exploratory = self._last_action_exploratory
        self._last_action_exploratory = False

        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            next_q_values = self.q_table.get(next_state)
            max_next_q = float(np.max(next_q_values)) if next_q_values is not None else 0.0
            td_target = reward + self.gamma * max_next_q

        td_error = td_target - current_q

        # Fast path: pure 1-step TD incurs zero trace overhead
        if self.trace_decay == 0.0:
            self.q_table[state][action] = current_q + self.alpha * td_error
            return float(td_error)

        # Watkins's Q(lambda): cut off-policy traces before applying suboptimal exploratory TD error
        if is_exploratory:
            self._traces.clear()

        self._traces[(state, action)] = 1.0
        alpha_delta = self.alpha * td_error
        for (tr_state, tr_action), eligibility in self._traces.items():
            self.q_table[tr_state][tr_action] += alpha_delta * eligibility

        if done:
            self._traces.clear()
        else:
            decay_factor = self.gamma * self.trace_decay
            threshold = self.trace_threshold
            next_traces: Dict[Tuple[DiscreteState, int], float] = {}
            for key, eligibility in self._traces.items():
                decayed = eligibility * decay_factor
                if decayed >= threshold:
                    next_traces[key] = decayed
            self._traces = next_traces

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
                "trace_decay": self.trace_decay,
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
                "trace_decay": self.trace_decay,
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
            loaded_actions = data.get("num_actions", self.num_actions)
            self.num_actions = max(self.num_actions, loaded_actions)
            for state_str, q_vals in data.get("q_table", {}).items():
                state_tuple = tuple(map(int, state_str.split(",")))
                vec = np.zeros(self.num_actions, dtype=np.float32)
                vec[:min(len(q_vals), self.num_actions)] = q_vals[:self.num_actions]
                self.q_table[state_tuple] = vec
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
            loaded_actions = data.get("num_actions", self.num_actions)
            self.num_actions = max(self.num_actions, loaded_actions)
            for state, q_vals in data.get("q_table", {}).items():
                vec = np.zeros(self.num_actions, dtype=np.float32)
                vec[:min(len(q_vals), self.num_actions)] = q_vals[:self.num_actions]
                self.q_table[state] = vec

        # Restore hyperparameters if available in checkpoint
        for attr in ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay"):
            if attr in data:
                setattr(self, attr, data[attr])
        self.trace_decay = float(data.get("trace_decay", 0.0))
        self.reset_traces()

        print(f"[Agent] Loaded Q-table ({len(self.q_table)} states) from {filepath}")

    # Aliases for backward compatibility
    load_q_table = load
    save_q_table = save
