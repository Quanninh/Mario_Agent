import json
from collections import defaultdict
from pathlib import Path
import pickle
import random
from typing import Dict, Optional, Tuple

import numpy as np

from core.base_env import Action
from core.interfaces import BaseAgent, DiscreteState


class HybridQSARSAAgent(BaseAgent):
    """
    Hybrid Reinforcement Learning Agent combining SARSA (On-Policy)
    and Q-Learning (Off-Policy).

    Supports three unification paradigms:
      1. 'linear': Q(sigma) framework interpolating between SARSA and Q-Learning targets:
         Target = r + gamma * [beta * Q(s', a') + (1 - beta) * max_a' Q(s', a')]
      2. 'expected': Expected SARSA using the full policy expectation:
         Target = r + gamma * [(1 - epsilon) * max_a' Q(s', a') + (epsilon / |A|) * sum(Q(s', a'))]
      3. 'hazard_gated': Mario hazard-conditioned hybrid:
         Uses Q-Learning (max_a') near lethal hazards (pit or ghost) to overcome risk-freezing,
         and SARSA / linear blend on safe corridors to avoid reckless collisions.
    """

    def __init__(
        self,
        num_actions: int = Action.NUM_ACTIONS,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        beta: float = 0.25,  # Weight on SARSA (0.0 = Pure Q-Learning, 1.0 = Pure SARSA)
        mode: str = "linear",  # 'linear', 'expected', or 'hazard_gated'
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        training: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0], got {beta}")
        if mode not in ("linear", "expected", "hazard_gated"):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'linear', 'expected', or 'hazard_gated'."
            )

        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.beta = beta
        self.mode = mode
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

    def _get_q(self, state: DiscreteState) -> np.ndarray:
        """Returns the Q-value array for the given discrete state."""
        return self.q_table[state]

    def select_action(self, state: DiscreteState) -> int:
        """
        Selects an action using epsilon-greedy exploration in training mode,
        or pure greedy exploitation in evaluation mode.
        Performs uniform random tie-breaking among maximal action values.
        """
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        q_values = self.q_table[state]
        max_val = np.max(q_values)
        candidates = np.flatnonzero(q_values == max_val)
        return int(np.random.choice(candidates))

    def compute_td_target(
        self,
        reward: float,
        next_state: DiscreteState,
        next_action: Optional[int],
        done: bool,
    ) -> float:
        """
        Computes the TD Target according to the selected hybrid mode:
          - Terminal: r
          - 'linear': r + gamma * [beta * Q(s', a') + (1 - beta) * max_a' Q(s', a')]
          - 'expected': r + gamma * [(1 - eps) * max_a' Q(s', a') + (eps / |A|) * sum(Q(s', a'))]
          - 'hazard_gated': Q-Learning if hazard ahead (pit or ghost), else linear SARSA/Q blend
        """
        if done:
            return float(reward)

        next_q = self.q_table[next_state]
        max_next_q = float(np.max(next_q))

        if self.mode == "expected":
            # Expected SARSA under epsilon-greedy policy
            if self.training:
                eps = self.epsilon
                mean_next_q = float(np.mean(next_q))
                expected_next_val = (1.0 - eps) * max_next_q + eps * mean_next_q
            else:
                expected_next_val = max_next_q
            return float(reward + self.gamma * expected_next_val)

        if self.mode == "hazard_gated":
            # Check proximity to lethal hazards:
            # - Index 4: pit_dist_bin (0: NONE, 1: FAR, 2: IMMEDIATE)
            # - Index 5: ghost_dist_bin (0: NONE, 1: FAR, 2: CLOSE)
            pit_dist_bin = next_state[4] if len(next_state) > 4 else 0
            ghost_dist_bin = next_state[5] if len(next_state) > 5 else 0
            is_lethal_hazard = (pit_dist_bin != 0) or (ghost_dist_bin != 0)

            if is_lethal_hazard:
                # Lethal hazard zone: use pure Q-Learning to overcome fear of ledges/ghosts
                target_future_val = max_next_q
            else:
                # Safe corridor: blend SARSA to avoid collision recklessness
                if next_action is None:
                    next_action = self.select_action(next_state)
                sarsa_next_q = float(next_q[next_action])
                target_future_val = (
                    self.beta * sarsa_next_q + (1.0 - self.beta) * max_next_q
                )
            return float(reward + self.gamma * target_future_val)

        # Default 'linear' Q(sigma) interpolation
        if next_action is None:
            next_action = self.select_action(next_state)
        sarsa_next_q = float(next_q[next_action])
        target_future_val = self.beta * sarsa_next_q + (1.0 - self.beta) * max_next_q
        return float(reward + self.gamma * target_future_val)

    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        done: bool,
        next_action: Optional[int] = None,
    ) -> float:
        """
        Updates Q(s, a) using the hybrid TD control update.
        Conforms strictly to BaseAgent.update while allowing optional next_action.
        If next_action is not supplied, it is sampled from the current policy.
        """
        current_q = self.q_table[state][action]
        td_target = self.compute_td_target(reward, next_state, next_action, done)
        td_error = td_target - current_q

        self.q_table[state][action] = current_q + self.alpha * td_error
        return float(td_error)

    def update_hybrid(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        next_action: int,
        done: bool,
    ) -> float:
        """
        Explicit update method adhering to the blueprint in .agent/HYBRID_SARSA_Q_LEARNING.md.
        """
        return self.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_action=next_action,
        )

    def decay_epsilon(self) -> float:
        """Decays epsilon monotonically down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def set_mode(self, training: bool) -> None:
        """Toggles between training mode and greedy evaluation mode."""
        self.training = training

    def save(self, filepath: str) -> None:
        """
        Saves the Q-table to disk in .pkl or .json format.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            serializable_q = {
                ",".join(map(str, s)): q_vals.tolist()
                for s, q_vals in self.q_table.items()
            }
            metadata = {
                "num_states_discovered": len(self.q_table),
                "num_actions": self.num_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "beta": self.beta,
                "mode": self.mode,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "q_table": serializable_q,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        else:
            checkpoint = {
                "num_actions": self.num_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "beta": self.beta,
                "mode": self.mode,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "q_table": dict(self.q_table),
            }
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[HybridAgent] Saved Q-table ({len(self.q_table)} states) to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Loads the Q-table and configuration from disk into the agent.
        Supports both .pkl and .json formats.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

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

        for attr in (
            "num_actions",
            "alpha",
            "gamma",
            "beta",
            "mode",
            "epsilon",
            "epsilon_min",
            "epsilon_decay",
        ):
            if attr in data:
                setattr(self, attr, data[attr])

        print(f"[HybridAgent] Loaded Q-table ({len(self.q_table)} states) from {filepath}")

    # Aliases for backward compatibility and uniform API with TabularQAgent
    load_q_table = load
    save_q_table = save


# Convenience alias
HybridAgent = HybridQSARSAAgent
