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
         Uses Q-Learning (max_a') near lethal hazards (pit, fire, ghost, laser, wisp, monster) to overcome risk-freezing,
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
        trace_decay: float = 0.0,
        trace_threshold: float = 1e-3,
        hazard_safe_expected: bool = True,
        hazard_expected_eps_cap: float = 0.0,
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
        if not (0.0 <= trace_decay <= 1.0):
            raise ValueError(f"trace_decay must be in [0.0, 1.0], got {trace_decay}")
        if trace_threshold < 0.0:
            raise ValueError(f"trace_threshold must be >= 0.0, got {trace_threshold}")

        if not (0.0 <= hazard_expected_eps_cap <= 1.0):
            raise ValueError(
                f"hazard_expected_eps_cap must be in [0.0, 1.0], got {hazard_expected_eps_cap}"
            )

        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.beta = beta
        self.mode = mode
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.trace_decay = float(trace_decay)
        self.trace_threshold = float(trace_threshold)
        self.hazard_safe_expected = bool(hazard_safe_expected)
        self.hazard_expected_eps_cap = float(hazard_expected_eps_cap)
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
        self._recent_exploratory: Dict[Tuple[DiscreteState, int], bool] = {}

    def reset_traces(self) -> None:
        """Clears eligibility traces and exploration state between episodes."""
        self._traces.clear()
        self._last_action_exploratory = False
        self._recent_exploratory.clear()

    @staticmethod
    def _is_immediate_hazard(state: DiscreteState) -> bool:
        """Returns True if an immediate pit/ghost/laser/wisp/monster hazard is present."""
        pit_bin = state[4] if len(state) > 4 else 0
        ghost_bin = state[5] if len(state) > 5 else 0
        laser_bin = state[7] if len(state) > 7 else 0
        wisp_bin = state[8] if len(state) > 8 else 0
        monster_bin = state[11] if len(state) > 11 else 0
        return (
            (pit_bin >= 2)
            or (ghost_bin >= 2)
            or (laser_bin != 0)
            or (wisp_bin != 0)
            or (monster_bin != 0)
        )

    @staticmethod
    def _is_lethal_hazard(state: DiscreteState) -> bool:
        """Returns True if any lethal hazard proximity bin is active in the discrete state."""
        pit_dist_bin = state[4] if len(state) > 4 else 0
        ghost_dist_bin = state[5] if len(state) > 5 else 0
        laser_bin = state[7] if len(state) > 7 else 0
        wisp_bin = state[8] if len(state) > 8 else 0
        monster_bin = state[11] if len(state) > 11 else 0
        return (
            (pit_dist_bin != 0)
            or (ghost_dist_bin != 0)
            or (laser_bin != 0)
            or (wisp_bin != 0)
            or (monster_bin != 0)
        )

    def _get_q(self, state: DiscreteState) -> np.ndarray:
        """Returns the Q-value array for the given discrete state."""
        return self.q_table[state]

    def select_action(self, state: DiscreteState) -> int:
        """
        Selects an action using epsilon-greedy exploration in training mode,
        or pure greedy exploitation in evaluation mode (without mutating q_table).
        Performs uniform random tie-breaking among maximal action values.
        """
        if self.training and random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            if self.trace_decay > 0.0:
                q_values = self.q_table.get(state)
                is_exp = (
                    q_values is not None and float(q_values[action]) < float(np.max(q_values))
                )
                self._last_action_exploratory = is_exp
                self._recent_exploratory[(state, action)] = is_exp
            else:
                self._last_action_exploratory = False
            return action

        self._last_action_exploratory = False

        # Read-only lookup so evaluation never inserts unseen states into q_table
        q_values = self.q_table.get(state)
        if q_values is None:
            action = random.randrange(self.num_actions)
        else:
            max_val = np.max(q_values)
            candidates = np.flatnonzero(q_values == max_val)
            action = int(np.random.choice(candidates))

        if self.training and self.trace_decay > 0.0:
            self._recent_exploratory[(state, action)] = False
        return action

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
          - 'hazard_gated': Q-Learning if lethal hazard ahead (pit, ghost, laser, wisp, or monster), else linear SARSA/Q blend
        """
        if done:
            return float(reward)

        next_q = self.q_table.get(next_state)
        if next_q is None:
            return float(reward)

        max_next_q = float(np.max(next_q))

        if self.mode == "expected":
            # Expected SARSA under epsilon-greedy policy, gated at immediate lethal hazards
            if self.training:
                if self.hazard_safe_expected and self._is_immediate_hazard(next_state):
                    eps = min(self.epsilon, self.hazard_expected_eps_cap)
                else:
                    eps = self.epsilon
                mean_next_q = float(np.mean(next_q))
                expected_next_val = (1.0 - eps) * max_next_q + eps * mean_next_q
            else:
                expected_next_val = max_next_q
            return float(reward + self.gamma * expected_next_val)

        if self.mode == "hazard_gated":
            # Lethal hazards checked across pit(4), ghost(5), laser(7), wisp(8), monster(11)
            if self._is_lethal_hazard(next_state):
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


    def update_hybrid(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        next_action: Optional[int],
        done: bool,
    ) -> float:
        """Compatibility alias for hybrid SARSA/Q-learning update."""
        return self.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_action=next_action,
        )

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
        Updates Q(s, a) using the hybrid TD control update (or replacing traces if trace_decay > 0).
        Conforms strictly to BaseAgent.update while allowing optional next_action.
        If next_action is not supplied, it is sampled from the current policy.
        """
        is_exploratory = (
            self._recent_exploratory.pop((state, action), self._last_action_exploratory)
            if self.training
            else False
        )
        self._last_action_exploratory = False

        current_q = self.q_table[state][action]
        td_target = self.compute_td_target(reward, next_state, next_action, done)
        td_error = td_target - current_q

        # Fast path: pure 1-step TD incurs zero trace overhead
        if self.trace_decay == 0.0:
            self.q_table[state][action] = current_q + self.alpha * td_error
            return float(td_error)

        if (
            self.mode == "hazard_gated"
            and is_exploratory
            and (self._is_lethal_hazard(state) or self._is_lethal_hazard(next_state))
        ):
            self._traces.clear()
        elif (
            self.mode == "expected"
            and self.hazard_safe_expected
            and float(current_q) < float(np.max(self.q_table[state])) - 1e-9
            and (self._is_immediate_hazard(state) or self._is_immediate_hazard(next_state))
        ):
            self.reset_traces()

        self._traces[(state, action)] = 1.0
        alpha_delta = self.alpha * td_error
        for (tr_state, tr_action), eligibility in self._traces.items():
            self.q_table[tr_state][tr_action] += alpha_delta * eligibility

        if done:
            self._traces.clear()
            self._recent_exploratory.clear()
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
                "trace_decay": self.trace_decay,
                "hazard_safe_expected": self.hazard_safe_expected,
                "hazard_expected_eps_cap": self.hazard_expected_eps_cap,
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
                "trace_decay": self.trace_decay,
                "hazard_safe_expected": self.hazard_safe_expected,
                "hazard_expected_eps_cap": self.hazard_expected_eps_cap,
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

        for attr in (
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
        self.trace_decay = float(data.get("trace_decay", 0.0))
        self.hazard_safe_expected = bool(data.get("hazard_safe_expected", True))
        self.hazard_expected_eps_cap = float(data.get("hazard_expected_eps_cap", 0.0))
        self.reset_traces()

        print(f"[HybridAgent] Loaded Q-table ({len(self.q_table)} states) from {filepath}")

    # Aliases for backward compatibility and uniform API with TabularQAgent
    load_q_table = load
    save_q_table = save


# Convenience alias
HybridAgent = HybridQSARSAAgent
