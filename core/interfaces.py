from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from core.base_env import RawState, GameEvents

# Type alias for the 5-tuple discrete state representation:
# (is_grounded, vy_sign, obstacle_dist_bin, obstacle_height_bin, pit_dist_bin)
DiscreteState = Tuple[int, int, int, int, int]


class BaseDiscretizer(ABC):
    """Abstract interface for continuous-to-discrete state mapping."""

    @abstractmethod
    def discretize(self, raw_state: RawState) -> DiscreteState:
        """
        Converts continuous physical measurements into a compact hashable discrete state tuple.
        """
        pass


class BaseRewardStrategy(ABC):
    """Abstract interface for modular reward calculation strategies."""

    @abstractmethod
    def calculate_reward(
        self,
        prev_state: RawState,
        action: int,
        next_state: RawState,
        events: GameEvents,
        done: bool,
    ) -> float:
        """
        Calculates the scalar reinforcement feedback signal r_{t+1}.
        """
        pass


class BaseAgent(ABC):
    """Abstract interface for RL decision agents."""

    @abstractmethod
    def select_action(self, state: DiscreteState) -> int:
        """Selects an action conforming to core.Action based on policy."""
        pass

    @abstractmethod
    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        done: bool,
    ) -> float:
        """
        Updates internal action-value estimates using experience tuple.
        Returns the TD error delta.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serializes agent policy / Q-table to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Restores agent policy / Q-table from disk."""
        pass