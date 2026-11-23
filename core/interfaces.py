from abc import ABC, abstractmethod
from typing import Tuple
from core.base_env import RawState, GameEvents

DiscreteState = Tuple[int, ...]


class BaseDiscretizer(ABC):
    """Abstract interface for continuous-to-discrete state mapping."""

    @abstractmethod
    def discretize(self, raw_state: RawState) -> DiscreteState:
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
        pass


class BaseAgent(ABC):
    """Abstract interface for RL decision agents."""

    @abstractmethod
    def select_action(self, state: DiscreteState) -> int:
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
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @abstractmethod
    def decay_epsilon(self) -> float:
        pass

    @abstractmethod
    def set_mode(self, training: bool) -> None:
        pass
