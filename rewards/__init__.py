from .base_reward import BaseRewardStrategy
from .balanced_reward import BalancedRewardStrategy
from .dense_reward import DenseRewardStrategy
from .sparse_reward import SparseRewardStrategy


def create_reward_strategy(name: str = "balanced", **kwargs) -> BaseRewardStrategy:
    """Factory helper to instantiate a reward strategy by name."""
    strategy_name = name.lower().strip()
    if strategy_name == "balanced":
        return BalancedRewardStrategy(**kwargs)
    elif strategy_name == "sparse":
        return SparseRewardStrategy(**kwargs)
    elif strategy_name == "dense":
        return DenseRewardStrategy(**kwargs)
    else:
        raise ValueError(
            f"Unknown reward strategy '{name}'. Available: 'balanced', 'sparse', 'dense'."
        )


__all__ = [
    "BaseRewardStrategy",
    "BalancedRewardStrategy",
    "DenseRewardStrategy",
    "SparseRewardStrategy",
    "create_reward_strategy",
]
