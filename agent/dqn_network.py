import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network (D3QN) architecture.

    Splits the representation into two separate streams:
      1. Value stream V(s): Scalar estimate of state desirability.
      2. Advantage stream A(s, a): Relative payoff for each discrete action.

    Aggregation:
      Q(s, a) = V(s) + [ A(s, a) - (1 / |A|) * sum_{a'} A(s, a') ]

    Subtracting the mean advantage stabilizes optimization and maintains identifiability.
    """

    def __init__(self, state_dim: int = 32, num_actions: int = 6, hidden_dim: int = 128):
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

        # State-Value Stream V(s)
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
        values = self.value_stream(features)           # [batch_size, 1]
        advantages = self.advantage_stream(features)   # [batch_size, num_actions]

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
