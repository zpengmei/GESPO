"""Categorical policy over discrete VAMP/DeepTICA states."""
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

class StateLogitPolicy(nn.Module):
    """
    State-conditioned categorical policy over a fixed number of discrete
    states (e.g., VAMP/DeepTICA soft states). Used for runs where we only
    want to decide which basin/state to start from, with the actual
    coordinates drawn from history.
    """
    def __init__(self, num_states: int, feature_dim: Optional[int] = None, hidden_dim: int = 128):
        super().__init__()
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        if feature_dim is None:
            # Global logits over states; ignores features entirely.
            self.logits = nn.Parameter(torch.zeros(num_states, dtype=torch.float32))
            self.net = None
        else:
            # Simple MLP that maps flat features -> logits over states.
            self.logits = None
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_states),
            )
            # Initialize to an exactly uniform policy: all logits = 0
            # so that softmax(logits) = 1 / num_states for every state.
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            last_linear = self.net[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def forward(self, features: torch.Tensor) -> Categorical:
        """
        Args:
            features: Start features [B, F]. In the backward-compatible
                      path (feature_dim is None), only the batch dimension
                      is used to expand a global logits vector.
        Returns:
            Categorical distribution over states for each batch element.
        """
        if features.dim() == 0:
            batch_size = 1
        else:
            batch_size = int(features.shape[0])

        if self.net is None:
            # Global, state-independent logits.
            logits = self.logits.unsqueeze(0).expand(batch_size, -1)
        else:
            # Use the provided features to compute per-state logits.
            if features.dim() > 2:
                # Flatten any trailing dimensions after batch.
                features = features.view(batch_size, -1)
            logits = self.net(features)

        return Categorical(logits=logits)
