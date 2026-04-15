"""Reinforcement learning policies for proposing metadynamics hills or CV targets."""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import VonMises, LogNormal, Categorical, Normal

from gespo.models.transformer import PositionalEncoding, TransformerEncoder

class GRPOAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_mixtures: int, hidden_dim: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.0),  # Dropout disabled for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        action_size_hill = num_mixtures * (1 + action_dim) 
        self.hill_params = nn.Linear(hidden_dim, action_size_hill)
        self.log_stds = nn.Parameter(torch.ones(action_size_hill) * -0.7) 

        self.start_mu = nn.Parameter(torch.zeros(action_dim))
        self.log_std_s = nn.Parameter(torch.full((action_dim,), 3.0))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.body.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.constant_(module.bias, 0)
        nn.init.uniform_(self.hill_params.weight, -3e-3, 3e-3)
        if self.hill_params.bias is not None: nn.init.constant_(self.hill_params.bias, 0)

    def forward(self, s: torch.Tensor) -> Normal:
        h = self.body(s)
        mus = self.hill_params(h)
        # Clamp log_stds before exp to prevent overflow
        log_stds_clamped = torch.clamp(self.log_stds, min=-10.0, max=5.0)
        stds = torch.exp(log_stds_clamped)
        # Guard against numerical issues: ensure finite, non-negative std
        stds = torch.where(torch.isfinite(stds), stds, torch.full_like(stds, 1.0))
        stds = torch.clamp(stds, min=1e-6, max=1e3)
        return Normal(mus, stds)

    @torch.no_grad()
    def sample_start(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.exp(self.log_std_s)
        scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, 1.0))
        scale = torch.clamp(scale, min=1e-6)
        dist = Normal(self.start_mu.expand(batch_size, -1), scale.expand(batch_size, -1))
        raw = dist.sample()
        wrapped = torch.remainder(raw + math.pi, 2 * math.pi) - math.pi
        logp = dist.log_prob(raw).sum(-1)
        return wrapped, logp

    def start_distribution(self, batch_size: int) -> Normal:
        """Return the Normal distribution used for start CVs for a given batch size.

        Exposes the per-batch start distribution so callers can sample raw values
        (with gradient) and compute cross-log-probs under a reference policy.
        """
        scale = torch.exp(self.log_std_s)
        scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, 1.0))
        scale = torch.clamp(scale, min=1e-6)
        return Normal(self.start_mu.expand(batch_size, -1), scale.expand(batch_size, -1))

