"""Von Mises start policy and mixture distribution utilities."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import VonMises, Normal

class VonMisesStartPolicy(nn.Module):
    """
    Simple start policy over angular CV targets using independent Von Mises factors.

    This is intended for lightweight toy experiments (e.g., Ala2) where the start
    action is just a small set of angles such as (phi, psi).
    """

    def __init__(
        self,
        action_dim: int = 2,
        kappa_floor: float = 1e-3,
        log_kappa_init: float = 0.0,
        train_kappa: bool = True,
        kappa_max: Optional[float] = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        self.kappa_floor = float(kappa_floor)
        if self.kappa_floor < 0:
            raise ValueError(f"kappa_floor must be >= 0, got {self.kappa_floor}")
        self.log_kappa_init = float(log_kappa_init)
        self.train_kappa = bool(train_kappa)
        self.kappa_max = None if kappa_max is None else float(kappa_max)
        if self.kappa_max is not None and self.kappa_max <= 0.0:
            self.kappa_max = None

        self.mu = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32))
        # log_kappa is passed through softplus to obtain a positive concentration.
        # log_kappa_init=0 => kappa≈0.693 (moderate); negative values make sampling wider.
        self.log_kappa = nn.Parameter(torch.full((self.action_dim,), float(self.log_kappa_init), dtype=torch.float32))
        if not self.train_kappa:
            self.log_kappa.requires_grad_(False)

    def start_distribution(self, batch_size: int) -> VonMises:
        bsz = int(batch_size)
        if bsz <= 0:
            raise ValueError(f"batch_size must be > 0, got {bsz}")
        mu = self.mu.unsqueeze(0).expand(bsz, -1)
        kappa = F.softplus(self.log_kappa).unsqueeze(0).expand(bsz, -1) + self.kappa_floor
        if self.kappa_max is not None:
            kappa = torch.clamp(kappa, max=float(self.kappa_max))
        return VonMises(mu, kappa)


def mixture_vonmises_log_prob(x: torch.Tensor,
                              logits: torch.Tensor,
                              mu: torch.Tensor,
                              kappa: torch.Tensor) -> torch.Tensor:
    """
    Mixture of von Mises log-probability for angular data.

    Args:
        x:      [...], angles in radians.
        logits: [..., K] unnormalized mixture logits.
        mu:     [..., K] mixture means in radians.
        kappa:  [..., K] positive concentrations.

    Returns:
        log_prob: [...] total log probability under the mixture.
    """
    logw = F.log_softmax(logits, dim=-1)  # [..., K]
    vm = VonMises(mu, kappa)              # broadcastable over leading dims
    lp = vm.log_prob(x.unsqueeze(-1))     # [..., K]
    return torch.logsumexp(logw + lp, dim=-1)  # [...]


def mixture_normal_log_prob(
    x: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Mixture of Normal log-probability for scalar data.

    Args:
        x:      [...], real-valued targets.
        logits: [..., K] unnormalized mixture logits.
        mu:     [..., K] mixture means.
        sigma:  [..., K] positive stddevs.

    Returns:
        log_prob: [...] total log probability under the mixture.
    """
    logw = F.log_softmax(logits, dim=-1)
    sigma = torch.clamp(sigma, min=1e-6)
    norm = Normal(mu, sigma)
    lp = norm.log_prob(x.unsqueeze(-1))  # [..., K]
    return torch.logsumexp(logw + lp, dim=-1)  # [...]
