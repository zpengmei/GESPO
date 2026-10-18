"""Likelihoods for the clipped global descriptors emitted by the launch policy."""
import math
from unittest.mock import patch

import torch

from gespo.policies.torsion_ar import TorsionARPolicy


def test_zero_descriptor_uses_mixture_probability_mass_and_positive_uses_density():
    policy = TorsionARPolicy(num_residues=2, d_model=16, nhead=2,
                            num_layers=1, kmix=2, n_global_tokens=1, dropout=0)
    logits = torch.log(torch.tensor([0.25, 0.75])).expand(2, 1, 2)
    mu = torch.tensor([0.0, 1.0], requires_grad=True)
    sigma = torch.tensor([1.0, 2.0])
    globals_ = torch.tensor([[0.0], [1.0]])
    torsions = torch.zeros(2, 2, 3)
    mask = torch.zeros_like(torsions, dtype=torch.bool)
    params = (logits, mu.expand(2, 1, 2), sigma.expand(2, 1, 2))
    with patch.object(policy, '_global_params', return_value=params):
        result = policy.log_prob(torsions, angle_mask=mask, global_values=globals_)

    weights = torch.tensor([0.25, 0.75])
    normals = torch.distributions.Normal(mu, sigma)
    mass = (weights * normals.cdf(torch.tensor(0.0))).sum()
    density = (weights * normals.log_prob(torch.tensor(math.log(2.0))).exp()).sum()
    torch.testing.assert_close(result, torch.stack([mass.log(), density.log()]))
    result.sum().backward()
    assert torch.isfinite(mu.grad).all()
    assert mu.grad.abs().sum() > 0
