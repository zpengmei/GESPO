"""Properties of the sampled KL estimator used in the GRPO update."""
import math
import sys
from pathlib import Path

import torch
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.config import TrainingConfig
from gespo.training.optimization import GRPOOptimizer, _k3_kl


def test_k3_is_nonnegative_sample_by_sample():
    """The old estimator could go negative on a finite group; this one cannot."""
    torch.manual_seed(0)
    for _ in range(200):
        old = torch.randn(12)
        new = old + 0.05 * torch.randn(12)
        naive = (old - new).mean()
        k3 = _k3_kl(new, old)
        assert k3 >= 0.0
        if naive < 0:
            # exactly the case the previous code had to floor at zero
            assert k3 > 0.0


def test_k3_is_zero_for_identical_policies():
    x = torch.randn(16)
    assert torch.isclose(_k3_kl(x, x), torch.zeros(()), atol=1e-7)


def test_k3_is_unbiased_for_two_gaussians():
    """E[k3] should recover KL(p_old || p_new) for a case with a closed form."""
    torch.manual_seed(0)
    mu_old, mu_new, sigma = 0.0, 0.3, 1.0
    analytic = (mu_new - mu_old) ** 2 / (2.0 * sigma ** 2)

    def logp(x, mu):
        return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))

    y = torch.randn(400_000) * sigma + mu_old          # drawn from pi_old
    est = _k3_kl(logp(y, mu_new), logp(y, mu_old))
    assert abs(float(est) - analytic) < 0.01 * max(analytic, 1e-3) + 2e-4


def test_per_token_k3_preserves_sequence_ratio_and_gradient():
    new = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([2.0, 5.0], dtype=torch.float64)
    estimate = _k3_kl(new, torch.zeros_like(new), token_counts=counts)
    expected = ((new.exp() - new - 1) / counts).mean()
    torch.testing.assert_close(estimate, expected)
    estimate.backward()
    torch.testing.assert_close(new.grad, (new.detach().exp() - 1) / counts / 2)


@pytest.mark.parametrize('use_mask', [False, True])
def test_optimizer_combines_per_token_kl_with_sequence_policy_gradient(use_mask):
    class Policy(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(shift))

        def log_prob(self, torsions, *, angle_mask=None, global_values=None, **kwargs):
            counts = (angle_mask.sum((1, 2)) if angle_mask is not None else
                      torsions.new_full((torsions.shape[0],), torsions.shape[1] * torsions.shape[2]))
            return (counts + global_values.shape[1]) * self.shift

        def mean_token_entropy(self, torsions, **kwargs):
            return torsions.new_zeros(torsions.shape[0])

    agent, old = Policy(-0.02), Policy(0.0)
    mask = torch.zeros((3, 4, 3), dtype=torch.bool)
    mask[1, :2] = True
    mask[2] = True
    mask = mask if use_mask else None
    counts = mask.sum((1, 2)) + 2 if use_mask else torch.full((3,), 14)
    config = TrainingConfig(grpo_iters=1, norm_adv=False, kl_coef=1, ent_coef=0)
    data = dict(mode='torsion_ar', torsions=torch.zeros(3, 4, 3),
                angle_mask=mask, global_values=torch.zeros(3, 2),
                rewards=torch.ones(3), valid_mask=torch.ones(3, dtype=torch.bool))
    optimizer = GRPOOptimizer(config, agent, old, [torch.optim.SGD(agent.parameters(), lr=0)])
    stats = optimizer.update(data)
    log_ratio = -0.02 * counts
    expected_kl = ((log_ratio.expm1() - log_ratio) / counts).mean()
    assert stats['kl_s'] == pytest.approx(float(expected_kl), abs=1e-7)
    assert stats['pg_s'] == pytest.approx(float(log_ratio.exp().mean()), rel=1e-6)
