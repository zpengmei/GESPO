"""Properties of the sampled KL estimator used in the GRPO update."""
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.training.optimization import _k3_kl


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
