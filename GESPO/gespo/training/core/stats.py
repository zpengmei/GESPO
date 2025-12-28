"""Statistics helpers for training loops and VAMP scoring."""

from __future__ import annotations

import numpy as np


class SufficientStatistics:
    """
    Track sufficient statistics for fast leave-one-out VAMP-2 calculation.

    Maintains running covariance matrices C00, C11, C01 so that removing
    one trajectory requires only O(T) updates rather than full recomputation.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n_samples = 0
        self.sum_x = np.zeros(dim, dtype=np.float64)
        self.sum_y = np.zeros(dim, dtype=np.float64)
        self.C00 = np.zeros((dim, dim), dtype=np.float64)
        self.C11 = np.zeros((dim, dim), dtype=np.float64)
        self.C01 = np.zeros((dim, dim), dtype=np.float64)

    def add_trajectory(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add trajectory pair (x[t], y[t+lag]) to statistics."""
        assert x.shape == y.shape
        n = x.shape[0]

        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.C00 += x.T @ x
        self.C11 += y.T @ y
        self.C01 += x.T @ y
        self.n_samples += n

    def remove_trajectory(self, x: np.ndarray, y: np.ndarray) -> None:
        """Remove trajectory pair from statistics (for LOO)."""
        assert x.shape == y.shape
        n = x.shape[0]

        self.sum_x -= x.sum(axis=0)
        self.sum_y -= y.sum(axis=0)
        self.C00 -= x.T @ x
        self.C11 -= y.T @ y
        self.C01 -= x.T @ y
        self.n_samples -= n

    def compute_vamp2_score(self, epsilon: float = 1e-6) -> float:
        """
        Compute VAMP-2 score from current sufficient statistics.

        Returns:
            VAMP-2 score (sum of squared singular values of Koopman operator)
        """
        if self.n_samples < 2:
            return 0.0
        # Center covariances
        mu_x = self.sum_x / self.n_samples
        mu_y = self.sum_y / self.n_samples

        C00_centered = self.C00 / self.n_samples - np.outer(mu_x, mu_x)
        C11_centered = self.C11 / self.n_samples - np.outer(mu_y, mu_y)
        C01_centered = self.C01 / self.n_samples - np.outer(mu_x, mu_y)

        def _inv_sqrt(mat: np.ndarray) -> np.ndarray:
            vals, vecs = np.linalg.eigh(mat)
            vals = np.maximum(vals, epsilon)
            return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        try:
            C00_inv = _inv_sqrt(C00_centered)
            C11_inv = _inv_sqrt(C11_centered)
            K = C00_inv @ C01_centered @ C11_inv
            singular_values = np.linalg.svd(K, compute_uv=False)
            vamp2 = np.sum(singular_values**2)
            return float(vamp2)
        except np.linalg.LinAlgError:
            return 0.0
