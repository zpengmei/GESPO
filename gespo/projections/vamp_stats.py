"""
Sufficient statistics for VAMP-2 leave-one-out calculations.

This module provides the canonical SufficientStatistics class for tracking
running covariance matrices needed for efficient LOO-VAMP2 scoring.
"""

from typing import Optional
import numpy as np


class SufficientStatistics:
    """
    Track sufficient statistics for fast leave-one-out VAMP-2 calculation.

    Maintains running covariance matrices so that removing one trajectory
    requires only O(T) updates rather than full recomputation.

    Parameters
    ----------
    dim : int
        Dimensionality of the feature vectors.
    symmetrized : bool
        If True, use symmetrized VAMP-2 (C0 = 0.5*(C00+C11), C1 = 0.5*(C01+C01^T)).
    """

    def __init__(self, dim: int, *, symmetrized: bool = False):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        self.symmetrized = bool(symmetrized)
        self.n_samples = 0
        self.sum_x = np.zeros((self.dim,), dtype=np.float64)
        self.sum_y = np.zeros((self.dim,), dtype=np.float64)
        self.C00 = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.C11 = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.C01 = np.zeros((self.dim, self.dim), dtype=np.float64)

    def add_trajectory(self, x: np.ndarray, y: np.ndarray, ind_stop: Optional[np.ndarray] = None) -> None:
        """Add trajectory pair (x[t], y[t+lag]) to statistics."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x/y shape (N,{self.dim}), got {x.shape}.")
        if x.shape[0] < 1:
            return
        self.sum_x += np.sum(x, axis=0)
        self.sum_y += np.sum(y, axis=0)
        self.C00 += x.T @ x
        self.C11 += y.T @ y
        self.C01 += x.T @ y
        self.n_samples += int(x.shape[0])

    def remove_trajectory(self, x: np.ndarray, y: np.ndarray, ind_stop: Optional[np.ndarray] = None) -> None:
        """Remove trajectory pair from statistics (for LOO)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x/y shape (N,{self.dim}), got {x.shape}.")
        if x.shape[0] < 1:
            return
        self.sum_x -= np.sum(x, axis=0)
        self.sum_y -= np.sum(y, axis=0)
        self.C00 -= x.T @ x
        self.C11 -= y.T @ y
        self.C01 -= x.T @ y
        self.n_samples -= int(x.shape[0])

    def compute_vamp2_score(self, epsilon: float = 1e-6) -> float:
        """
        Compute VAMP-2 score from current sufficient statistics.

        Uses Bessel-corrected (1/(n-1)) sample covariances with mean removal.

        Returns:
            VAMP-2 score (sum of squared singular values of Koopman operator)
        """
        n = int(self.n_samples)
        if n < 2:
            return 0.0
        eps = float(epsilon) if float(epsilon) > 0 else 1e-6

        mu_x = self.sum_x / float(n)
        mu_y = self.sum_y / float(n)

        # Sample covariances with mean removal and Bessel correction.
        factor = 1.0 / float(max(n - 1, 1))
        c00 = (self.C00 - float(n) * np.outer(mu_x, mu_x)) * factor
        c11 = (self.C11 - float(n) * np.outer(mu_y, mu_y)) * factor
        c01 = (self.C01 - float(n) * np.outer(mu_x, mu_y)) * factor

        def _inv_sqrt(mat: np.ndarray) -> np.ndarray:
            vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64))
            vals = np.maximum(vals, eps)
            return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        try:
            if not self.symmetrized:
                c00_inv = _inv_sqrt(c00)
                c11_inv = _inv_sqrt(c11)
                K = c00_inv @ c01 @ c11_inv
                s = np.linalg.svd(K, compute_uv=False)
                return float(np.sum(s * s))
            cov0 = 0.5 * (c00 + c11)
            cov1 = 0.5 * (c01 + c01.T)
            cov0_inv = _inv_sqrt(cov0)
            K = cov0_inv @ cov1 @ cov0_inv
            s = np.linalg.svd(K, compute_uv=False)
            return float(np.sum(s * s))
        except np.linalg.LinAlgError:
            return 0.0
