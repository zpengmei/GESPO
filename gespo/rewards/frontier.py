"""Start-distance novelty reward.

After each geom2vec VAMP refresh, fit PCCA++ on the K-state transition matrix to
identify N coarse-grained metastable macrostates, then derive a representative
centroid for each macrostate as a membership-weighted average of the K kmeans
microstate centroids:

    macro_centroid[m] = sum_k M[k,m] * micro_centroid[k] / sum_k M[k,m]

For each walker, the trajectory reward is the L2 distance (in normalized linear-
VAMP space) from its trajectory's first frame to the nearest macrostate
centroid:

    reward(traj) = min_k ||z_start - macro_centroid_k||

A start far from all known macrostates → high reward; a start sitting on top of
an existing macrostate → low reward. The reward is applied at the trajectory
level (one scalar per walker) and is computed once per epoch from the realized
starting positions — it does not depend on 2 ns of MD relaxation. This rewards
the AR policy for proposing torsion starts that escape known basins.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _safe_normalize_rows(P: np.ndarray) -> np.ndarray:
    """Row-normalize a stochastic-like matrix; rows that sum to 0 become uniform."""
    P = np.asarray(P, dtype=np.float64).copy()
    row_sums = P.sum(axis=1, keepdims=True)
    zero = (row_sums.squeeze(-1) == 0)
    if zero.any():
        P[zero, :] = 1.0 / P.shape[1]
    else:
        P = P / np.maximum(row_sums, 1e-12)
    return P


def fit_pcca_memberships(
    transition_matrix: np.ndarray,
    n_macrostates: int,
) -> Optional[np.ndarray]:
    """Fit PCCA++ on a microstate transition matrix.

    Returns
    -------
    M : np.ndarray of shape (K, n_macrostates) or None
        Membership matrix mapping K microstates to n_macrostates macrostates.
        Returns None if PCCA fitting fails.
    """
    try:
        from deeptime.markov.tools.analysis.dense._pcca import pcca
    except ImportError:
        return None

    C = np.asarray(transition_matrix, dtype=np.float64)
    C = 0.5 * (C + C.T)
    P = _safe_normalize_rows(C)
    K = int(P.shape[0])
    n_macro = max(2, min(int(n_macrostates), K - 1))

    try:
        M = pcca(P, n_macro)
        M = np.asarray(M, dtype=np.float64)
        M = np.clip(M, 0.0, 1.0)
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / np.maximum(row_sums, 1e-12)
        return M.astype(np.float32, copy=False)
    except Exception:
        return None


class StartDistanceReward:
    """Per-trajectory reward: L2 distance from start frame to nearest PCCA macrostate centroid.

    Workflow per epoch (called from generative loop after geom2vec refresh):
        1. ``fit(centroids, transition_matrix)`` — fit PCCA, compute macrostate centroids.
        2. ``start_distance(z_starts)`` — returns per-walker min-distance to macrostate centroids.

    The input ``z_starts`` must be in the same normalized space as the kmeans
    centroids that produced ``transition_matrix`` (i.e., already shifted by
    msm_ig mu/std).
    """

    def __init__(self, n_macrostates: int = 7):
        self.n_macrostates = int(max(2, n_macrostates))
        self.centroids: Optional[np.ndarray] = None       # (K, D) microstate centroids
        self.M: Optional[np.ndarray] = None               # (K, n_macro) PCCA membership
        self.macro_centroids: Optional[np.ndarray] = None  # (n_macro, D)
        self._K: int = 0
        self._D: int = 0

    def fit(self, centroids: np.ndarray, transition_matrix: np.ndarray) -> bool:
        """Fit PCCA on the current MSM and compute macrostate centroids.

        Parameters
        ----------
        centroids : (K, D) array
            Microstate centroids in normalized linear VAMP coordinate space.
        transition_matrix : (K, K) array
            Microstate transition counts or probabilities.
        """
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2:
            return False
        self.centroids = centroids
        self._K = int(centroids.shape[0])
        self._D = int(centroids.shape[1])

        M = fit_pcca_memberships(transition_matrix, self.n_macrostates)
        if M is None or int(M.shape[0]) != self._K:
            self.M = None
            self.macro_centroids = None
            return False
        self.M = M
        weights = M.sum(axis=0).astype(np.float32)  # (n_macro,)
        macro = (M.T.astype(np.float32) @ centroids) / np.maximum(weights[:, None], 1e-12)
        self.macro_centroids = macro.astype(np.float32, copy=False)
        return True

    def is_ready(self) -> bool:
        return self.macro_centroids is not None

    def start_distance(self, z_starts: np.ndarray) -> np.ndarray:
        """Return per-frame min L2 distance to any macrostate centroid.

        Parameters
        ----------
        z_starts : (N, D) array of normalized linear-VAMP coordinates.

        Returns
        -------
        distances : (N,) float32 array of min-distances in [0, +inf).
        """
        if not self.is_ready():
            return np.zeros((int(np.asarray(z_starts).shape[0]),), dtype=np.float32)
        x = np.asarray(z_starts, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self._D:
            return np.zeros((x.shape[0],), dtype=np.float32)
        c = self.macro_centroids  # (n_macro, D)
        x_sq = (x * x).sum(axis=1, keepdims=True)
        c_sq = (c * c).sum(axis=1)[None, :]
        d2 = x_sq + c_sq - 2.0 * (x @ c.T)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(np.min(d2, axis=1)).astype(np.float32)
