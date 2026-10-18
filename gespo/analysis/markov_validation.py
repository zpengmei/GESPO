"""Model-level committors and reversibility diagnostics for finite Markov chains.

These calculations validate the fitted chain's equations, not the adequacy of
its state discretization, lag time, or underlying adaptive trajectory data.
"""
from __future__ import annotations

import numpy as np


def analyze_transition_model(transition, reactant, product):
    """Compute forward/backward committors without assuming detailed balance.

    Requires a stochastic matrix with a unique, strictly positive stationary
    distribution and nonempty, disjoint endpoint index sets.
    """
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Transition matrix must be square.')
    n = matrix.shape[0]
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Transition probabilities must be finite and nonnegative.')
    if not np.allclose(matrix.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('Transition rows must sum to one; counts are not probabilities.')
    a, b = np.unique(reactant), np.unique(product)
    for indices in (a, b):
        if not indices.size or indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= n).any():
            raise ValueError('Endpoint sets must contain valid integer state indices.')
    if np.intersect1d(a, b).size:
        raise ValueError('Reactant and product sets must be disjoint.')
    if np.linalg.matrix_rank(matrix.T - np.eye(n)) != n - 1:
        raise ValueError('Stationary distribution is not unique.')
    stationary = np.linalg.lstsq(
        np.vstack([matrix.T - np.eye(n), np.ones(n)]), np.r_[np.zeros(n), 1], rcond=None)[0]
    if (stationary <= 0).any():
        raise ValueError('Time reversal requires positive stationary weight for every state.')
    reverse = matrix.T * stationary[None, :] / stationary[:, None]
    interior = np.setdiff1d(np.arange(n), np.r_[a, b])

    def committor(kernel, zero, one):
        result = np.zeros(n)
        result[one] = 1
        if interior.size:
            result[interior] = np.linalg.solve(
                np.eye(interior.size) - kernel[np.ix_(interior, interior)],
                kernel[np.ix_(interior, one)].sum(axis=1))
        return result

    forward = committor(matrix, a, b)
    backward = committor(reverse, b, a)
    traffic = stationary[:, None] * matrix
    flux = traffic * backward[:, None] * forward[None, :]
    return dict(
        transition=matrix, stationary=stationary, reverse_transition=reverse,
        forward_committor=forward, backward_committor=backward,
        reactive_flux=flux, net_reactive_flux=np.maximum(flux - flux.T, 0),
        detailed_balance_residual=float(np.abs(traffic - traffic.T).sum() / (2 * traffic.sum())),
        backward_complement_error=float(np.max(np.abs(backward - (1 - forward)))),
    )
