"""Utilities for Geom2Vec-based training coordination."""

from __future__ import annotations
from gespo.utils.logging import log_print

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_states_by_overlap(
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Compute a permutation that best aligns new states to previous ones
    based on a confusion matrix of frame-wise assignments.

    Only aligns states that are actually populated in the data.
    Unpopulated states use identity mapping.

    Args:
        old_labels: [N] - state labels under the previous projector
        new_labels: [N] - raw state labels under the refreshed projector
        n_states:   int - number of discrete states (assumed consistent)

    Returns:
        permutation: [n_states] - mapping old_label -> new_label
    """
    old_labels = np.asarray(old_labels, dtype=int)
    new_labels = np.asarray(new_labels, dtype=int)
    if old_labels.shape[0] != new_labels.shape[0]:
        raise ValueError(
            "old_labels and new_labels must have same length, "
            f"got {old_labels.shape[0]} vs {new_labels.shape[0]}"
        )

    # Identify which states are actually populated in old and new
    old_populated = np.unique(old_labels)
    new_populated = np.unique(new_labels)
    old_populated = old_populated[(old_populated >= 0) & (old_populated < n_states)]
    new_populated = new_populated[(new_populated >= 0) & (new_populated < n_states)]

    if len(old_populated) == 0 or len(new_populated) == 0:
        # No valid states to align
        return np.arange(n_states, dtype=int)

    # Build confusion matrix ONLY for populated states
    # overlap[i, j] = count of frames where old=old_populated[i] and new=new_populated[j]
    overlap = np.zeros((len(old_populated), len(new_populated)), dtype=np.int64)
    old_to_idx = {state: idx for idx, state in enumerate(old_populated)}
    new_to_idx = {state: idx for idx, state in enumerate(new_populated)}

    for o, n in zip(old_labels, new_labels):
        if o in old_to_idx and n in new_to_idx:
            overlap[old_to_idx[o], new_to_idx[n]] += 1

    if overlap.sum() == 0:
        # No usable overlap information; fall back to identity mapping.
        return np.arange(n_states, dtype=int)

    # Hungarian solver minimizes cost, so use negative overlap as cost
    # to maximize agreement between old and new labels.
    cost_matrix = -overlap.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation: start with identity for all states
    permutation = np.arange(n_states, dtype=int)

    # Only update permutation for populated states that were matched
    for old_idx, new_idx in zip(row_ind, col_ind):
        old_state = old_populated[old_idx]
        new_state = new_populated[new_idx]
        permutation[old_state] = new_state

    # Ensure permutation is a true bijection (no collisions). After the
    # Hungarian assignment, some old states may remain unmapped and
    # some new states may be unused. We fill the remaining old states
    # so that every new state appears exactly once in the image.
    used_new = set(permutation[old_populated])
    all_new = set(range(n_states))
    unused_new = [s for s in sorted(all_new) if s not in used_new]

    for old_state in range(n_states):
        if old_state in old_populated:
            continue
        if old_state not in used_new:
            permutation[old_state] = old_state
            used_new.add(old_state)
        elif unused_new:
            new_state = unused_new.pop(0)
            permutation[old_state] = new_state
            used_new.add(new_state)

    log_print(
        "[State Alignment] Matched "
        f"{len(row_ind)} populated states: {len(old_populated)} old → {len(new_populated)} new"
    )

    return permutation
