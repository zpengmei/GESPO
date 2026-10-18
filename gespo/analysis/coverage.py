"""Coverage on an explicitly specified, shared rectangular CV grid."""
import numpy as np


def coverage_curve(budgets, coordinates, x_range, y_range, grid_size=60):
    """Return unique budgets and cumulative occupied bins, excluding out-of-grid frames.

    The caller supplies aggregate unbiased production budgets. This function does
    not infer elapsed time or convert coverage into a speedup or kinetic score.
    Both endpoints of each coordinate range are included.
    """
    budgets, points = np.asarray(budgets, float), np.asarray(coordinates, float)
    bounds = np.asarray([x_range, y_range], float)
    if points.shape != (len(budgets), 2) or budgets.ndim != 1:
        raise ValueError('Expected one budget and two CVs per frame.')
    if grid_size < 1 or int(grid_size) != grid_size:
        raise ValueError('grid_size must be a positive integer.')
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError('Supply increasing finite bounds for both coordinates.')
    if not np.isfinite(budgets).all() or (budgets < 0).any() or not np.isfinite(points).all():
        raise ValueError('Budgets and coordinates must be finite; budgets must be nonnegative.')
    times = np.unique(budgets)
    inside = ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all(axis=1)
    valid = points[inside]
    cells = np.floor((valid - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) * grid_size).astype(int)
    cells = np.minimum(cells, grid_size - 1)
    first = {}
    for time, cell in zip(budgets[inside], cells):
        key = tuple(cell)
        first[key] = min(first.get(key, time), time)
    discoveries = np.sort(list(first.values()))
    return times, np.searchsorted(discoveries, times, side='right'), int((~inside).sum())
