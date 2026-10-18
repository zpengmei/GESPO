import numpy as np
import pytest

from gespo.analysis.coverage import coverage_curve


def test_coverage_is_order_independent_and_handles_revisits_boundaries_and_exclusions():
    budget = [3, 1, 2, 4, 5]
    points = [[.9, .9], [0, 0], [.1, .1], [1, 1], [2, 2]]
    times, counts, excluded = coverage_curve(budget, points, [0, 1], [0, 1], 2)
    np.testing.assert_array_equal(times, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(counts, [1, 1, 2, 2, 2])
    assert excluded == 1


def test_distinct_frames_can_share_an_aggregate_budget():
    times, counts, _ = coverage_curve([1, 1], [[0, 0], [1, 1]], [0, 1], [0, 1], 2)
    np.testing.assert_array_equal(times, [1])
    np.testing.assert_array_equal(counts, [2])


def test_invalid_coordinates_are_not_silently_counted():
    with pytest.raises(ValueError):
        coverage_curve([1], [[np.nan, 0]], [0, 1], [0, 1])
