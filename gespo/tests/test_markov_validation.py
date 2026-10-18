import numpy as np
import pytest

from gespo.analysis.markov_validation import analyze_transition_model


@pytest.mark.parametrize('transition,reversible', [
    ([[.5, .4, .1], [.4, .2, .4], [.1, .4, .5]], True),
    ([[.1, .8, .1], [.2, .1, .7], [.7, .2, .1]], False),
])
def test_forward_and_backward_satisfy_their_own_harmonic_equations(transition, reversible):
    result = analyze_transition_model(transition, [0], [2])
    matrix, reverse = result['transition'], result['reverse_transition']
    pi = result['stationary']
    plus, minus = result['forward_committor'], result['backward_committor']
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-12)
    np.testing.assert_allclose(reverse.sum(1), 1, atol=1e-12)
    assert plus[0] == 0 and plus[2] == 1
    assert minus[0] == 1 and minus[2] == 0
    assert plus[1] == pytest.approx(matrix[1] @ plus)
    assert minus[1] == pytest.approx(reverse[1] @ minus)
    if reversible:
        np.testing.assert_allclose(minus, 1 - plus, atol=1e-12)
    else:
        assert result['backward_complement_error'] > .1
    flux = result['reactive_flux']
    assert flux[:, 0].sum() == 0 and flux[2].sum() == 0


@pytest.mark.parametrize('matrix,a,b', [
    ([[1, 1], [0, 1]], [0], [1]),
    (np.eye(3), [0], [2]),
    ([[.5, .5], [.5, .5]], [0], [0]),
    ([[.5, .5], [.5, .5]], [], [1]),
])
def test_invalid_models_are_rejected(matrix, a, b):
    with pytest.raises(ValueError):
        analyze_transition_model(matrix, a, b)
