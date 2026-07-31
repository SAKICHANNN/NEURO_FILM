import numpy as np
import pytest

from src.roll2film.analytical_residual_composition import (
    compose_analytical_residual,
)


def test_analytical_composition_preserves_base_boundaries_and_guards_new_ones() -> None:
    base = np.asarray(
        [[0.0, 0.2, 1.0], [0.4, 0.5, 0.6], [0.01, 0.99, 0.5]],
        dtype=np.float64,
    )
    target = np.asarray(
        [[0.3, 0.0, 0.7], [1.0, 0.0, 0.2], [0.0, 1.0, 0.9]],
        dtype=np.float64,
    )
    result = compose_analytical_residual(
        base,
        target,
        hard_boundary_epsilon=0.5 / 255.0,
        guard_boundary_epsilon=1.0 / 255.0,
    )
    assert np.all(result.output >= 0.0)
    assert np.all(result.output <= 1.0)
    assert result.output[0, 0] > 0.0
    assert result.output[0, 2] < 1.0
    assert np.all(result.residual_scale >= 0.0)
    assert np.all(result.residual_scale <= 1.0)
    assert np.any(result.residual_scale < 1.0)


def test_analytical_composition_is_exact_when_target_is_inside_rails() -> None:
    base = np.full((8, 3), 0.5)
    target = np.linspace(0.1, 0.9, 24).reshape(8, 3)
    result = compose_analytical_residual(
        base,
        target,
        hard_boundary_epsilon=0.5 / 255.0,
        guard_boundary_epsilon=1.0 / 255.0,
    )
    np.testing.assert_allclose(result.output, target, atol=1e-15, rtol=0.0)
    np.testing.assert_array_equal(result.residual_scale, np.ones(8))


def test_analytical_composition_rejects_invalid_rails() -> None:
    with pytest.raises(ValueError, match="invalid"):
        compose_analytical_residual(
            np.zeros((2, 3)),
            np.ones((2, 3)),
            hard_boundary_epsilon=0.1,
            guard_boundary_epsilon=0.05,
        )
