from __future__ import annotations

import numpy as np
import pytest

from src.film_physics.analytical_density_direction import (
    apply_analytical_density_direction,
)


def test_density_direction_is_identity_when_in_bounds() -> None:
    base = np.full((3, 4, 3), 0.5, dtype=np.float32)
    residual = np.array([0.1, -0.1, 0.05], dtype=np.float64)
    residual = np.broadcast_to(residual, base.shape).copy()
    output, receipt = apply_analytical_density_direction(base, residual)
    expected = base.astype(np.float64) * np.power(10.0, -residual)
    assert np.array_equal(output, expected.astype(np.float32))
    assert receipt["minimum_density_direction_scale"] == 1.0
    assert receipt["limited_fraction"] == 0.0


def test_density_direction_preserves_vector_under_brightening_limit() -> None:
    base = np.array([[[0.95, 0.4, 0.2]]], dtype=np.float32)
    residual = np.array([[[-0.4, -0.2, 0.3]]], dtype=np.float64)
    output, receipt = apply_analytical_density_direction(base, residual)
    realized = -np.log10(output.astype(np.float64) / base.astype(np.float64))
    ratios = realized / residual
    assert np.max(ratios) - np.min(ratios) < 2e-6
    assert np.all(output > 0.0)
    assert np.all(output <= 1.0)
    assert receipt["minimum_density_direction_scale"] < 1.0
    assert receipt["hard_clipping_used"] == 0.0


@pytest.mark.parametrize(
    ("base", "residual"),
    [
        (np.zeros((1, 1, 3)), np.zeros((1, 1, 3))),
        (np.ones((1, 1, 3)), np.full((1, 1, 3), np.nan)),
        (np.ones((1, 1, 3)), np.ones((1, 1, 2))),
    ],
)
def test_density_direction_rejects_invalid_inputs(
    base: np.ndarray, residual: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        apply_analytical_density_direction(base, residual)
