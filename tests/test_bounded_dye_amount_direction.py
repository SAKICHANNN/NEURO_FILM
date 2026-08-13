import numpy as np
import pytest

from src.film_physics.bounded_dye_amount_direction import (
    apply_bounded_dye_amount_direction,
)


def test_shared_scale_preserves_rgb_direction() -> None:
    base = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    candidate = np.array([[[1.4, 0.2, -0.4]]], dtype=np.float64)
    output, receipt = apply_bounded_dye_amount_direction(base, candidate)
    ratios = (output.astype(np.float64) - base) / (candidate - base)
    assert np.ptp(ratios) <= 2e-7
    assert receipt["hard_clipping_used"] == 0.0
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_interior_candidate_is_unchanged_to_float32() -> None:
    base = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    candidate = np.array([[[0.3, 0.4, 0.7]]], dtype=np.float64)
    output, receipt = apply_bounded_dye_amount_direction(base, candidate)
    np.testing.assert_array_equal(output, candidate.astype(np.float32))
    assert receipt["minimum_shared_scale"] == 1.0


def test_invalid_base_fails_closed() -> None:
    base = np.array([[[np.nan, 0.5, 0.8]]], dtype=np.float64)
    with pytest.raises(ValueError, match="invalid"):
        apply_bounded_dye_amount_direction(base, np.zeros_like(base))
