from __future__ import annotations

import numpy as np
import pytest

from src.film_physics.scanner import (
    SCANNER_SAFE_RESIDUAL_RUNTIME_ID,
    apply_scanner_safe_residual,
)


def test_identity_is_exact_and_inputs_remain_unchanged() -> None:
    source = np.linspace(0.0, 1.0, 36, dtype=np.float64).reshape(3, 4, 3)
    original = source.copy()
    output, receipt = apply_scanner_safe_residual(source, source.copy())
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(source, original)
    assert receipt.runtime_id == SCANNER_SAFE_RESIDUAL_RUNTIME_ID
    assert receipt.limited_pixel_fraction == 0.0
    assert receipt.median_scale == 1.0


def test_outside_candidate_is_scaled_without_clipping() -> None:
    source = np.asarray([[[0.2, 0.4, 0.6], [0.3, 0.3, 0.3]]], dtype=np.float64)
    candidate = np.asarray([[[1.4, 0.2, -0.2], [0.4, 0.2, 0.5]]], dtype=np.float64)
    output, receipt = apply_scanner_safe_residual(source, candidate)
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert receipt.limited_pixel_fraction == 0.5
    assert receipt.maximum_collinearity_error <= 2e-16
    assert np.any(output[0, 0] != candidate[0, 0])


def test_partition_execution_is_exact() -> None:
    rng = np.random.default_rng(4)
    source = rng.random((9, 7, 3), dtype=np.float64)
    candidate = source + rng.normal(scale=0.4, size=source.shape)
    full, _ = apply_scanner_safe_residual(source, candidate)
    tiled = np.concatenate(
        [apply_scanner_safe_residual(source[y : y + 2], candidate[y : y + 2])[0] for y in range(0, 9, 2)],
        axis=0,
    )
    np.testing.assert_array_equal(tiled, full)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)),
        (np.zeros((2, 3), dtype=np.float64), np.zeros((3, 3), dtype=np.float64)),
        (np.full((2, 3), np.nan), np.zeros((2, 3), dtype=np.float64)),
        (np.full((2, 3), -0.1), np.zeros((2, 3), dtype=np.float64)),
    ],
)
def test_invalid_inputs_fail_before_output(source: np.ndarray, candidate: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scanner safe residual"):
        apply_scanner_safe_residual(source, candidate)
