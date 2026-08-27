from __future__ import annotations

import numpy as np

from src.eval.wildrelight_spatial_envmap_explicit import (
    apply_log_gain_grid,
    environment_difference_features,
    fit_ridge,
    log_grid_descriptor,
    predict_ridge,
    source_dct_features,
    spherical_harmonic_coefficients,
    target_log_gain_grid,
)


def test_constant_environment_is_channel_symmetric_and_self_difference_zero() -> None:
    envmap = np.ones((8, 16, 3), dtype=np.float32)
    coefficients = spherical_harmonic_coefficients(envmap, 2.0**-16)
    np.testing.assert_allclose(coefficients[0], 1.0, atol=1e-15)
    np.testing.assert_allclose(coefficients[:, 0], coefficients[:, 1], atol=1e-15)
    np.testing.assert_allclose(coefficients[:, 1], coefficients[:, 2], atol=1e-15)
    np.testing.assert_allclose(
        environment_difference_features(envmap, envmap, 2.0**-16), 0.0
    )


def test_grid_and_dct_descriptors_have_frozen_shapes() -> None:
    source = np.linspace(0.1, 1.0, 8 * 12 * 3, dtype=np.float64).reshape(8, 12, 3)
    grid = log_grid_descriptor(source, grid_rows=4, grid_columns=4, epsilon=2**-16)
    assert grid.shape == (4, 4, 3)
    assert source_dct_features(grid).shape == (27,)


def test_ridge_recovers_deterministic_multi_output_map() -> None:
    x = np.arange(60, dtype=np.float64).reshape(12, 5) / 13.0
    weights = np.arange(15, dtype=np.float64).reshape(5, 3) / 17.0
    y = x @ weights + np.array([0.2, -0.1, 0.05])
    state = fit_ridge(x, y, 1e-8)
    np.testing.assert_allclose(predict_ridge(state, x), y, atol=1e-7)


def test_explicit_grid_preserves_zeros_and_constant_gain() -> None:
    source = np.ones((9, 11, 3), dtype=np.float32)
    source[0, 0] = 0.0
    grid = np.full((4, 4, 3), 1.0, dtype=np.float64)
    output = apply_log_gain_grid(
        source, grid, minimum_log2_gain=-4.0, maximum_log2_gain=4.0
    )
    np.testing.assert_array_equal(output[0, 0], 0.0)
    np.testing.assert_allclose(output[1:], 2.0, atol=1e-6)


def test_target_grid_recovers_uniform_gain() -> None:
    source = np.full((8, 8, 3), 0.25, dtype=np.float32)
    target = source * 4.0
    grid = target_log_gain_grid(
        source,
        target,
        grid_rows=4,
        grid_columns=4,
        epsilon=2**-16,
        minimum_log2_gain=-4.0,
        maximum_log2_gain=4.0,
    )
    expected = np.log2((1.0 + 2**-16) / (0.25 + 2**-16))
    np.testing.assert_allclose(grid, expected, atol=1e-12)
