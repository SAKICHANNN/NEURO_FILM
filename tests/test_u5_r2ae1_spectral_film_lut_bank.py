from __future__ import annotations

import numpy as np

from src.eval.velvia_datasheet_witness import encoded_srgb_to_linear
from src.eval.spectral_film_lut_bank import (
    array_sha256,
    local_jacobian_metrics,
    median_delta_e76,
    strength_path_metrics,
    symmetric_basic_residual,
    synthetic_cube,
)


def test_synthetic_cube_has_exact_endpoints_and_shape() -> None:
    cube = synthetic_cube(5)
    assert cube.shape == (5, 5, 5, 3)
    np.testing.assert_array_equal(cube[0, 0, 0], np.zeros(3))
    np.testing.assert_array_equal(cube[-1, -1, -1], np.ones(3))


def test_identity_jacobian_is_positive_and_unit_norm() -> None:
    metrics = local_jacobian_metrics(synthetic_cube(7))
    assert abs(metrics["minimum_jacobian_determinant"] - 1.0) < 1e-12
    assert metrics["negative_jacobian_fraction"] == 0.0
    assert abs(metrics["maximum_jacobian_spectral_norm"] - 1.0) < 1e-12


def test_folded_red_axis_is_detected() -> None:
    cube = synthetic_cube(7)
    cube[..., 0] = 1.0 - cube[..., 0]
    metrics = local_jacobian_metrics(cube)
    assert metrics["minimum_jacobian_determinant"] < 0.0
    assert metrics["negative_jacobian_fraction"] == 1.0


def test_strength_path_is_one_direction_not_a_second_mode() -> None:
    identity = synthetic_cube(5)
    full = np.clip(identity**1.4 + np.array([0.03, 0.0, 0.01]), 0.0, 1.0)
    metrics = strength_path_metrics(identity, full, 0.75)
    assert abs(metrics["fitted_strength"] - 0.75) < 1e-15
    assert metrics["residual_rgb_rmse"] < 1e-15
    assert metrics["explained_energy_fraction"] == 1.0


def test_array_hash_binds_dtype_and_shape() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(4, 2, 3))


def test_symmetric_basic_residual_absorbs_simple_exposure() -> None:
    cube = synthetic_cube(7)
    linear = encoded_srgb_to_linear(cube) * 0.8
    darker = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear ** (1.0 / 2.4) - 0.055,
    )
    residual = symmetric_basic_residual(cube, darker)
    assert residual["conservative_minimum_delta_e76_median"] < 1e-3


def test_delta_e_detects_nonidentity() -> None:
    cube = synthetic_cube(5)
    shifted = np.clip(cube + np.array([0.05, 0.0, 0.0]), 0.0, 1.0)
    assert median_delta_e76(cube, shifted) > 1.0
