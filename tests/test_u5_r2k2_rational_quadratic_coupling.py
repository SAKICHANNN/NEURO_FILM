from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.rational_quadratic_coupling import (
    RationalQuadraticCouplingOperator,
    finite_difference_jacobians,
    fit_rational_quadratic_coupling,
)


def _identity() -> RationalQuadraticCouplingOperator:
    return RationalQuadraticCouplingOperator.identity(
        stage_channels=(0, 1, 2),
        axis_size=2,
        sigma=0.38,
        bin_count=4,
        minimum_bin_width=0.04,
        minimum_bin_height=0.04,
        minimum_derivative=0.02,
    )


def test_zero_coefficients_are_exact_identity() -> None:
    operator = _identity()
    rgb = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.13, 0.47, 0.91],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(operator.apply(rgb), rgb, atol=2e-16, rtol=0.0)
    np.testing.assert_allclose(operator.inverse(rgb), rgb, atol=2e-16, rtol=0.0)
    statistics = operator.stage_statistics(rgb)
    assert statistics["minimum_bin_width"] == pytest.approx(0.25)
    assert statistics["minimum_bin_height"] == pytest.approx(0.25)
    assert statistics["minimum_stage_derivative"] == pytest.approx(1.0)


def test_nontrivial_operator_is_bounded_invertible_and_positive() -> None:
    identity = _identity()
    coefficients = identity.coefficients.copy()
    coefficients[0, 0, :4] = [-0.4, 0.2, 0.5, -0.3]
    coefficients[0, 1, 4:8] = [0.3, -0.2, 0.1, -0.4]
    coefficients[1, 2, 8:] = [0.5, -0.25, 0.4]
    operator = RationalQuadraticCouplingOperator(
        stage_channels=identity.stage_channels,
        centers=identity.centers,
        sigma=identity.sigma,
        bin_count=identity.bin_count,
        minimum_bin_width=identity.minimum_bin_width,
        minimum_bin_height=identity.minimum_bin_height,
        minimum_derivative=identity.minimum_derivative,
        coefficients=coefficients,
    )
    axis = np.linspace(0.05, 0.95, 6)
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    transformed = operator.apply(rgb)
    assert np.min(transformed) >= 0.0
    assert np.max(transformed) <= 1.0
    np.testing.assert_allclose(operator.inverse(transformed), rgb, atol=1e-12)
    determinants = np.linalg.det(
        finite_difference_jacobians(operator, rgb, step=1e-6)
    )
    assert np.min(determinants) > 0.0


def test_serialization_and_partition_are_exact() -> None:
    identity = _identity()
    coefficients = identity.coefficients.copy()
    coefficients[:, 0, :] = np.linspace(
        -0.2, 0.2, coefficients.shape[-1]
    )
    operator = RationalQuadraticCouplingOperator(
        stage_channels=identity.stage_channels,
        centers=identity.centers,
        sigma=identity.sigma,
        bin_count=identity.bin_count,
        minimum_bin_width=identity.minimum_bin_width,
        minimum_bin_height=identity.minimum_bin_height,
        minimum_derivative=identity.minimum_derivative,
        coefficients=coefficients,
    )
    replay = RationalQuadraticCouplingOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    rgb = np.random.default_rng(260726).random((31, 3))
    expected = operator.apply(rgb)
    assert np.array_equal(replay.apply(rgb), expected)
    assert np.array_equal(
        np.concatenate(
            [operator.apply(rgb[:7]), operator.apply(rgb[7:19]), operator.apply(rgb[19:])]
        ),
        expected,
    )


def test_small_fit_is_repeat_deterministic() -> None:
    source = np.random.default_rng(4).random((48, 3))
    target = source.copy()
    target[:, 0] = source[:, 0] ** 1.2
    kwargs = {
        "stage_channels": (0, 1, 2),
        "axis_size": 2,
        "sigma": 0.38,
        "bin_count": 4,
        "minimum_bin_width": 0.04,
        "minimum_bin_height": 0.04,
        "minimum_derivative": 0.02,
        "epsilon": 1e-12,
        "maximum_absolute_coefficient": 2.0,
        "seed": 11,
        "steps": 20,
        "learning_rate": 0.02,
        "coefficient_l2": 1e-5,
        "gradient_clip_norm": 10.0,
        "thread_count": 1,
    }
    first = fit_rational_quadratic_coupling(source, target, **kwargs)
    second = fit_rational_quadratic_coupling(source, target, **kwargs)
    assert np.array_equal(first.coefficients, second.coefficients)
    assert np.sqrt(np.mean((first.apply(source) - target) ** 2)) < 0.02
