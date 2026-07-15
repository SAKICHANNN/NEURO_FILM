from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.lut import bake_dense_lut
from src.roll2film.simulator import default_truth_operator
from src.roll2film.splines import AffineMonotoneSplineOperator, RationalQuadraticSpline


def _operator() -> AffineMonotoneSplineOperator:
    x = np.array([-0.25, 0.0, 0.18, 0.45, 0.75, 1.0, 1.35])
    curves = (
        np.array([-0.24, 0.0, 0.15, 0.47, 0.82, 1.04, 1.37]),
        np.array([-0.27, -0.01, 0.17, 0.44, 0.72, 0.98, 1.32]),
        np.array([-0.23, 0.01, 0.20, 0.49, 0.78, 1.02, 1.36]),
    )
    return AffineMonotoneSplineOperator(
        default_truth_operator(),
        tuple(RationalQuadraticSpline.from_knots(x, y) for y in curves),
    )


def test_rational_quadratic_spline_round_trip_and_positive_derivative() -> None:
    spline = _operator().splines[0]
    values = np.linspace(-0.5, 1.6, 4097)
    restored = spline.inverse(spline.apply(values))

    assert np.max(np.abs(restored - values)) < 2e-12
    assert float(spline.derivative(values).min()) > 0.0


def test_l2_operator_round_trip_jacobian_and_serialization() -> None:
    operator = _operator()
    rgb = np.random.default_rng(17).uniform(-0.1, 1.1, size=(8192, 3))
    restored = operator.inverse(operator.apply(rgb))
    serialized = json.dumps(operator.to_dict(), sort_keys=True)
    replay = AffineMonotoneSplineOperator.from_dict(json.loads(serialized))

    assert np.max(np.abs(restored - rgb)) < 5e-12
    assert float(operator.jacobian_determinant(rgb).min()) > 0.0
    assert np.array_equal(replay.apply(rgb), operator.apply(rgb))


def test_l2_identity_is_exact_and_invalid_spline_fails_closed() -> None:
    rgb = np.random.default_rng(18).uniform(-0.2, 1.2, size=(256, 3))
    assert np.array_equal(AffineMonotoneSplineOperator.identity().apply(rgb), rgb)

    with pytest.raises(ValueError, match="strictly increasing"):
        RationalQuadraticSpline.from_knots(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        RationalQuadraticSpline(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]))


def test_dense_lut_bake_parity_improves_from_33_to_65() -> None:
    operator = _operator()
    rgb = np.random.default_rng(19).uniform(0.0, 1.0, size=(4096, 3))
    truth = operator.apply(rgb)
    lut33 = bake_dense_lut(operator, 33)
    lut65 = bake_dense_lut(operator, 65)
    error33 = float(np.max(np.abs(lut33.apply(rgb) - truth)))
    error65 = float(np.max(np.abs(lut65.apply(rgb) - truth)))

    assert error33 < 7e-4
    assert error65 < error33 * 0.35
    with pytest.raises(ValueError, match="outside"):
        lut33.apply(np.array([[1.01, 0.5, 0.5]]))
