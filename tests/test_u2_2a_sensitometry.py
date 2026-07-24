from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.sensitometry import (
    AnchoredCharacteristicCurve,
    LogExposureEncoder,
    RGBSensitometryOperator,
)
from src.roll2film.splines import RationalQuadraticSpline


def _operator() -> RGBSensitometryOperator:
    x = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    rows = (
        ("red", [0.12, 0.2, 0.5, 1.0, 1.7, 2.0]),
        ("green", [0.1, 0.22, 0.52, 1.0, 1.62, 1.92]),
        ("blue", [0.15, 0.26, 0.58, 1.0, 1.55, 1.85]),
    )
    curves = tuple(
        AnchoredCharacteristicCurve(
            RationalQuadraticSpline.from_knots(x, np.asarray(y)), 0.0, 1.0, name
        )
        for name, y in rows
    )
    return RGBSensitometryOperator(LogExposureEncoder(0.18, 0.0001), curves)


def test_exposure_encoder_zero_anchor_roundtrip_and_guards() -> None:
    encoder = LogExposureEncoder(0.18, 0.0001)
    values = np.array([0.0, 0.001, 0.18, 1.0, 16.0])
    encoded = encoder.apply(values)
    assert encoded[2] == 0.0
    assert np.max(np.abs(encoder.inverse(encoded) - values)) < 2e-15
    assert np.all(encoder.derivative(values) > 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        encoder.apply(np.array([-1e-9]))
    with pytest.raises(ValueError, match="below"):
        encoder.inverse(np.array([encoder.minimum_log_exposure - 1e-9]))


def test_sensitometry_roundtrip_jacobian_anchor_and_replay() -> None:
    operator = _operator()
    rgb = np.random.default_rng(51).uniform(0.0, 16.0, size=(4096, 3))
    density = operator.apply(rgb)
    replay = RGBSensitometryOperator.from_dict(json.loads(json.dumps(operator.to_dict())))
    assert np.max(np.abs(operator.inverse(density) - rgb)) < 1e-11
    assert np.min(operator.jacobian_determinant(rgb)) > 0.0
    assert np.array_equal(replay.apply(rgb), density)
    assert np.array_equal(operator.apply(np.full((1, 3), 0.18)), np.ones((1, 3)))
    assert np.max(np.abs(operator.inverse(operator.apply(np.zeros((1, 3)))))) < 1e-15


def test_characteristic_curve_requires_shared_physical_anchor() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    spline = RationalQuadraticSpline.from_knots(x, np.array([0.2, 1.0, 1.5]))
    with pytest.raises(ValueError, match="neutral anchor"):
        AnchoredCharacteristicCurve(spline, 0.0, 1.1, "red")


def test_encoder_allows_only_roundoff_at_zero_boundary() -> None:
    encoder = LogExposureEncoder(0.18, 0.0001)
    assert encoder.inverse(np.array([encoder.minimum_log_exposure - 1e-13]))[0] == 0.0
    with pytest.raises(ValueError, match="below"):
        encoder.inverse(np.array([encoder.minimum_log_exposure - 2e-12]))
