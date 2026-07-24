from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.lut import LogShaperSpec, ShapedLUT3D, bake_shaped_lut
from src.roll2film.photometric import (
    PhotometricColorOperator,
    RollNuisanceGauge,
    canonicalize_roll_nuisance,
)
from src.roll2film.simulator import default_truth_operator


def test_l0_roundtrip_jacobian_and_serialization() -> None:
    operator = PhotometricColorOperator(0.35, np.array([0.18, -0.07, -0.11]))
    rgb = np.random.default_rng(41).uniform(-0.5, 4.0, size=(2048, 3))
    output = operator.apply(rgb)
    replay = PhotometricColorOperator.from_dict(json.loads(json.dumps(operator.to_dict())))

    assert np.max(np.abs(operator.inverse(output) - rgb)) < 1e-14
    assert operator.jacobian_determinant > 0.0
    assert np.array_equal(replay.apply(rgb), output)
    assert np.array_equal(PhotometricColorOperator.identity().apply(rgb), rgb)


def test_l0_rejects_noncanonical_white_balance() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        PhotometricColorOperator(0.0, np.array([0.1, 0.0, 0.0]))


def test_roll_nuisance_gauge_recomposes_original_log_gains() -> None:
    exposure = np.array([0.3, -0.1, 0.25, -0.45, 0.2])
    white_balance = np.array(
        [
            [0.2, -0.1, -0.04],
            [-0.05, 0.08, 0.01],
            [0.12, -0.09, 0.03],
            [-0.15, 0.04, 0.02],
            [0.06, -0.03, -0.01],
        ]
    )
    gauge = canonicalize_roll_nuisance(exposure, white_balance)
    replay = RollNuisanceGauge.from_dict(json.loads(json.dumps(gauge.to_dict())))

    assert abs(float(np.mean(gauge.frame_log_exposure))) < 1e-16
    assert np.max(np.abs(np.sum(gauge.frame_log_white_balance, axis=1))) < 1e-16
    assert np.max(np.abs(gauge.recomposed_log_gains - (exposure[:, None] + white_balance))) < 1e-16
    assert np.array_equal(replay.recomposed_log_gains, gauge.recomposed_log_gains)


def test_log_shaper_and_shaped_lut_roundtrip_replay() -> None:
    shaper = LogShaperSpec(16.0, 4.0)
    linear = np.random.default_rng(42).uniform(0.0, 16.0, size=(4096, 3))
    shaped = shaper.apply(linear)
    assert np.max(np.abs(shaper.inverse(shaped) - linear)) < 2e-14
    assert np.min(shaper.derivative(linear)) > 0.0

    bundle = bake_shaped_lut(default_truth_operator(), 17, shaper)
    replay = ShapedLUT3D.from_dict(json.loads(json.dumps(bundle.to_dict())))
    assert np.array_equal(replay.apply(linear), bundle.apply(linear))

    with pytest.raises(ValueError, match="outside"):
        bundle.apply(np.array([[16.01, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="outside"):
        shaper.inverse(np.array([-0.01, 0.5, 1.0]))

