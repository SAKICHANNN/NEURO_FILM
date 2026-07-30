from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.sensitometry_primitive import build_operator
from src.film_physics.response_bounded_exposure import (
    apply_response_bounded_exposure_residual,
    scan_transmittance,
)


ROOT = Path(__file__).resolve().parents[1]


def _operator():
    return build_operator(
        json.loads(
            (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
                encoding="utf-8"
            )
        )
    )


def test_response_bound_preserves_one_shared_exposure_direction() -> None:
    rng = np.random.default_rng(3014)
    source = rng.uniform(0.0, 2.0, size=(17, 19, 3)).astype(np.float64)
    candidate = np.maximum(
        source + rng.normal(0.0, 0.7, size=source.shape), 0.0
    )
    result = apply_response_bounded_exposure_residual(
        source,
        candidate,
        _operator(),
        maximum_transmittance_delta=0.009,
    )
    residual = candidate - source
    assert result.shared_scale.shape == (17, 19, 1)
    assert np.allclose(
        result.exposure - source,
        result.shared_scale * residual,
        rtol=0.0,
        atol=2e-15,
    )
    assert 0.0 <= float(result.shared_scale.min())
    assert float(result.shared_scale.max()) <= 1.0


def test_response_bound_is_exact_identity_for_zero_residual() -> None:
    source = np.linspace(0.0, 4.0, 3 * 11 * 13, dtype=np.float64).reshape(
        11, 13, 3
    )
    result = apply_response_bounded_exposure_residual(
        source,
        source.copy(),
        _operator(),
        maximum_transmittance_delta=0.009,
    )
    assert np.array_equal(result.exposure, source)
    assert np.array_equal(result.shared_scale, np.ones((11, 13, 1)))
    assert np.count_nonzero(result.raw_max_transmittance_delta) == 0
    assert np.count_nonzero(result.soft_transmittance_limit) == 0


def test_response_bound_limits_positive_and_negative_residuals() -> None:
    source = np.full((23, 29, 3), 0.18, dtype=np.float64)
    candidate = source.copy()
    candidate[:11, :, 0] = 16.0
    candidate[11:, :, 1] = 0.0
    candidate[:, :13, 2] = 8.0
    operator = _operator()
    result = apply_response_bounded_exposure_residual(
        source,
        candidate,
        operator,
        maximum_transmittance_delta=0.009,
    )
    delta = np.abs(
        scan_transmittance(result.exposure, operator)
        - scan_transmittance(source, operator)
    )
    assert float(delta.max()) <= 0.009 + 3e-15
    assert float(result.shared_scale.min()) < 1.0
    assert np.all(np.isfinite(result.exposure))


@pytest.mark.parametrize(
    ("source", "candidate", "error"),
    [
        (
            np.zeros((3, 4, 3), dtype=np.float32),
            np.zeros((3, 4, 3), dtype=np.float64),
            TypeError,
        ),
        (
            np.zeros((3, 4, 3), dtype=np.float64),
            np.zeros((3, 5, 3), dtype=np.float64),
            ValueError,
        ),
        (
            np.zeros((3, 4, 3), dtype=np.float64),
            -np.ones((3, 4, 3), dtype=np.float64),
            ValueError,
        ),
    ],
)
def test_response_bound_rejects_invalid_exposures(
    source: np.ndarray, candidate: np.ndarray, error: type[Exception]
) -> None:
    with pytest.raises(error):
        apply_response_bounded_exposure_residual(
            source,
            candidate,
            _operator(),
            maximum_transmittance_delta=0.009,
        )


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, float("nan")])
def test_response_bound_rejects_invalid_limit(value: float) -> None:
    source = np.zeros((3, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        apply_response_bounded_exposure_residual(
            source,
            source,
            _operator(),
            maximum_transmittance_delta=value,
        )
