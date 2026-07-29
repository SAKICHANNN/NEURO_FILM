from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.fixed_gaussian_residual import (
    FixedNeutralGaussianLogOddsOperator,
    FixedNeutralGaussianResidualOperator,
    fit_fixed_neutral_gaussian_log_odds,
    fit_fixed_neutral_gaussian_residual,
    fixed_cube_centers,
)
from src.roll2film.positive_film import PositiveFilmResponseOperator


def _base() -> PositiveFilmResponseOperator:
    return PositiveFilmResponseOperator(
        capture_matrix=np.eye(3),
        response_midpoints=np.full(3, -3.0),
        response_slopes=np.ones(3),
        maximum_responses=np.ones(3),
        scan_matrix=np.eye(3),
    )


def test_fixed_centres_and_neutral_features_are_exact() -> None:
    centers = fixed_cube_centers((0.25, 0.75))
    assert centers.shape == (8, 3)
    operator = FixedNeutralGaussianResidualOperator(
        base=_base(),
        centers=centers,
        sigma=0.35,
        coefficients=np.arange(24, dtype=np.float64).reshape(8, 3) / 100.0,
    )
    neutral = np.repeat(np.linspace(0.0, 1.0, 19)[:, None], 3, axis=1)
    assert np.array_equal(operator.features(neutral), np.zeros((19, 8)))
    assert np.array_equal(operator.apply(neutral), operator.base.apply(neutral))


def test_ridge_fit_reduces_known_local_residual_and_roundtrips() -> None:
    rng = np.random.default_rng(20260730)
    source = rng.uniform(0.03, 0.97, size=(128, 3))
    centers = fixed_cube_centers((0.25, 0.75))
    truth = FixedNeutralGaussianResidualOperator(
        base=_base(),
        centers=centers,
        sigma=0.35,
        coefficients=rng.normal(0.0, 0.002, size=(8, 3)),
    )
    target = truth.apply(source)
    fitted = fit_fixed_neutral_gaussian_residual(
        _base(),
        source,
        target,
        centers=centers,
        sigma=0.35,
        ridge=1e-8,
    )
    base_rmse = float(np.sqrt(np.mean((_base().apply(source) - target) ** 2)))
    fit_rmse = float(np.sqrt(np.mean((fitted.apply(source) - target) ** 2)))
    assert fit_rmse < base_rmse * 1e-4
    restored = FixedNeutralGaussianResidualOperator.from_dict(fitted.to_dict())
    assert np.array_equal(restored.apply(source), fitted.apply(source))


def test_invalid_geometry_and_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        fixed_cube_centers((0.25, 0.25))
    with pytest.raises(ValueError):
        FixedNeutralGaussianResidualOperator(
            base=_base(),
            centers=np.zeros((8, 3)),
            sigma=0.35,
            coefficients=np.zeros((8, 3)),
        )
    operator = FixedNeutralGaussianResidualOperator(
        base=_base(),
        centers=fixed_cube_centers((0.25, 0.75)),
        sigma=0.35,
        coefficients=np.zeros((8, 3)),
    )
    with pytest.raises(ValueError):
        operator.apply(np.asarray([[np.nan, 0.0, 0.0]]))


def test_log_odds_variant_is_cube_and_endpoint_preserving() -> None:
    rng = np.random.default_rng(20260731)
    centers = fixed_cube_centers((0.25, 0.75))
    operator = FixedNeutralGaussianLogOddsOperator(
        base=_base(),
        centers=centers,
        sigma=0.35,
        coefficients=rng.normal(0.0, 3.0, size=(8, 3)),
    )
    rgb = np.vstack(
        (
            np.zeros((1, 3)),
            np.ones((1, 3)),
            rng.random((256, 3)),
        )
    )
    output = operator.apply(rgb)
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    assert np.array_equal(output[:2], rgb[:2])
    neutral = np.repeat(np.linspace(0.0, 1.0, 23)[:, None], 3, axis=1)
    assert np.array_equal(operator.apply(neutral), operator.base.apply(neutral))


def test_log_odds_fit_recovers_known_local_transform() -> None:
    rng = np.random.default_rng(20260801)
    source = rng.uniform(0.02, 0.98, size=(256, 3))
    centers = fixed_cube_centers((0.25, 0.75))
    truth = FixedNeutralGaussianLogOddsOperator(
        base=_base(),
        centers=centers,
        sigma=0.35,
        coefficients=rng.normal(0.0, 0.4, size=(8, 3)),
    )
    target = truth.apply(source)
    fitted = fit_fixed_neutral_gaussian_log_odds(
        _base(),
        source,
        target,
        centers=centers,
        sigma=0.35,
        ridge=1e-8,
    )
    base_rmse = float(np.sqrt(np.mean((_base().apply(source) - target) ** 2)))
    fit_rmse = float(np.sqrt(np.mean((fitted.apply(source) - target) ** 2)))
    assert fit_rmse < base_rmse * 1e-4
    replay = FixedNeutralGaussianLogOddsOperator.from_dict(fitted.to_dict())
    assert np.array_equal(replay.apply(source), fitted.apply(source))
