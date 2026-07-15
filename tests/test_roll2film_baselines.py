from __future__ import annotations

import json

import numpy as np

from src.roll2film.baselines import (
    fit_basic_adjustment_family,
    fit_per_channel_quantile_operator,
    fit_sliced_transport_operator,
    SlicedTransportOperator,
)
from src.roll2film.simulator import sample_neutral_prior


def _moment_error(source: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
        + np.linalg.norm(np.cov(source, rowvar=False) - np.cov(target, rowvar=False))
    )


def test_basic_adjustment_family_is_explicit_and_improves_moments() -> None:
    source = sample_neutral_prior(4096, seed=501)
    target = source * np.array([1.12, 0.96, 1.04]) + 0.02
    family = fit_basic_adjustment_family(source, target)

    assert set(family) == {
        "exposure_only",
        "white_balance_only",
        "contrast_only",
        "saturation_only",
        "wb_contrast_saturation",
    }
    assert _moment_error(family["wb_contrast_saturation"].apply(source), target) < _moment_error(
        source, target
    )
    assert all(operator.determinant > 0.0 for operator in family.values())


def test_quantile_operator_matches_channel_marginals_and_round_trips() -> None:
    source = sample_neutral_prior(8192, seed=502)
    target = np.sign(source) * np.abs(source) ** 1.15 + np.array([0.02, -0.01, 0.015])
    operator = fit_per_channel_quantile_operator(source, target)
    rendered = operator.apply(source)

    assert np.max(np.abs(operator.inverse(rendered) - source)) < 1e-10
    assert np.mean(np.abs(np.quantile(rendered, 0.5, axis=0) - np.quantile(target, 0.5, axis=0))) < 1e-6


def test_sliced_transport_is_deterministic_invertible_and_improves_distribution() -> None:
    source = sample_neutral_prior(4096, seed=503)
    matrix = np.array([[1.08, -0.08, 0.03], [0.04, 0.94, 0.05], [-0.03, 0.07, 1.02]])
    target = source @ matrix.T + np.array([0.01, -0.015, 0.02])
    first = fit_sliced_transport_operator(source, target, iterations=4, seed=9)
    second = fit_sliced_transport_operator(source, target, iterations=4, seed=9)
    rendered = first.apply(source)
    replay = SlicedTransportOperator.from_dict(json.loads(json.dumps(first.to_dict())))

    assert first.to_dict() == second.to_dict()
    assert np.max(np.abs(first.inverse(rendered) - source)) < 1e-9
    assert np.array_equal(replay.apply(source), rendered)
    assert _moment_error(rendered, target) < _moment_error(source, target) * 0.25
