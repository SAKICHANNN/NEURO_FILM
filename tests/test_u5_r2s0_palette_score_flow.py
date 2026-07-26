from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.palette_score_flow import (
    DiagonalGaussianMixturePalette,
    palette_score_operator,
    palette_score_velocity_grid,
)


def _palette() -> DiagonalGaussianMixturePalette:
    return DiagonalGaussianMixturePalette(
        weights=np.array([0.6, 0.4]),
        means=np.array([[0.25, 0.35, 0.45], [0.75, 0.65, 0.55]]),
        standard_deviations=np.array(
            [[0.18, 0.16, 0.14], [0.15, 0.17, 0.19]]
        ),
    )


def test_analytic_score_matches_log_density_finite_difference() -> None:
    palette = _palette()
    points = np.array([[0.2, 0.4, 0.6], [0.7, 0.3, 0.5]])
    step = 1e-6
    columns = []
    for channel in range(3):
        offset = np.zeros(3)
        offset[channel] = step
        columns.append(
            (
                palette.log_density(points + offset)
                - palette.log_density(points - offset)
            )
            / (2.0 * step)
        )
    numeric = np.stack(columns, axis=-1)
    assert np.max(np.abs(palette.score(points) - numeric)) < 1e-8


def test_velocity_coefficients_obey_vector_norm_cap() -> None:
    grid = palette_score_velocity_grid(
        _palette(), axis_size=4, coefficient_vector_norm_cap=2.0
    )
    assert grid.shape == (4, 4, 4, 3)
    assert np.max(np.linalg.norm(grid, axis=-1)) < 2.0


def test_operator_is_bounded_invertible_and_replayable() -> None:
    operator = palette_score_operator(
        _palette(),
        axis_size=4,
        integration_steps=24,
        coefficient_vector_norm_cap=1.5,
    )
    points = np.random.default_rng(91).uniform(0.02, 0.98, size=(43, 3))
    output = operator.apply(points)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.max(np.abs(operator.inverse(output) - points)) < 1e-5
    replay = type(operator).from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(replay.apply(points), output)


def test_distinct_palettes_produce_distinct_operators() -> None:
    first = _palette()
    second = DiagonalGaussianMixturePalette(
        weights=np.array([0.6, 0.4]),
        means=1.0 - first.means,
        standard_deviations=first.standard_deviations,
    )
    left = palette_score_operator(
        first,
        axis_size=4,
        integration_steps=16,
        coefficient_vector_norm_cap=1.0,
    )
    right = palette_score_operator(
        second,
        axis_size=4,
        integration_steps=16,
        coefficient_vector_norm_cap=1.0,
    )
    points = np.random.default_rng(92).uniform(size=(31, 3))
    assert np.sqrt(np.mean((left.apply(points) - right.apply(points)) ** 2)) > 0.01


def test_invalid_palette_and_cap_fail_closed() -> None:
    with pytest.raises(ValueError):
        DiagonalGaussianMixturePalette(
            weights=np.array([0.2]),
            means=np.array([[0.5, 0.5, 0.5]]),
            standard_deviations=np.array([[0.1, 0.1, 0.1]]),
        )
    with pytest.raises(ValueError):
        palette_score_velocity_grid(
            _palette(), axis_size=1, coefficient_vector_norm_cap=1.0
        )
    with pytest.raises(ValueError):
        palette_score_velocity_grid(
            _palette(), axis_size=3, coefficient_vector_norm_cap=0.0
        )
