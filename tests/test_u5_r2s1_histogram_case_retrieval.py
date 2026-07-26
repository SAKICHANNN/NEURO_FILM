from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.histogram_case_retrieval import (
    HistogramCaseBank,
    canonical_rgb_histogram,
    generate_synthetic_palette,
    histogram_kde_velocity_grid,
    sample_palette,
)
from src.roll2film.palette_score_flow import palette_score_velocity_grid


def _bank() -> HistogramCaseBank:
    histograms = np.array(
        [
            [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.7, 0.0, 0.0],
        ]
    )
    grids = np.stack(
        (
            np.full((2, 2, 2, 3), 0.25),
            np.full((2, 2, 2, 3), -0.5),
        )
    )
    return HistogramCaseBank(histograms, grids)


def test_histogram_is_exactly_spatially_permutation_invariant() -> None:
    samples = np.random.default_rng(5).uniform(size=(8192, 3))
    shuffled = samples[np.random.default_rng(6).permutation(len(samples))]
    assert np.array_equal(
        canonical_rgb_histogram(samples, axis_size=8),
        canonical_rgb_histogram(shuffled, axis_size=8),
    )


def test_hard_retrieval_and_sparse_blend_are_deterministic() -> None:
    bank = _bank()
    query = bank.histograms[0]
    for distance in ("hellinger", "jensen_shannon"):
        index, grid = bank.hard_retrieve(query, distance=distance)
        assert index == 0
        assert np.array_equal(grid, bank.velocity_grids[0])
    indices, blend = bank.inverse_distance_blend(
        query, distance="hellinger", neighbors=2, epsilon=1e-12
    )
    assert np.array_equal(indices, np.array([0, 1]))
    assert np.array_equal(blend, bank.velocity_grids[0])


def test_histogram_kde_grid_is_finite_and_vector_bounded() -> None:
    histogram = np.zeros(8)
    histogram[[1, 6]] = [0.4, 0.6]
    grid = histogram_kde_velocity_grid(
        histogram,
        histogram_axis_size=2,
        bandwidth=0.12,
        velocity_grid_axis_size=4,
        coefficient_vector_norm_cap=2.0,
    )
    assert grid.shape == (4, 4, 4, 3)
    assert np.all(np.isfinite(grid))
    assert np.max(np.linalg.norm(grid, axis=-1)) < 2.0


def test_synthetic_palette_and_samples_replay_exactly() -> None:
    kwargs = {
        "component_counts": [2, 3, 4],
        "weight_dirichlet_alpha": 1.5,
        "mean_minimum": 0.06,
        "mean_maximum": 0.94,
        "standard_deviation_minimum": 0.09,
        "standard_deviation_maximum": 0.24,
    }
    left = generate_synthetic_palette(np.random.default_rng(10), **kwargs)
    right = generate_synthetic_palette(np.random.default_rng(10), **kwargs)
    assert np.array_equal(left.weights, right.weights)
    assert np.array_equal(left.means, right.means)
    assert np.array_equal(
        sample_palette(left, sample_count=128, rng=np.random.default_rng(11)),
        sample_palette(right, sample_count=128, rng=np.random.default_rng(11)),
    )
    grid = palette_score_velocity_grid(
        left, axis_size=4, coefficient_vector_norm_cap=2.0
    )
    assert np.max(np.linalg.norm(grid, axis=-1)) < 2.0


def test_invalid_histogram_and_bank_fail_closed() -> None:
    with pytest.raises(ValueError):
        canonical_rgb_histogram(np.array([[1.2, 0.0, 0.0]]), axis_size=8)
    with pytest.raises(ValueError):
        HistogramCaseBank(
            np.array([[0.2, 0.2]]), np.zeros((1, 2, 2, 2, 3))
        )
    with pytest.raises(ValueError):
        _bank().hard_retrieve(
            np.full(8, 1.0 / 8.0), distance="not-a-distance"  # type: ignore[arg-type]
        )
