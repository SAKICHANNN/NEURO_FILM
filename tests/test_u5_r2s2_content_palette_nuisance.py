from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.content_palette_nuisance import (
    content_residual_features,
    density_ratio_velocity_grid,
    fit_bounded_multi_output_ridge,
)


def test_content_residual_features_are_canonical_and_finite() -> None:
    styled = np.array([0.1, 0.2, 0.3, 0.4])
    neutral = styled[::-1]
    features = content_residual_features(styled, neutral)
    assert features.shape == (12,)
    assert np.all(np.isfinite(features))
    assert np.array_equal(features[8:], features[:4] - features[4:8])


def test_density_ratio_identity_histograms_produce_identity_grid() -> None:
    histogram = np.full(8, 1.0 / 8.0)
    grid = density_ratio_velocity_grid(
        histogram,
        histogram,
        histogram_axis_size=2,
        bandwidth=0.12,
        velocity_grid_axis_size=4,
        score_difference_scale=0.5,
        coefficient_vector_norm_cap=2.0,
    )
    assert np.array_equal(grid, np.zeros_like(grid))


def test_bounded_ridge_fits_training_only_mapping_and_projects_norms() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(24, 9))
    mapping = rng.normal(size=(9, 24))
    flat = features @ mapping
    targets = flat.reshape(24, 2, 2, 2, 3)
    model = fit_bounded_multi_output_ridge(
        features, targets, alpha=0.1, maximum_vector_norm=0.75
    )
    prediction = model.predict(features)
    assert prediction.shape == targets.shape
    assert np.max(np.linalg.norm(prediction, axis=-1)) <= 0.75 + 1e-12
    assert np.all(np.isfinite(prediction))


def test_ridge_prediction_is_exactly_row_permutation_equivariant() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(12, 6))
    targets = rng.normal(size=(12, 2, 2, 2, 3))
    model = fit_bounded_multi_output_ridge(
        features, targets, alpha=1.0, maximum_vector_norm=2.0
    )
    order = np.array([5, 2, 9, 0, 1, 3, 4, 6, 7, 8, 10, 11])
    assert np.array_equal(
        model.predict(features[order]), model.predict(features)[order]
    )


def test_invalid_residual_and_ridge_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        content_residual_features(
            np.array([0.5, 0.5]), np.array([0.2, 0.2])
        )
    with pytest.raises(ValueError):
        fit_bounded_multi_output_ridge(
            np.zeros((1, 4)),
            np.zeros((1, 2, 2, 2, 3)),
            alpha=1.0,
            maximum_vector_norm=2.0,
        )
