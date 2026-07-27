from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.research import (
    StatLUTFeaturePolicy,
    aggregate_statlut_features,
    extract_statlut_lab_features,
    validate_statlut_lab_features,
)


def _pixels(seed: int = 2026072704) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(24, 32, 3)).astype(np.float32)


def test_statlut_features_have_frozen_layout_and_normalization() -> None:
    policy = StatLUTFeaturePolicy(luminance_bins=32, chroma_bins=8)
    features = extract_statlut_lab_features(_pixels(), policy=policy)
    assert features.luminance_histogram.shape == (32,)
    assert features.chroma_histogram_sqrt.shape == (8, 8)
    assert features.conditional_luminance.shape == (8, 8)
    assert features.vector().shape == (160,)
    assert np.isclose(features.luminance_histogram.sum(), 1.0)
    assert np.isfinite(features.vector()).all()


def test_statlut_features_are_exactly_spatially_invariant() -> None:
    pixels = _pixels()
    flattened = pixels.reshape(-1, 3)
    permutation = np.random.default_rng(99).permutation(len(flattened))
    shuffled = flattened[permutation].reshape(pixels.shape)
    first = extract_statlut_lab_features(pixels)
    second = extract_statlut_lab_features(shuffled)
    assert np.allclose(
        first.vector(),
        second.vector(),
        atol=2e-15,
        rtol=0.0,
    )


def test_statlut_equal_image_aggregation_is_order_invariant() -> None:
    features = [
        extract_statlut_lab_features(_pixels(seed))
        for seed in (1, 2, 3)
    ]
    first = aggregate_statlut_features(features)
    second = aggregate_statlut_features(list(reversed(features)))
    assert np.allclose(first, second, atol=2e-15, rtol=0.0)


def test_statlut_features_fail_closed_on_invalid_inputs() -> None:
    with pytest.raises(ReferenceMatchContractError, match="finite"):
        extract_statlut_lab_features(
            np.full((8, 8, 3), np.nan, dtype=np.float32)
        )
    with pytest.raises(ReferenceMatchContractError, match="luminance_bins"):
        extract_statlut_lab_features(
            _pixels(),
            policy=StatLUTFeaturePolicy(luminance_bins=4),
        )
    features = extract_statlut_lab_features(_pixels())
    corrupted = replace(
        features,
        luminance_histogram=np.zeros_like(features.luminance_histogram),
    )
    with pytest.raises(ReferenceMatchContractError, match="normalization"):
        validate_statlut_lab_features(corrupted)


def test_statlut_neutral_axis_has_finite_conditional_statistics() -> None:
    neutral = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    pixels = np.repeat(neutral[:, None], 3, axis=1).reshape(16, 16, 3)
    features = extract_statlut_lab_features(pixels)
    assert np.isfinite(features.vector()).all()
    assert np.count_nonzero(features.chroma_histogram_sqrt > 1e-5) < 16
