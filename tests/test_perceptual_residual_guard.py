from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.perceptual_residual_guard import (
    apply_target_perceptual_residual_guard,
)


KWARGS = {
    "lightness_strength": 0.15,
    "chroma_strength": 0.35,
    "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
    "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
}


def test_identity_is_exact_including_cube_endpoints() -> None:
    source = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.8]]]
    )
    result = apply_target_perceptual_residual_guard(
        source, source, **KWARGS
    )
    assert np.array_equal(result.output, source)
    assert np.all(result.lightness_scale == 1.0)
    assert np.all(result.chroma_scale == 1.0)


def test_lightness_only_request_preserves_neutral_axis() -> None:
    source = np.full((2, 3, 3), 0.4)
    target = np.full_like(source, 0.2)
    result = apply_target_perceptual_residual_guard(
        source, target, **KWARGS
    )
    assert np.max(np.ptp(result.output, axis=-1)) < 2e-6


def test_requested_perceptual_stages_stay_inside_source_inclusive_rails() -> None:
    source = np.asarray(
        [[[0.0, 0.5, 1.0], [0.0001, 0.9999, 0.25]]],
        dtype=np.float64,
    )
    target = 1.0 - source
    result = apply_target_perceptual_residual_guard(
        source,
        target,
        **{**KWARGS, "lightness_strength": 1.0, "chroma_strength": 2.0},
    )
    assert np.all(np.isfinite(result.output))
    assert np.all(result.output >= -1e-13)
    assert np.all(result.output <= 1.0 + 1e-13)
    assert np.all(
        (result.lightness_scale >= 0.0)
        & (result.lightness_scale <= 1.0)
    )
    assert np.all(
        (result.chroma_scale >= 0.0) & (result.chroma_scale <= 1.0)
    )


def test_invalid_inputs_fail_closed() -> None:
    source = np.full((1, 1, 3), 0.5)
    with pytest.raises(ValueError):
        apply_target_perceptual_residual_guard(
            source,
            source,
            **{**KWARGS, "lightness_strength": -0.1},
        )
    with pytest.raises(ValueError):
        apply_target_perceptual_residual_guard(
            source,
            np.full((1, 1, 3), np.nan),
            **KWARGS,
        )
