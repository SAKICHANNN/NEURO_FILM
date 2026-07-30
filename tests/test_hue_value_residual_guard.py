from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.hue_value_residual_guard import (
    _hsv_to_rgb,
    _rgb_to_hsv,
    apply_target_hue_value_residual_guard,
)


KWARGS = {
    "value_strength": 0.15,
    "hue_saturation_strength": 0.35,
    "neutral_saturation_floor": 1.0 / 64.0,
    "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
    "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
}


def test_hsv_roundtrip_is_exact_within_float64_tolerance() -> None:
    rng = np.random.default_rng(20260730)
    values = rng.random((5, 7, 3))
    assert np.max(np.abs(_hsv_to_rgb(_rgb_to_hsv(values)) - values)) < 1e-15


def test_identity_is_exact_including_neutral_and_endpoints() -> None:
    source = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.8]]]
    )
    result = apply_target_hue_value_residual_guard(
        source, source, **KWARGS
    )
    assert np.array_equal(result.output, source)
    assert np.all(result.value_scale == 1.0)
    assert np.all(result.hue_saturation_scale == 1.0)


def test_colour_emerging_from_neutral_uses_target_hue() -> None:
    source = np.full((2, 3, 3), 0.4)
    target = np.zeros_like(source)
    target[..., 2] = 0.4
    result = apply_target_hue_value_residual_guard(
        source, target, **KWARGS
    )
    assert np.all(result.output[..., 2] > result.output[..., 0])
    assert np.all(result.output[..., 2] > result.output[..., 1])


def test_extreme_request_stays_inside_source_inclusive_rails() -> None:
    source = np.asarray(
        [[[0.0, 0.5, 1.0], [0.0001, 0.9999, 0.25]]]
    )
    result = apply_target_hue_value_residual_guard(
        source,
        1.0 - source,
        **{
            **KWARGS,
            "value_strength": 1.0,
            "hue_saturation_strength": 1.0,
        },
    )
    assert np.all(np.isfinite(result.output))
    assert np.all(result.output >= -1e-14)
    assert np.all(result.output <= 1.0 + 1e-14)


def test_invalid_saturation_floor_fails_closed() -> None:
    source = np.full((1, 1, 3), 0.5)
    with pytest.raises(ValueError):
        apply_target_hue_value_residual_guard(
            source,
            source,
            **{**KWARGS, "neutral_saturation_floor": 0.0},
        )
