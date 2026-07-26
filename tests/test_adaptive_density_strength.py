from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.adaptive_density_strength import (
    apply_strength_preflight,
    linear_to_srgb8,
    new_hard_clipping_fraction_rgb8,
    render_strength_rgb8,
    srgb8_to_linear,
)


class _IdentityOperator:
    def apply(self, linear_rgb: np.ndarray, *, strength: float) -> np.ndarray:
        return linear_rgb.copy()


class _EndpointOperator:
    def __init__(self, endpoint_pixels: int) -> None:
        self.endpoint_pixels = endpoint_pixels

    def apply(self, linear_rgb: np.ndarray, *, strength: float) -> np.ndarray:
        result = linear_rgb.copy()
        if strength > 0.6:
            result.reshape(-1, 3)[: self.endpoint_pixels] = 1.0
        return result


def test_srgb8_roundtrip_is_exact() -> None:
    values = np.arange(256, dtype=np.uint8)
    rgb = np.stack(np.meshgrid(values[::17], values[::23], indexing="ij"), axis=-1)
    rgb = np.concatenate((rgb, rgb[..., :1]), axis=-1)
    np.testing.assert_array_equal(linear_to_srgb8(srgb8_to_linear(rgb)), rgb)


def test_identity_preflight_reuses_challenger_bytes() -> None:
    source = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    result = apply_strength_preflight(
        source,
        _IdentityOperator(),
        challenger_strength=0.65,
        baseline_strength=0.5,
        epsilon=1.0 / 510.0,
        maximum_new_hard_clipping_fraction=0.005,
    )
    assert not result.fallback_applied
    assert result.selected_strength == 0.65
    assert result.selected_rgb8 is result.challenger_rgb8
    np.testing.assert_array_equal(result.selected_rgb8, source)


def test_threshold_equality_selects_challenger() -> None:
    source = np.full((10, 10, 3), 128, dtype=np.uint8)
    result = apply_strength_preflight(
        source,
        _EndpointOperator(1),
        challenger_strength=0.65,
        baseline_strength=0.5,
        epsilon=1.0 / 510.0,
        maximum_new_hard_clipping_fraction=0.01,
    )
    assert result.challenger_new_hard_clipping_fraction == 0.01
    assert not result.fallback_applied


def test_above_threshold_hard_falls_back() -> None:
    source = np.full((10, 10, 3), 128, dtype=np.uint8)
    result = apply_strength_preflight(
        source,
        _EndpointOperator(2),
        challenger_strength=0.65,
        baseline_strength=0.5,
        epsilon=1.0 / 510.0,
        maximum_new_hard_clipping_fraction=0.01,
    )
    assert result.fallback_applied
    assert result.selected_strength == 0.5
    np.testing.assert_array_equal(result.selected_rgb8, source)


def test_helpers_fail_closed_on_invalid_rgb_or_policy() -> None:
    with pytest.raises(ValueError, match="uint8"):
        srgb8_to_linear(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        new_hard_clipping_fraction_rgb8(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((3, 2, 3), dtype=np.uint8),
            epsilon=0.0,
        )
    with pytest.raises(ValueError, match="below challenger"):
        apply_strength_preflight(
            np.zeros((2, 2, 3), dtype=np.uint8),
            _IdentityOperator(),
            challenger_strength=0.5,
            baseline_strength=0.5,
            epsilon=0.0,
            maximum_new_hard_clipping_fraction=0.0,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        linear_to_srgb8(np.full((2, 2, 3), 1.1))


def test_render_strength_rejects_invalid_strength() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        render_strength_rgb8(
            np.zeros((2, 2, 3), dtype=np.uint8),
            _IdentityOperator(),
            strength=1.1,
        )
