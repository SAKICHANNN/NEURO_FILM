from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.filmfx.boolean_grain_composite import (
    linear_luminance,
    render_boolean_grain_composite,
    resize_float_plane,
    signed_headroom_composite,
)


def test_signed_headroom_is_identity_at_zero_strength_and_bounded() -> None:
    rgb = np.linspace(0.0, 1.0, 4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
    grain = np.linspace(0.0, 1.0, 20, dtype=np.float64).reshape(4, 5)
    reference = np.flipud(grain)
    identity = signed_headroom_composite(
        rgb,
        grain_luminance=grain,
        reference_luminance=reference,
        strength=0.0,
    )
    assert np.array_equal(identity, rgb)
    output = signed_headroom_composite(
        rgb,
        grain_luminance=grain,
        reference_luminance=reference,
        strength=1.0,
    )
    assert float(np.min(output)) >= 0.0
    assert float(np.max(output)) <= 1.0
    assert not output.flags.writeable


def test_signed_headroom_matches_frozen_piecewise_equation() -> None:
    rgb = np.asarray([[[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]])
    output = signed_headroom_composite(
        rgb,
        grain_luminance=np.asarray([[0.8, 0.1]]),
        reference_luminance=np.asarray([[0.3, 0.6]]),
        strength=0.2,
    )
    expected_positive = rgb[0, 0] + 0.1 * (1.0 - rgb[0, 0])
    expected_negative = rgb[0, 1] - 0.1 * rgb[0, 1]
    np.testing.assert_allclose(output[0, 0], expected_positive, atol=1e-15)
    np.testing.assert_allclose(output[0, 1], expected_negative, atol=1e-15)


def test_composite_repeat_shape_and_context_are_exact() -> None:
    y, x = np.mgrid[:12, :16]
    rgb = np.stack(
        (
            0.05 + 0.5 * x / 15.0,
            0.1 + 0.6 * y / 11.0,
            np.full_like(x, 0.35, dtype=np.float64),
        ),
        axis=2,
    )
    parameters = {
        "input_shape": (6, 8),
        "output_zoom": 2,
        "radius_input_pixels": 0.38,
        "luma_residual_strength": 0.12,
        "monte_carlo_samples": 8,
        "gaussian_filter_sigma_output_pixels": 0.6,
        "maximum_input_intensity": 255.0 / 256.0,
        "epsilon": 1e-12,
        "seed": 17,
    }
    first = render_boolean_grain_composite(rgb, **parameters)
    second = render_boolean_grain_composite(rgb, **parameters)
    assert first.linear_rgb.shape == rgb.shape
    assert first.grain_luminance.shape == rgb.shape[:2]
    assert first.reference_luminance.shape == rgb.shape[:2]
    assert first.context.fingerprint() == second.context.fingerprint()
    assert np.array_equal(first.linear_rgb, second.linear_rgb)
    assert np.array_equal(first.grain_luminance, second.grain_luminance)
    assert np.isfinite(first.linear_rgb).all()


def test_resize_and_luminance_validate_inputs() -> None:
    rgb = np.ones((2, 3, 3), dtype=np.float64)
    np.testing.assert_allclose(linear_luminance(rgb), 1.0)
    resized = resize_float_plane(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        (4, 6),
        resample=Image.Resampling.BILINEAR,
    )
    assert resized.shape == (4, 6)
    with pytest.raises(ValueError):
        linear_luminance(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        signed_headroom_composite(
            rgb,
            grain_luminance=np.zeros((2, 3)),
            reference_luminance=np.zeros((2, 3)),
            strength=1.1,
        )
