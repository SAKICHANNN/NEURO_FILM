"""Bounded real-image composition for the clean-room Boolean grain primitive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .boolean_grain import (
    BooleanGrainContext,
    build_boolean_grain_context,
    render_boolean_grain,
)


_LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _linear_rgb(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)
    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or not rgb.size
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ValueError("linear RGB must be a non-empty finite HxWx3 array in [0,1]")
    return rgb


def linear_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Return D65 linear-sRGB relative luminance."""

    return _linear_rgb(linear_rgb) @ _LUMA_WEIGHTS


def resize_float_plane(
    plane: np.ndarray,
    output_shape: tuple[int, int],
    *,
    resample: Image.Resampling,
) -> np.ndarray:
    """Resize one finite float plane without quantizing through integer pixels."""

    values = np.asarray(plane, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("plane must be a non-empty finite 2D array")
    height, width = (int(value) for value in output_shape)
    if height <= 0 or width <= 0:
        raise ValueError("output shape must be positive")
    image = Image.fromarray(values, mode="F")
    resized = np.asarray(
        image.resize((width, height), resample=resample),
        dtype=np.float64,
    )
    return resized


def signed_headroom_composite(
    linear_rgb: np.ndarray,
    *,
    grain_luminance: np.ndarray,
    reference_luminance: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply a signed luma residual while preserving the RGB unit interval."""

    rgb = _linear_rgb(linear_rgb)
    grain = np.asarray(grain_luminance, dtype=np.float64)
    reference = np.asarray(reference_luminance, dtype=np.float64)
    if (
        grain.shape != rgb.shape[:2]
        or reference.shape != rgb.shape[:2]
        or not np.all(np.isfinite(grain))
        or not np.all(np.isfinite(reference))
        or np.any(grain < 0.0)
        or np.any(grain > 1.0)
        or np.any(reference < 0.0)
        or np.any(reference > 1.0)
    ):
        raise ValueError("grain/reference luminance must be finite HxW data in [0,1]")
    if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("strength must be finite and in [0,1]")
    delta = float(strength) * (grain - reference)
    positive = np.maximum(delta, 0.0)[..., None]
    negative = np.minimum(delta, 0.0)[..., None]
    output = rgb + positive * (1.0 - rgb) + negative * rgb
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise RuntimeError("signed headroom composition escaped the unit interval")
    output = output.astype(np.float64, copy=False)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class BooleanGrainCompositeResult:
    linear_rgb: np.ndarray
    grain_luminance: np.ndarray
    reference_luminance: np.ndarray
    context: BooleanGrainContext


def render_boolean_grain_composite(
    linear_rgb: np.ndarray,
    *,
    input_shape: tuple[int, int],
    output_zoom: int,
    radius_input_pixels: float,
    luma_residual_strength: float,
    monte_carlo_samples: int,
    gaussian_filter_sigma_output_pixels: float,
    maximum_input_intensity: float,
    epsilon: float,
    seed: int,
) -> BooleanGrainCompositeResult:
    """Render and compose one fixed Boolean-grain policy."""

    rgb = _linear_rgb(linear_rgb)
    input_height, input_width = (int(value) for value in input_shape)
    if (
        input_height <= 0
        or input_width <= 0
        or rgb.shape[:2]
        != (input_height * int(output_zoom), input_width * int(output_zoom))
    ):
        raise ValueError("input shape and output zoom do not match linear RGB")
    luminance = linear_luminance(rgb)
    low_luminance = resize_float_plane(
        luminance,
        (input_height, input_width),
        resample=Image.Resampling.BOX,
    )
    model_intensity = np.minimum(low_luminance, float(maximum_input_intensity))
    context = build_boolean_grain_context(
        model_intensity,
        radius_input_pixels=float(radius_input_pixels),
        monte_carlo_samples=int(monte_carlo_samples),
        gaussian_filter_sigma_output_pixels=float(
            gaussian_filter_sigma_output_pixels
        ),
        maximum_input_intensity=float(maximum_input_intensity),
        epsilon=float(epsilon),
        seed=int(seed),
    )
    grain = render_boolean_grain(context, output_zoom=int(output_zoom))
    reference = resize_float_plane(
        low_luminance,
        rgb.shape[:2],
        resample=Image.Resampling.BILINEAR,
    )
    output = signed_headroom_composite(
        rgb,
        grain_luminance=grain,
        reference_luminance=reference,
        strength=float(luma_residual_strength),
    )
    return BooleanGrainCompositeResult(
        linear_rgb=output,
        grain_luminance=grain,
        reference_luminance=reference,
        context=context,
    )


__all__ = [
    "BooleanGrainCompositeResult",
    "linear_luminance",
    "render_boolean_grain_composite",
    "resize_float_plane",
    "signed_headroom_composite",
]
