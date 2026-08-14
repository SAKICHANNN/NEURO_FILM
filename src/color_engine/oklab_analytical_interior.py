"""Analytical hue-direction-preserving ingress for extended linear Rec.2020."""

from __future__ import annotations

import numpy as np

from src.color_engine.oklch_local_minde import (
    linear_rec2020_to_oklab,
    oklab_to_linear_rec2020,
)


def _finite_rgb(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray):
        raise TypeError("pixels must be a numpy ndarray")
    if pixels.dtype != np.float32:
        raise TypeError("pixels must be float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must be HxWx3")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite")


def analytical_oklab_interior_rec2020(
    pixels: np.ndarray,
    *,
    softness: float = 1.0 / 64.0,
    margin: float = 2.0 / 65535.0,
    iterations: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Map only out-of-gamut pixels into the Rec.2020 RGB interior.

    The mapping first compresses OKLab lightness monotonically into the neutral
    Rec.2020 interval implied by ``margin``. It then preserves the original
    OKLab ``(a, b)`` direction and finds the largest feasible chroma scale by
    deterministic bisection. Pixels already in Rec.2020 are copied bit-exactly.
    No component clipping or post-hoc limiting is used.
    """

    _finite_rgb(pixels)
    if not np.isfinite(softness) or softness <= 0.0:
        raise ValueError("softness must be positive and finite")
    if not np.isfinite(margin) or margin <= 0.0 or margin >= 0.5:
        raise ValueError("margin must be finite and in (0, 0.5)")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    shape = pixels.shape
    source = pixels.reshape(-1, 3)
    source_in_gamut = np.all((source >= 0.0) & (source <= 1.0), axis=1)
    ratio = np.ones(source.shape[0], dtype=np.float32)
    if bool(np.all(source_in_gamut)):
        return pixels.copy(), ratio.reshape(shape[:2])

    indexes = np.flatnonzero(~source_in_gamut)
    origin = linear_rec2020_to_oklab(pixels).reshape(-1, 3)[indexes]
    lightness = origin[:, 0]
    chroma = np.hypot(origin[:, 1], origin[:, 2])
    direction = np.zeros((origin.shape[0], 2), dtype=np.float64)
    chromatic = chroma > 4e-12
    direction[chromatic] = origin[chromatic, 1:3] / chroma[chromatic, None]

    # Stable soft interval compression: softplus(L) - softplus(L - 1).
    normalized_lightness = softness * (
        np.logaddexp(0.0, lightness / softness)
        - np.logaddexp(0.0, (lightness - 1.0) / softness)
    )
    neutral_minimum = np.cbrt(margin)
    neutral_maximum = np.cbrt(1.0 - margin)
    mapped_lightness = neutral_minimum + (
        neutral_maximum - neutral_minimum
    ) * normalized_lightness

    minimum = np.zeros(origin.shape[0], dtype=np.float64)
    maximum = np.ones(origin.shape[0], dtype=np.float64)
    neutral = np.column_stack(
        (mapped_lightness, np.zeros_like(mapped_lightness), np.zeros_like(mapped_lightness))
    )
    neutral_rgb = oklab_to_linear_rec2020(neutral)
    if not bool(
        np.all(np.isfinite(neutral_rgb))
        and np.all(neutral_rgb >= margin)
        and np.all(neutral_rgb <= 1.0 - margin)
    ):
        raise RuntimeError("neutral OKLab lightness is not inside the RGB margin")

    for _ in range(iterations):
        current = (minimum + maximum) * 0.5
        candidate = np.column_stack(
            (
                mapped_lightness,
                direction[:, 0] * chroma * current,
                direction[:, 1] * chroma * current,
            )
        )
        candidate_rgb = oklab_to_linear_rec2020(candidate)
        candidate_in_gamut = np.all(
            (candidate_rgb >= margin) & (candidate_rgb <= 1.0 - margin), axis=1
        )
        minimum = np.where(candidate_in_gamut, current, minimum)
        maximum = np.where(candidate_in_gamut, maximum, current)

    mapped_oklab = np.column_stack(
        (
            mapped_lightness,
            direction[:, 0] * chroma * minimum,
            direction[:, 1] * chroma * minimum,
        )
    )
    mapped_rgb = oklab_to_linear_rec2020(mapped_oklab)
    if not bool(
        np.all(np.isfinite(mapped_rgb))
        and np.all(mapped_rgb >= margin)
        and np.all(mapped_rgb <= 1.0 - margin)
    ):
        raise RuntimeError("analytical OKLab mapping did not reach the RGB interior")

    output = source.astype(np.float64, copy=True)
    output[indexes] = mapped_rgb
    output_f32 = output.reshape(shape).astype(np.float32)
    if not bool(np.all(np.isfinite(output_f32))):
        raise RuntimeError("analytical OKLab mapping produced non-finite float32")
    if not np.array_equal(output_f32.reshape(-1, 3)[source_in_gamut], source[source_in_gamut]):
        raise RuntimeError("analytical OKLab mapping changed an in-gamut pixel")
    ratio[indexes] = minimum.astype(np.float32)
    return output_f32, ratio.reshape(shape[:2])


__all__ = ["analytical_oklab_interior_rec2020"]
