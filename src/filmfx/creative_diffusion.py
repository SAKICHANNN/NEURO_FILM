"""Bounded multi-scale optical diffusion, separate from film halation.

This module models a downstream creative/lens diffusion effect as a convex
mixture of the unmodified signal and positive Gaussian scatter.  It is not an
emulsion backscatter model: there is no film-layer tint, density response,
threshold mask, or screen blend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .fast_blur import gaussian_filter_direct


CREATIVE_DIFFUSION_VERSION = "creative-optical-diffusion-v1"


@dataclass(frozen=True, slots=True)
class CreativeDiffusionProfile:
    """A fixed positive PSF mixture in pixel coordinates."""

    profile_id: str
    sigmas_px: tuple[float, ...]
    scatter_fractions: tuple[float, ...]


def validate_creative_diffusion_profile(
    profile: CreativeDiffusionProfile,
) -> CreativeDiffusionProfile:
    if not profile.profile_id or not profile.profile_id.isascii():
        raise ValueError("profile_id must be non-empty ASCII")
    if len(profile.sigmas_px) != len(profile.scatter_fractions):
        raise ValueError("sigma and scatter-fraction counts differ")
    if not 1 <= len(profile.sigmas_px) <= 8:
        raise ValueError("creative diffusion requires one to eight scales")

    previous_sigma = 0.0
    total_scatter = 0.0
    for sigma, fraction in zip(
        profile.sigmas_px, profile.scatter_fractions, strict=True
    ):
        if not math.isfinite(sigma) or sigma <= previous_sigma:
            raise ValueError("sigmas must be finite, positive, and increasing")
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError("scatter fractions must be finite and nonnegative")
        previous_sigma = sigma
        total_scatter += fraction
    if total_scatter > 0.35 + 1e-12:
        raise ValueError("total creative diffusion scatter exceeds 0.35")
    return profile


def apply_creative_diffusion_linear(
    linear_rgb: np.ndarray,
    profile: CreativeDiffusionProfile,
) -> np.ndarray:
    """Apply a bounded positive multi-scale scatter in relative linear RGB.

    Because the source and every blurred term have nonnegative weights summing
    to one, the operator cannot create a value outside the source component
    range.  It deliberately does not clip the result.
    """

    source = np.asarray(linear_rgb)
    if source.dtype != np.float32:
        raise ValueError("creative diffusion requires float32 input")
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("creative diffusion requires HxWx3 input")
    if min(source.shape[:2]) < 2:
        raise ValueError("creative diffusion requires at least 2x2 pixels")
    if not np.isfinite(source).all():
        raise ValueError("creative diffusion requires finite input")
    if float(source.min()) < 0.0:
        raise ValueError("creative diffusion requires nonnegative linear RGB")

    checked = validate_creative_diffusion_profile(profile)
    scatter_total = float(sum(checked.scatter_fractions))
    output = source * np.float32(1.0 - scatter_total)
    for sigma, fraction in zip(
        checked.sigmas_px, checked.scatter_fractions, strict=True
    ):
        if fraction == 0.0:
            continue
        blurred = gaussian_filter_direct(
            source,
            sigma=(float(sigma), float(sigma), 0.0),
            truncate=4.0,
        )
        output = output + blurred * np.float32(fraction)
    return output.astype(np.float32, copy=False)


__all__ = [
    "CREATIVE_DIFFUSION_VERSION",
    "CreativeDiffusionProfile",
    "apply_creative_diffusion_linear",
    "validate_creative_diffusion_profile",
]
