"""Characteristic-response-bounded physical exposure residuals.

This module keeps a proposed spatial film exposure operator in the physical
exposure domain while limiting its downstream scan-transmittance excursion.
The same scalar is applied to all three exposure-layer residuals at a pixel,
so the candidate direction is preserved rather than channel-clipped.

The bound is a product-safety projection around an uncalibrated physical
candidate.  It is not a measured emulsion response or a replacement for
severe-artifact evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.roll2film.sensitometry import RGBSensitometryOperator


@dataclass(frozen=True)
class ResponseBoundedExposure:
    """One bounded exposure candidate and its per-pixel audit fields."""

    exposure: np.ndarray
    shared_scale: np.ndarray
    raw_max_transmittance_delta: np.ndarray
    soft_transmittance_limit: np.ndarray


def _exposure(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float64:
        raise TypeError(f"{name} must use float64")
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError(f"{name} must be finite nonnegative HxWx3 exposure")
    return array


def scan_transmittance(
    exposure: np.ndarray,
    sensitometry: RGBSensitometryOperator,
) -> np.ndarray:
    """Evaluate the monotone characteristic curves as scan transmittance."""

    values = _exposure(exposure, name="exposure")
    if not isinstance(sensitometry, RGBSensitometryOperator):
        raise TypeError("sensitometry must be RGBSensitometryOperator")
    return np.power(10.0, -sensitometry.apply(values))


def apply_response_bounded_exposure_residual(
    source_exposure: np.ndarray,
    candidate_exposure: np.ndarray,
    sensitometry: RGBSensitometryOperator,
    *,
    maximum_transmittance_delta: float,
) -> ResponseBoundedExposure:
    """Apply a smooth, shared-scale safety projection to an exposure residual.

    Let ``R = candidate - source`` and let ``m`` be the largest absolute
    scan-transmittance change produced by the unbounded candidate at a pixel.
    The desired response magnitude is ``c * tanh(m / c)``, where ``c`` is the
    configured maximum.  Monotonicity of each characteristic curve provides
    an exact exposure interval for that response magnitude.  The minimum
    admissible scale across the three layers is then applied to all layers.
    """

    source = _exposure(source_exposure, name="source_exposure")
    candidate = _exposure(candidate_exposure, name="candidate_exposure")
    if source.shape != candidate.shape:
        raise ValueError("source and candidate exposure shapes differ")
    if not isinstance(sensitometry, RGBSensitometryOperator):
        raise TypeError("sensitometry must be RGBSensitometryOperator")
    cap = float(maximum_transmittance_delta)
    if not math.isfinite(cap) or cap <= 0.0 or cap >= 1.0:
        raise ValueError("maximum_transmittance_delta must be finite in (0, 1)")

    residual = candidate - source
    source_density = sensitometry.apply(source)
    source_transmittance = np.power(10.0, -source_density)
    candidate_transmittance = np.power(
        10.0, -sensitometry.apply(candidate)
    )
    raw_max_delta = np.max(
        np.abs(candidate_transmittance - source_transmittance),
        axis=-1,
        keepdims=True,
    )
    soft_limit = cap * np.tanh(raw_max_delta / cap)

    zero_density = sensitometry.apply(np.zeros((1, 1, 3), dtype=np.float64))
    zero_transmittance = np.power(10.0, -zero_density)
    ratios = np.ones_like(source)

    positive = residual > 0.0
    positive_bounded = positive & (soft_limit < source_transmittance)
    positive_target_transmittance = np.where(
        positive_bounded,
        source_transmittance - soft_limit,
        source_transmittance,
    )
    positive_target_density = np.where(
        positive_bounded,
        -np.log10(positive_target_transmittance),
        source_density,
    )
    maximum_exposure = sensitometry.inverse(positive_target_density)
    ratios[positive_bounded] = (
        np.maximum(maximum_exposure - source, 0.0)[positive_bounded]
        / residual[positive_bounded]
    )

    negative = residual < 0.0
    available_transmittance_increase = (
        zero_transmittance - source_transmittance
    )
    negative_bounded = negative & (
        soft_limit < available_transmittance_increase
    )
    negative_target_transmittance = np.where(
        negative_bounded,
        source_transmittance + soft_limit,
        source_transmittance,
    )
    negative_target_density = np.where(
        negative_bounded,
        -np.log10(negative_target_transmittance),
        source_density,
    )
    minimum_exposure = sensitometry.inverse(negative_target_density)
    ratios[negative_bounded] = (
        np.maximum(source - minimum_exposure, 0.0)[negative_bounded]
        / (-residual[negative_bounded])
    )

    shared_scale = np.clip(
        np.min(ratios, axis=-1, keepdims=True), 0.0, 1.0
    )
    bounded = source + shared_scale * residual
    if not np.all(np.isfinite(bounded)) or np.any(bounded < 0.0):
        raise RuntimeError("bounded exposure left its physical domain")

    bounded_delta = np.max(
        np.abs(scan_transmittance(bounded, sensitometry) - source_transmittance),
        axis=-1,
        keepdims=True,
    )
    tolerance = 128.0 * np.finfo(np.float64).eps
    if float(np.max(bounded_delta)) > cap + tolerance:
        raise RuntimeError("bounded exposure exceeded its response limit")

    return ResponseBoundedExposure(
        exposure=bounded,
        shared_scale=shared_scale,
        raw_max_transmittance_delta=raw_max_delta,
        soft_transmittance_limit=soft_limit,
    )


__all__ = [
    "ResponseBoundedExposure",
    "apply_response_bounded_exposure_residual",
    "scan_transmittance",
]
