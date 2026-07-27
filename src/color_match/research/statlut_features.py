"""Spatially invariant Lab statistics for explicit-LUT research.

This is an independent, compact reproduction of the feature *idea* described
by StatLUT, not its unpublished model or weights.  It extracts luminance,
joint chroma, and chroma-conditioned luminance statistics without semantic or
spatial features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab

from ..contracts import ReferenceMatchContractError


STATLUT_FEATURE_SCHEMA_ID = "neuro-film.statlut-lab-features.v1"


@dataclass(frozen=True)
class StatLUTFeaturePolicy:
    """Frozen feature resolution and chroma-stretch parameters."""

    luminance_bins: int = 32
    chroma_bins: int = 8
    chroma_scale: float = 128.0
    chroma_gamma: float = 0.5
    epsilon: float = 1e-12


@dataclass(frozen=True)
class StatLUTLabFeatures:
    """One spatially invariant Lab feature bundle."""

    schema_id: str
    policy: StatLUTFeaturePolicy
    luminance_histogram: np.ndarray
    chroma_histogram_sqrt: np.ndarray
    conditional_luminance: np.ndarray
    pixel_count: int

    def vector(self) -> np.ndarray:
        """Return the frozen L, ab, then L|ab concatenation order."""

        return np.concatenate(
            (
                self.luminance_histogram.reshape(-1),
                self.chroma_histogram_sqrt.reshape(-1),
                self.conditional_luminance.reshape(-1),
            )
        )


def _validate_policy(policy: StatLUTFeaturePolicy) -> None:
    if not isinstance(policy, StatLUTFeaturePolicy):
        raise ReferenceMatchContractError(
            "StatLUT feature policy must be StatLUTFeaturePolicy"
        )
    for label, value, minimum, maximum in (
        ("luminance_bins", policy.luminance_bins, 8, 256),
        ("chroma_bins", policy.chroma_bins, 4, 32),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise ReferenceMatchContractError(
                f"StatLUT {label} must be in [{minimum}, {maximum}]"
            )
    for label, value in (
        ("chroma_scale", policy.chroma_scale),
        ("chroma_gamma", policy.chroma_gamma),
        ("epsilon", policy.epsilon),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ReferenceMatchContractError(
                f"StatLUT {label} must be finite and positive"
            )
    if policy.chroma_gamma >= 1.0:
        raise ReferenceMatchContractError(
            "StatLUT chroma_gamma must be below one"
        )


def _validate_pixels(pixels: np.ndarray) -> np.ndarray:
    if not isinstance(pixels, np.ndarray):
        raise ReferenceMatchContractError(
            "StatLUT pixels must be a numpy array"
        )
    values = np.asarray(pixels, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or values.size < 64 * 3
        or not np.isfinite(values).all()
        or float(np.min(values)) < 0.0
        or float(np.max(values)) > 1.0
    ):
        raise ReferenceMatchContractError(
            "StatLUT pixels must be finite HxWx3 in [0, 1] with >=64 pixels"
        )
    return values


def _soft_coordinates(
    values: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = np.asarray(values, dtype=np.float64) * float(bins - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, bins - 1)
    upper_weight = scaled - lower
    return lower, upper, upper_weight


def extract_statlut_lab_features(
    pixels: np.ndarray,
    *,
    policy: StatLUTFeaturePolicy | None = None,
) -> StatLUTLabFeatures:
    """Extract soft-binned spatially invariant Lab statistics."""

    resolved = policy or StatLUTFeaturePolicy()
    _validate_policy(resolved)
    values = _validate_pixels(pixels)
    lab = linear_rgb_to_lab(values, working_space="linear_srgb").reshape(-1, 3)
    lightness = np.clip(
        lab[:, 0].astype(np.float64) / 100.0,
        0.0,
        1.0,
    )
    chroma = lab[:, 1:].astype(np.float64)
    stretched = np.sign(chroma) * np.power(
        np.abs(chroma) / float(resolved.chroma_scale),
        float(resolved.chroma_gamma),
    )
    stretched = np.clip(0.5 * (stretched + 1.0), 0.0, 1.0)
    pixel_count = len(lightness)

    l0, l1, lw1 = _soft_coordinates(lightness, resolved.luminance_bins)
    luminance_histogram = np.zeros(
        resolved.luminance_bins,
        dtype=np.float64,
    )
    np.add.at(luminance_histogram, l0, 1.0 - lw1)
    np.add.at(luminance_histogram, l1, lw1)
    luminance_histogram /= float(pixel_count)

    a0, a1, aw1 = _soft_coordinates(
        stretched[:, 0],
        resolved.chroma_bins,
    )
    b0, b1, bw1 = _soft_coordinates(
        stretched[:, 1],
        resolved.chroma_bins,
    )
    chroma_weights = np.zeros(
        (resolved.chroma_bins, resolved.chroma_bins),
        dtype=np.float64,
    )
    lightness_weights = np.zeros_like(chroma_weights)
    for ai, aw in ((a0, 1.0 - aw1), (a1, aw1)):
        for bi, bw in ((b0, 1.0 - bw1), (b1, bw1)):
            weight = aw * bw
            np.add.at(chroma_weights, (ai, bi), weight)
            np.add.at(
                lightness_weights,
                (ai, bi),
                weight * lightness,
            )
    normalized_chroma = chroma_weights / float(pixel_count)
    chroma_histogram_sqrt = np.sqrt(
        normalized_chroma + float(resolved.epsilon)
    )
    conditional_luminance = lightness_weights / (
        chroma_weights + float(resolved.epsilon)
    )
    result = StatLUTLabFeatures(
        schema_id=STATLUT_FEATURE_SCHEMA_ID,
        policy=resolved,
        luminance_histogram=luminance_histogram,
        chroma_histogram_sqrt=chroma_histogram_sqrt,
        conditional_luminance=conditional_luminance,
        pixel_count=pixel_count,
    )
    validate_statlut_lab_features(result)
    return result


def validate_statlut_lab_features(features: StatLUTLabFeatures) -> None:
    """Fail closed on feature shape, normalization or finite-value drift."""

    if not isinstance(features, StatLUTLabFeatures):
        raise ReferenceMatchContractError(
            "StatLUT features must be StatLUTLabFeatures"
        )
    if features.schema_id != STATLUT_FEATURE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported StatLUT feature schema"
        )
    _validate_policy(features.policy)
    l_bins = features.policy.luminance_bins
    c_bins = features.policy.chroma_bins
    arrays = (
        (
            "luminance_histogram",
            features.luminance_histogram,
            (l_bins,),
        ),
        (
            "chroma_histogram_sqrt",
            features.chroma_histogram_sqrt,
            (c_bins, c_bins),
        ),
        (
            "conditional_luminance",
            features.conditional_luminance,
            (c_bins, c_bins),
        ),
    )
    for label, array, shape in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or not np.isfinite(array).all()
            or np.any(array < 0.0)
        ):
            raise ReferenceMatchContractError(
                f"StatLUT {label} is invalid"
            )
    if (
        not np.isclose(
            np.sum(features.luminance_histogram),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )
        or np.any(features.conditional_luminance > 1.0 + 1e-12)
        or isinstance(features.pixel_count, bool)
        or not isinstance(features.pixel_count, int)
        or features.pixel_count < 64
    ):
        raise ReferenceMatchContractError(
            "StatLUT feature normalization/count is invalid"
        )
    expected_size = l_bins + 2 * c_bins * c_bins
    vector = features.vector()
    if vector.shape != (expected_size,) or not np.isfinite(vector).all():
        raise ReferenceMatchContractError(
            "StatLUT feature vector layout is invalid"
        )


def aggregate_statlut_features(
    features: tuple[StatLUTLabFeatures, ...] | list[StatLUTLabFeatures],
) -> np.ndarray:
    """Return an equal-image aggregate for one complete source batch."""

    if not isinstance(features, (tuple, list)) or not features:
        raise ReferenceMatchContractError(
            "StatLUT aggregation requires a non-empty feature sequence"
        )
    first = features[0]
    validate_statlut_lab_features(first)
    vectors = []
    for item in features:
        validate_statlut_lab_features(item)
        if item.policy != first.policy:
            raise ReferenceMatchContractError(
                "StatLUT aggregate policies must match"
            )
        vectors.append(item.vector())
    return np.mean(np.stack(vectors), axis=0, dtype=np.float64)


__all__ = [
    "STATLUT_FEATURE_SCHEMA_ID",
    "StatLUTFeaturePolicy",
    "StatLUTLabFeatures",
    "aggregate_statlut_features",
    "extract_statlut_lab_features",
    "validate_statlut_lab_features",
]
