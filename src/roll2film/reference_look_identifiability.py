"""Fixed reference-look descriptors and bounded explicit-operator predictors.

This module deliberately contains no image encoder and no RGB renderer.  It
summarizes unordered linear-sRGB observations, then retrieves or regresses an
already validated explicit velocity grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roll2film.content_palette_nuisance import (
    BoundedMultiOutputRidge,
    fit_bounded_multi_output_ridge,
)


_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_LUMA_QUANTILES = np.array(
    [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98],
    dtype=np.float64,
)
_STRATUM_EDGES = np.linspace(0.0, 1.0, 6, dtype=np.float64)


def _validate_rgb_rows(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < 16
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must contain at least 16 finite [0, 1] rows")
    return values


def _quantile_strata(luma: np.ndarray) -> list[np.ndarray]:
    """Return five deterministic, nearly equal-count luma strata."""

    order = np.argsort(luma, kind="stable")
    boundaries = np.rint(_STRATUM_EDGES * len(order)).astype(np.int64)
    boundaries[0] = 0
    boundaries[-1] = len(order)
    return [
        order[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
    ]


def _weighted_hue_moments(
    red_green: np.ndarray,
    yellow_blue: np.ndarray,
    chroma: np.ndarray,
) -> np.ndarray:
    total = float(np.sum(chroma))
    if total <= 1e-15:
        return np.zeros(4, dtype=np.float64)
    angle = np.arctan2(yellow_blue, red_green)
    weights = chroma / total
    return np.array(
        [
            np.sum(weights * np.cos(angle)),
            np.sum(weights * np.sin(angle)),
            np.sum(weights * np.cos(2.0 * angle)),
            np.sum(weights * np.sin(2.0 * angle)),
        ],
        dtype=np.float64,
    )


def factorized_reference_descriptor(rgb: np.ndarray) -> np.ndarray:
    """Summarize one unordered final reference observation.

    Luma quantiles retain global tone.  The remaining statistics use
    equal-count luma strata so that their weighting is not directly determined
    by the scene's luma histogram.  The descriptor is still only a candidate;
    W1 must measure and reject content/nuisance leakage.
    """

    values = _validate_rgb_rows(rgb)
    luma = values @ _LUMA_WEIGHTS
    log_luma = np.log2(np.maximum(luma, 2.0**-16))
    red_green = values[:, 0] - values[:, 1]
    yellow_blue = 0.5 * (values[:, 0] + values[:, 1]) - values[:, 2]
    chroma = np.sqrt(red_green**2 + yellow_blue**2)

    parts = [np.quantile(log_luma, _LUMA_QUANTILES)]
    for indices in _quantile_strata(luma):
        rg = red_green[indices]
        yb = yellow_blue[indices]
        c = chroma[indices]
        channel = values[indices]
        parts.append(
            np.concatenate(
                (
                    np.array(
                        [
                            np.mean(rg),
                            np.std(rg),
                            np.mean(yb),
                            np.std(yb),
                            np.mean(c),
                            np.std(c),
                        ],
                        dtype=np.float64,
                    ),
                    _weighted_hue_moments(rg, yb, c),
                    np.mean(channel, axis=0),
                )
            )
        )

    neutral_count = max(1, int(np.ceil(0.2 * len(values))))
    neutral = np.argsort(chroma, kind="stable")[:neutral_count]
    neutral_rgb = values[neutral]
    neutral_mean = np.mean(neutral_rgb, axis=0)
    neutral_offset = neutral_mean - np.mean(neutral_mean)
    neutral_spread = np.std(neutral_rgb, axis=0)
    parts.append(np.concatenate((neutral_offset, neutral_spread)))

    descriptor = np.concatenate(parts)
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("reference descriptor is non-finite")
    return descriptor


def aggregate_reference_descriptors(descriptors: np.ndarray) -> np.ndarray:
    """Symmetrically aggregate two or more same-look reference descriptors."""

    values = np.asarray(descriptors, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) < 2
        or values.shape[1] < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("descriptors must be a finite 2D array with >=2 rows")
    return np.mean(values, axis=0)


def raw_rgb_histogram_descriptor(
    rgb: np.ndarray, *, bins_per_channel: int
) -> np.ndarray:
    """Raw marginal RGB-histogram negative-control descriptor."""

    values = _validate_rgb_rows(rgb)
    if bins_per_channel < 2:
        raise ValueError("bins_per_channel must be at least two")
    parts = []
    for channel in range(3):
        counts = np.histogram(
            values[:, channel],
            bins=bins_per_channel,
            range=(0.0, 1.0),
        )[0].astype(np.float64)
        parts.append(counts / np.sum(counts))
    return np.concatenate(parts)


@dataclass(frozen=True)
class ReferenceLookBank:
    """Train-only-standardized hard bank of bounded explicit look operators."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    standardized_features: np.ndarray
    velocity_grids: np.ndarray
    look_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        features = np.asarray(self.standardized_features, dtype=np.float64)
        grids = np.asarray(self.velocity_grids, dtype=np.float64)
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or features.ndim != 2
            or features.shape[1] != len(mean)
            or grids.ndim != 5
            or grids.shape[-1] != 3
            or len(features) != len(grids)
            or len(self.look_ids) != len(grids)
            or len(set(self.look_ids)) != len(self.look_ids)
            or np.any(scale <= 0.0)
            or not all(
                np.all(np.isfinite(value))
                for value in (mean, scale, features, grids)
            )
        ):
            raise ValueError("reference-look bank state is invalid")
        for name, value in (
            ("feature_mean", mean),
            ("feature_scale", scale),
            ("standardized_features", features),
            ("velocity_grids", grids),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "look_ids", tuple(str(x) for x in self.look_ids))

    def hard_retrieve(
        self, descriptors: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        values = np.asarray(descriptors, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("query descriptors are invalid")
        normalized = (values - self.feature_mean) / self.feature_scale
        distances = np.sqrt(
            np.mean(
                (
                    normalized[:, None, :]
                    - self.standardized_features[None, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        indices = np.argmin(distances, axis=1)
        return (
            self.velocity_grids[indices].copy(),
            tuple(self.look_ids[index] for index in indices),
            distances[np.arange(len(indices)), indices],
        )


def build_reference_look_bank(
    descriptors: np.ndarray,
    velocity_grids: np.ndarray,
    *,
    look_ids: list[str] | tuple[str, ...],
) -> ReferenceLookBank:
    values = np.asarray(descriptors, dtype=np.float64)
    grids = np.asarray(velocity_grids, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) < 2
        or grids.ndim != 5
        or grids.shape[-1] != 3
        or len(values) != len(grids)
        or len(look_ids) != len(grids)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(grids))
    ):
        raise ValueError("reference-look bank inputs are invalid")
    mean = np.mean(values, axis=0)
    scale = np.maximum(np.std(values, axis=0), 1e-8)
    return ReferenceLookBank(
        feature_mean=mean,
        feature_scale=scale,
        standardized_features=(values - mean) / scale,
        velocity_grids=grids,
        look_ids=tuple(str(x) for x in look_ids),
    )


@dataclass(frozen=True)
class FactorizedDirectionStrengthBank:
    """Hard direction retrieval with continuous piecewise-linear strength."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    standardized_prototypes: np.ndarray
    strengths: np.ndarray
    base_velocity_grids: np.ndarray
    direction_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        prototypes = np.asarray(
            self.standardized_prototypes, dtype=np.float64
        )
        strengths = np.asarray(self.strengths, dtype=np.float64)
        grids = np.asarray(self.base_velocity_grids, dtype=np.float64)
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or prototypes.ndim != 3
            or prototypes.shape[2] != len(mean)
            or strengths.shape != (prototypes.shape[1],)
            or grids.ndim != 5
            or grids.shape[-1] != 3
            or len(grids) != len(prototypes)
            or len(self.direction_ids) != len(grids)
            or len(set(self.direction_ids)) != len(self.direction_ids)
            or np.any(scale <= 0.0)
            or len(strengths) < 2
            or strengths[0] != 0.0
            or np.any(np.diff(strengths) <= 0.0)
            or not all(
                np.all(np.isfinite(value))
                for value in (mean, scale, prototypes, strengths, grids)
            )
        ):
            raise ValueError("direction-strength bank state is invalid")
        for name, value in (
            ("feature_mean", mean),
            ("feature_scale", scale),
            ("standardized_prototypes", prototypes),
            ("strengths", strengths),
            ("base_velocity_grids", grids),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(
            self, "direction_ids", tuple(str(x) for x in self.direction_ids)
        )

    def retrieve(
        self, descriptors: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
        values = np.asarray(descriptors, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("query descriptors are invalid")
        queries = (values - self.feature_mean) / self.feature_scale
        output_grids, output_ids, output_strengths, output_distances = (
            [],
            [],
            [],
            [],
        )
        for query in queries:
            best: tuple[float, int, int, float] | None = None
            for direction in range(len(self.standardized_prototypes)):
                path = self.standardized_prototypes[direction]
                for segment in range(len(self.strengths) - 1):
                    start = path[segment]
                    delta = path[segment + 1] - start
                    denominator = float(np.dot(delta, delta))
                    fraction = float(
                        np.clip(
                            np.dot(query - start, delta)
                            / max(denominator, 1e-15),
                            0.0,
                            1.0,
                        )
                    )
                    residual = query - (start + fraction * delta)
                    distance = float(np.sqrt(np.mean(residual**2)))
                    candidate = (distance, direction, segment, fraction)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                raise RuntimeError("direction-strength bank is empty")
            distance, direction, segment, fraction = best
            strength = float(
                self.strengths[segment]
                + fraction
                * (self.strengths[segment + 1] - self.strengths[segment])
            )
            output_grids.append(strength * self.base_velocity_grids[direction])
            output_ids.append(self.direction_ids[direction])
            output_strengths.append(strength)
            output_distances.append(distance)
        return (
            np.stack(output_grids),
            tuple(output_ids),
            np.asarray(output_strengths, dtype=np.float64),
            np.asarray(output_distances, dtype=np.float64),
        )


def build_direction_strength_bank(
    prototype_descriptors: np.ndarray,
    strengths: np.ndarray,
    base_velocity_grids: np.ndarray,
    *,
    direction_ids: list[str] | tuple[str, ...],
) -> FactorizedDirectionStrengthBank:
    prototypes = np.asarray(prototype_descriptors, dtype=np.float64)
    strength_values = np.asarray(strengths, dtype=np.float64)
    grids = np.asarray(base_velocity_grids, dtype=np.float64)
    if (
        prototypes.ndim != 3
        or len(prototypes) < 2
        or grids.ndim != 5
        or grids.shape[-1] != 3
        or len(prototypes) != len(grids)
        or len(direction_ids) != len(grids)
        or strength_values.shape != (prototypes.shape[1],)
        or not np.all(np.isfinite(prototypes))
        or not np.all(np.isfinite(grids))
    ):
        raise ValueError("direction-strength bank inputs are invalid")
    flat = prototypes.reshape(-1, prototypes.shape[-1])
    mean = np.mean(flat, axis=0)
    scale = np.maximum(np.std(flat, axis=0), 1e-8)
    return FactorizedDirectionStrengthBank(
        feature_mean=mean,
        feature_scale=scale,
        standardized_prototypes=(prototypes - mean) / scale,
        strengths=strength_values,
        base_velocity_grids=grids,
        direction_ids=tuple(str(x) for x in direction_ids),
    )


def fit_reference_descriptor_ridge(
    descriptors: np.ndarray,
    velocity_grids: np.ndarray,
    *,
    alpha: float,
    maximum_vector_norm: float,
) -> BoundedMultiOutputRidge:
    """Fit a parameter-only ridge predictor using train rows exclusively."""

    return fit_bounded_multi_output_ridge(
        descriptors,
        velocity_grids,
        alpha=alpha,
        maximum_vector_norm=maximum_vector_norm,
    )


__all__ = [
    "FactorizedDirectionStrengthBank",
    "ReferenceLookBank",
    "aggregate_reference_descriptors",
    "build_direction_strength_bank",
    "build_reference_look_bank",
    "factorized_reference_descriptor",
    "fit_reference_descriptor_ridge",
    "raw_rgb_histogram_descriptor",
]
