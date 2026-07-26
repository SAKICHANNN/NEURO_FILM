"""Canonical colour-histogram retrieval for explicit palette-score flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.roll2film.palette_score_flow import (
    DiagonalGaussianMixturePalette,
    palette_score_velocity_grid,
)


DistanceName = Literal["hellinger", "jensen_shannon"]


@dataclass(frozen=True)
class HistogramCaseBank:
    """A fixed bank of canonical histograms and explicit velocity grids."""

    histograms: np.ndarray
    velocity_grids: np.ndarray

    def __post_init__(self) -> None:
        histograms = np.asarray(self.histograms, dtype=np.float64)
        grids = np.asarray(self.velocity_grids, dtype=np.float64)
        if (
            histograms.ndim != 2
            or grids.ndim != 5
            or len(histograms) != len(grids)
            or len(histograms) < 1
            or grids.shape[-1] != 3
            or not np.all(np.isfinite(histograms))
            or not np.all(np.isfinite(grids))
            or np.any(histograms < 0.0)
        ):
            raise ValueError("case-bank arrays are invalid")
        totals = np.sum(histograms, axis=1)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("case-bank histograms must sum to one")
        histograms = histograms.copy()
        grids = grids.copy()
        histograms.setflags(write=False)
        grids.setflags(write=False)
        object.__setattr__(self, "histograms", histograms)
        object.__setattr__(self, "velocity_grids", grids)

    def distances(
        self, histogram: np.ndarray, *, distance: DistanceName
    ) -> np.ndarray:
        query = validate_histogram(histogram, self.histograms.shape[1])
        if distance == "hellinger":
            return np.sqrt(
                0.5
                * np.sum(
                    (
                        np.sqrt(self.histograms)
                        - np.sqrt(query)[None, :]
                    )
                    ** 2,
                    axis=1,
                )
            )
        if distance == "jensen_shannon":
            midpoint = 0.5 * (self.histograms + query[None, :])
            left = np.zeros_like(self.histograms)
            left_mask = self.histograms > 0.0
            np.log(
                self.histograms,
                out=left,
                where=left_mask,
            )
            left -= np.log(np.maximum(midpoint, 1e-300))
            left *= self.histograms
            right = np.zeros_like(midpoint)
            query_rows = np.broadcast_to(query[None, :], midpoint.shape)
            right_mask = query_rows > 0.0
            np.log(
                query_rows,
                out=right,
                where=right_mask,
            )
            right -= np.log(np.maximum(midpoint, 1e-300))
            right *= query_rows
            return np.sqrt(0.5 * (np.sum(left, axis=1) + np.sum(right, axis=1)))
        raise ValueError(f"unsupported histogram distance: {distance}")

    def hard_retrieve(
        self, histogram: np.ndarray, *, distance: DistanceName
    ) -> tuple[int, np.ndarray]:
        distances = self.distances(histogram, distance=distance)
        index = int(np.argmin(distances))
        return index, self.velocity_grids[index].copy()

    def inverse_distance_blend(
        self,
        histogram: np.ndarray,
        *,
        distance: DistanceName,
        neighbors: int,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if neighbors < 1 or neighbors > len(self.histograms):
            raise ValueError("neighbors must fit the case bank")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        distances = self.distances(histogram, distance=distance)
        indices = np.argsort(distances, kind="stable")[:neighbors]
        exact = distances[indices] <= epsilon
        if np.any(exact):
            weights = exact.astype(np.float64)
        else:
            weights = 1.0 / np.maximum(distances[indices], epsilon)
        weights /= np.sum(weights)
        grid = np.tensordot(weights, self.velocity_grids[indices], axes=(0, 0))
        return indices, grid


def validate_histogram(histogram: np.ndarray, expected_bins: int) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if (
        values.shape != (expected_bins,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("histogram must be a finite normalized vector")
    return values


def canonical_rgb_histogram(rgb: np.ndarray, *, axis_size: int) -> np.ndarray:
    """Return spatially permutation-invariant lexicographic RGB counts."""

    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or axis_size < 2
    ):
        raise ValueError("rgb samples must be finite [0,1] rows")
    indices = np.minimum((values * axis_size).astype(np.int64), axis_size - 1)
    flat = (
        (indices[:, 0] * axis_size + indices[:, 1]) * axis_size
        + indices[:, 2]
    )
    counts = np.bincount(flat, minlength=axis_size**3).astype(np.float64)
    return counts / float(len(values))


def histogram_bin_centres(*, axis_size: int) -> np.ndarray:
    if axis_size < 2:
        raise ValueError("axis_size must be at least two")
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def histogram_kde_velocity_grid(
    histogram: np.ndarray,
    *,
    histogram_axis_size: int,
    bandwidth: float,
    velocity_grid_axis_size: int,
    coefficient_vector_norm_cap: float,
) -> np.ndarray:
    """Estimate a score grid directly from histogram mass using isotropic KDE."""

    expected_bins = histogram_axis_size**3
    values = validate_histogram(histogram, expected_bins)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    centres = histogram_bin_centres(axis_size=histogram_axis_size)
    retained = values > 0.0
    weights = values[retained]
    weights /= np.sum(weights)
    palette = DiagonalGaussianMixturePalette(
        weights=weights,
        means=centres[retained],
        standard_deviations=np.full(
            (int(np.sum(retained)), 3), bandwidth, dtype=np.float64
        ),
    )
    return palette_score_velocity_grid(
        palette,
        axis_size=velocity_grid_axis_size,
        coefficient_vector_norm_cap=coefficient_vector_norm_cap,
    )


def generate_synthetic_palette(
    rng: np.random.Generator,
    *,
    component_counts: Sequence[int],
    weight_dirichlet_alpha: float,
    mean_minimum: float,
    mean_maximum: float,
    standard_deviation_minimum: float,
    standard_deviation_maximum: float,
) -> DiagonalGaussianMixturePalette:
    counts = tuple(int(value) for value in component_counts)
    if (
        not counts
        or min(counts) < 1
        or weight_dirichlet_alpha <= 0.0
        or not 0.0 <= mean_minimum < mean_maximum <= 1.0
        or not 0.0 < standard_deviation_minimum
        < standard_deviation_maximum
    ):
        raise ValueError("synthetic palette generator parameters are invalid")
    count = counts[int(rng.integers(0, len(counts)))]
    return DiagonalGaussianMixturePalette(
        weights=rng.dirichlet(np.full(count, weight_dirichlet_alpha)),
        means=rng.uniform(mean_minimum, mean_maximum, size=(count, 3)),
        standard_deviations=rng.uniform(
            standard_deviation_minimum,
            standard_deviation_maximum,
            size=(count, 3),
        ),
    )


def sample_palette(
    palette: DiagonalGaussianMixturePalette,
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    components = rng.choice(
        len(palette.weights), size=sample_count, p=palette.weights
    )
    samples = rng.normal(
        palette.means[components], palette.standard_deviations[components]
    )
    return np.clip(samples, 0.0, 1.0)


__all__ = [
    "HistogramCaseBank",
    "canonical_rgb_histogram",
    "generate_synthetic_palette",
    "histogram_bin_centres",
    "histogram_kde_velocity_grid",
    "sample_palette",
    "validate_histogram",
]
