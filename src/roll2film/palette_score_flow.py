"""Reference-palette score fields expressed as bounded RGB diffeomorphisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow


@dataclass(frozen=True)
class DiagonalGaussianMixturePalette:
    """A small analytic RGB palette-density oracle."""

    weights: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        deviations = np.asarray(self.standard_deviations, dtype=np.float64)
        if (
            weights.ndim != 1
            or len(weights) < 1
            or means.shape != (len(weights), 3)
            or deviations.shape != means.shape
        ):
            raise ValueError("palette mixture shapes are inconsistent")
        if (
            not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(means))
            or not np.all(np.isfinite(deviations))
            or np.any(weights <= 0.0)
            or np.any(deviations <= 0.0)
            or np.any(means < 0.0)
            or np.any(means > 1.0)
        ):
            raise ValueError("palette mixture parameters are invalid")
        total = float(np.sum(weights))
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("palette weights must sum to one")
        weights = weights.copy()
        means = means.copy()
        deviations = deviations.copy()
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "standard_deviations", deviations)
        self.weights.setflags(write=False)
        self.means.setflags(write=False)
        self.standard_deviations.setflags(write=False)

    @classmethod
    def from_config(
        cls, payload: Mapping[str, Any]
    ) -> "DiagonalGaussianMixturePalette":
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            means=np.asarray(payload["means"], dtype=np.float64),
            standard_deviations=np.asarray(
                payload["standard_deviations"], dtype=np.float64
            ),
        )

    def _component_log_density(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb_rows(rgb)
        residual = (
            values[:, None, :] - self.means[None, :, :]
        ) / self.standard_deviations[None, :, :]
        normalization = (
            np.log(self.weights)
            - np.sum(np.log(self.standard_deviations), axis=1)
            - 1.5 * np.log(2.0 * np.pi)
        )
        return normalization[None, :] - 0.5 * np.sum(residual**2, axis=2)

    def log_density(self, rgb: np.ndarray) -> np.ndarray:
        component = self._component_log_density(rgb)
        maximum = np.max(component, axis=1, keepdims=True)
        return (
            maximum[:, 0]
            + np.log(np.sum(np.exp(component - maximum), axis=1))
        )

    def score(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb_rows(rgb)
        component = self._component_log_density(values)
        maximum = np.max(component, axis=1, keepdims=True)
        responsibility = np.exp(component - maximum)
        responsibility /= np.sum(responsibility, axis=1, keepdims=True)
        component_score = (
            self.means[None, :, :] - values[:, None, :]
        ) / (self.standard_deviations[None, :, :] ** 2)
        return np.sum(responsibility[:, :, None] * component_score, axis=1)


def _validate_rgb_rows(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("RGB values must be finite [0,1] rows")
    return values


def palette_score_velocity_grid(
    palette: DiagonalGaussianMixturePalette,
    *,
    axis_size: int,
    coefficient_vector_norm_cap: float,
) -> np.ndarray:
    """Sample and smoothly bound an analytic palette score on an RGB grid."""

    if axis_size < 2:
        raise ValueError("axis_size must be at least two")
    cap = float(coefficient_vector_norm_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("coefficient_vector_norm_cap must be positive")
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    points = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    )
    score = palette.score(points.reshape(-1, 3)).reshape(points.shape)
    norm = np.linalg.norm(score, axis=-1, keepdims=True)
    return cap * score / (1.0 + norm)


def palette_score_operator(
    palette: DiagonalGaussianMixturePalette,
    *,
    axis_size: int,
    integration_steps: int,
    coefficient_vector_norm_cap: float,
) -> CubeDiffeomorphicColourFlow:
    return CubeDiffeomorphicColourFlow(
        palette_score_velocity_grid(
            palette,
            axis_size=axis_size,
            coefficient_vector_norm_cap=coefficient_vector_norm_cap,
        ),
        integration_steps=integration_steps,
    )


__all__ = [
    "DiagonalGaussianMixturePalette",
    "palette_score_operator",
    "palette_score_velocity_grid",
]
