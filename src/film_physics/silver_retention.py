"""Explicit retained-silver density for generic colour-film process research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SILVER_RETENTION_SCHEMA = "neuro_film.silver_retention_profile.v1"


@dataclass(frozen=True)
class SilverRetentionProfile:
    retention_fraction: float
    dye_to_silver_weights: tuple[float, float, float]
    maximum_input_dye_density: float
    maximum_output_total_density: float

    def __post_init__(self) -> None:
        fraction = float(self.retention_fraction)
        weights = np.asarray(self.dye_to_silver_weights, dtype=np.float64)
        input_maximum = float(self.maximum_input_dye_density)
        output_maximum = float(self.maximum_output_total_density)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("retention fraction must be finite in [0,1]")
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("dye-to-silver weights must be a nonnegative sum-one vector")
        if (
            not np.isfinite(input_maximum)
            or input_maximum <= 0.0
            or not np.isfinite(output_maximum)
            or output_maximum + 1e-12
            < input_maximum * (1.0 + fraction)
        ):
            raise ValueError("density maxima do not enclose the analytical output")
        object.__setattr__(self, "retention_fraction", fraction)
        object.__setattr__(
            self, "dye_to_silver_weights", tuple(float(x) for x in weights)
        )
        object.__setattr__(self, "maximum_input_dye_density", input_maximum)
        object.__setattr__(self, "maximum_output_total_density", output_maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SILVER_RETENTION_SCHEMA,
            "retention_fraction": self.retention_fraction,
            "dye_to_silver_weights": list(self.dye_to_silver_weights),
            "maximum_input_dye_density": self.maximum_input_dye_density,
            "maximum_output_total_density": self.maximum_output_total_density,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SilverRetentionProfile":
        if payload.get("schema") != SILVER_RETENTION_SCHEMA:
            raise ValueError("unsupported silver-retention schema")
        return cls(
            float(payload["retention_fraction"]),
            tuple(payload["dye_to_silver_weights"]),
            float(payload["maximum_input_dye_density"]),
            float(payload["maximum_output_total_density"]),
        )


@dataclass(frozen=True)
class SilverRetentionResult:
    total_density: np.ndarray
    silver_density: np.ndarray

    def __post_init__(self) -> None:
        total = np.asarray(self.total_density)
        silver = np.asarray(self.silver_density)
        if (
            total.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or total.ndim < 2
            or total.shape[-1] != 3
            or silver.shape != total.shape[:-1]
            or silver.dtype != total.dtype
            or not np.all(np.isfinite(total))
            or not np.all(np.isfinite(silver))
        ):
            raise ValueError("invalid silver-retention result")
        owned_total = np.array(total, copy=True, order="C")
        owned_silver = np.array(silver, copy=True, order="C")
        owned_total.setflags(write=False)
        owned_silver.setflags(write=False)
        object.__setattr__(self, "total_density", owned_total)
        object.__setattr__(self, "silver_density", owned_silver)


def apply_silver_retention(
    dye_density: np.ndarray, profile: SilverRetentionProfile
) -> SilverRetentionResult:
    if not isinstance(profile, SilverRetentionProfile):
        raise TypeError("profile must be SilverRetentionProfile")
    density = np.asarray(dye_density)
    if (
        density.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or density.ndim < 2
        or density.shape[-1] != 3
        or density.size == 0
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
        or np.any(density > profile.maximum_input_dye_density)
    ):
        raise ValueError("dye density must be finite, bounded and have shape (...,3)")
    weights = np.asarray(profile.dye_to_silver_weights, dtype=density.dtype)
    silver = (
        np.sum(density * weights, axis=-1)
        * density.dtype.type(profile.retention_fraction)
    )
    total = density + silver[..., None]
    if np.any(total > profile.maximum_output_total_density):
        raise ValueError("analytical total-density bound was violated")
    return SilverRetentionResult(total, silver)


def apply_silver_retention_row_tiled(
    dye_density: np.ndarray,
    profile: SilverRetentionProfile,
    *,
    tile_rows: int,
) -> SilverRetentionResult:
    density = np.asarray(dye_density)
    if density.ndim != 3 or density.shape[-1] != 3:
        raise ValueError("row-tiled input must be HxWx3")
    if not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    total_rows = []
    silver_rows = []
    for start in range(0, density.shape[0], tile_rows):
        result = apply_silver_retention(
            density[start : start + tile_rows], profile
        )
        total_rows.append(result.total_density)
        silver_rows.append(result.silver_density)
    return SilverRetentionResult(
        np.concatenate(total_rows, axis=0),
        np.concatenate(silver_rows, axis=0),
    )


def density_jacobian(profile: SilverRetentionProfile) -> np.ndarray:
    """Return d(total density output)/d(dye density input)."""
    weights = np.asarray(profile.dye_to_silver_weights, dtype=np.float64)
    return np.eye(3, dtype=np.float64) + profile.retention_fraction * np.repeat(
        weights[None, :], 3, axis=0
    )
