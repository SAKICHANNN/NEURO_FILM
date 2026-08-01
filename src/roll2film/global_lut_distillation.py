"""Fit and execute a fixed monotone-shaper plus bounded tetrahedral LUT.

This module is deliberately target-agnostic.  It fits only explicit colour
parameters and never predicts or emits final RGB through a neural model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.isotonic import IsotonicRegression

from src.roll2film.lut import DenseLUT3D


@dataclass(frozen=True)
class MonotoneRGBShaper:
    """Three strictly increasing piecewise-linear channel curves."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or values.shape[0] < 3
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or np.any(np.diff(values, axis=0) <= 0.0)
        ):
            raise ValueError("shaper values must be finite strictly increasing Kx3 curves")
        object.__setattr__(self, "values", values)

    @property
    def knots(self) -> int:
        return int(self.values.shape[0])

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(rgb, dtype=np.float64)
        if (
            values.ndim < 2
            or values.shape[-1] != 3
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("shaper input must be finite RGB in [0,1]")
        axis = np.linspace(0.0, 1.0, self.knots, dtype=np.float64)
        flat = values.reshape(-1, 3)
        result = np.empty_like(flat)
        for channel in range(3):
            result[:, channel] = np.interp(
                flat[:, channel], axis, self.values[:, channel]
            )
        return result.reshape(values.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuro_film.monotone_rgb_shaper.v1",
            "values": self.values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MonotoneRGBShaper":
        if payload.get("schema") != "neuro_film.monotone_rgb_shaper.v1":
            raise ValueError("unsupported monotone shaper schema")
        return cls(np.asarray(payload["values"], dtype=np.float64))


@dataclass(frozen=True)
class ShapedGlobalLUT:
    shaper: MonotoneRGBShaper
    lut: DenseLUT3D
    residual_strength: float

    def __post_init__(self) -> None:
        strength = float(self.residual_strength)
        if (
            self.lut.interpolation != "tetrahedral"
            or not np.isfinite(strength)
            or strength < 0.0
            or strength > 1.0
        ):
            raise ValueError("invalid shaped LUT")
        object.__setattr__(self, "residual_strength", strength)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return self.lut.apply(self.shaper.apply(rgb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuro_film.shaped_global_lut.v1",
            "shaper": self.shaper.to_dict(),
            "lut": self.lut.to_dict(),
            "residual_strength": self.residual_strength,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShapedGlobalLUT":
        if payload.get("schema") != "neuro_film.shaped_global_lut.v1":
            raise ValueError("unsupported shaped LUT schema")
        return cls(
            MonotoneRGBShaper.from_dict(payload["shaper"]),
            DenseLUT3D.from_dict(payload["lut"]),
            float(payload["residual_strength"]),
        )


def _strict_curve(values: np.ndarray, minimum_increment: float) -> np.ndarray:
    curve = np.asarray(values, dtype=np.float64).copy()
    identity = np.linspace(0.0, 1.0, len(curve), dtype=np.float64)
    curve[0] = 0.0
    curve[-1] = 1.0
    curve = np.maximum.accumulate(np.clip(curve, 0.0, 1.0))
    increments = np.diff(curve)
    identity_increment = 1.0 / (len(curve) - 1)
    if minimum_increment >= identity_increment:
        raise ValueError("minimum increment is incompatible with the knot count")
    required = 0.0
    for increment in increments:
        if increment < minimum_increment:
            required = max(
                required,
                (minimum_increment - increment) / (identity_increment - increment),
            )
    return (1.0 - required) * curve + required * identity


def fit_monotone_shaper(
    source: np.ndarray,
    target: np.ndarray,
    *,
    knots: int,
    minimum_increment: float,
) -> MonotoneRGBShaper:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 256:
        raise ValueError("shaper fit needs aligned RGB samples")
    axis = np.linspace(0.0, 1.0, knots, dtype=np.float64)
    curves = np.empty((knots, 3), dtype=np.float64)
    anchor_weight = max(float(len(source)), 1.0)
    for channel in range(3):
        x = np.concatenate(([0.0], source[:, channel], [1.0]))
        y = np.concatenate(([0.0], target[:, channel], [1.0]))
        weights = np.concatenate(([anchor_weight], np.ones(len(source)), [anchor_weight]))
        fitted = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(x, y, sample_weight=weights)
        curves[:, channel] = _strict_curve(
            fitted.predict(axis), minimum_increment
        )
    return MonotoneRGBShaper(curves)


def _trilinear_design(samples: np.ndarray, size: int) -> sparse.csr_matrix:
    values = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    coordinates = values * (size - 1)
    lower = np.minimum(np.floor(coordinates).astype(np.int64), size - 2)
    fraction = coordinates - lower
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    row_index = np.arange(len(values), dtype=np.int64)
    for red in (0, 1):
        for green in (0, 1):
            for blue in (0, 1):
                node = lower + np.asarray([red, green, blue], dtype=np.int64)
                weight = (
                    (fraction[:, 0] if red else 1.0 - fraction[:, 0])
                    * (fraction[:, 1] if green else 1.0 - fraction[:, 1])
                    * (fraction[:, 2] if blue else 1.0 - fraction[:, 2])
                )
                rows.append(row_index)
                columns.append((node[:, 0] * size + node[:, 1]) * size + node[:, 2])
                weights.append(weight)
    return sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(values), size**3),
    ).tocsr()


def _difference_design(size: int) -> sparse.csr_matrix:
    grid = np.arange(size**3, dtype=np.int64).reshape(size, size, size)
    pairs = []
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        pairs.append((grid[tuple(first)].ravel(), grid[tuple(second)].ravel()))
    left = np.concatenate([pair[0] for pair in pairs])
    right = np.concatenate([pair[1] for pair in pairs])
    row = np.arange(len(left), dtype=np.int64)
    return sparse.coo_matrix(
        (
            np.concatenate((np.ones(len(row)), -np.ones(len(row)))),
            (np.concatenate((row, row)), np.concatenate((left, right))),
        ),
        shape=(len(row), size**3),
    ).tocsr()


def _safe_strength(
    raw_values: np.ndarray,
    *,
    output_margin: float,
    minimum_determinant: float,
    iterations: int,
) -> tuple[float, DenseLUT3D]:
    size = int(raw_values.shape[0])
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    safe_identity = output_margin + (1.0 - 2.0 * output_margin) * identity

    def candidate(strength: float) -> DenseLUT3D:
        return DenseLUT3D(
            safe_identity + strength * (raw_values - safe_identity),
            np.zeros(3),
            np.ones(3),
            "tetrahedral",
        )

    def valid(strength: float) -> bool:
        lut = candidate(strength)
        return bool(
            np.min(lut.values) >= output_margin - 1e-15
            and np.max(lut.values) <= 1.0 - output_margin + 1e-15
            and np.min(lut.tetrahedron_jacobian_determinants())
            >= minimum_determinant
        )

    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) * 0.5
        if valid(middle):
            low = middle
        else:
            high = middle
    return low, candidate(low)


def fit_shaped_global_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    shaper_knots: int,
    minimum_shaper_increment: float,
    lut_size: int,
    identity_regularization: float,
    difference_regularization: float,
    lsqr_iteration_limit: int,
    output_margin: float,
    minimum_determinant: float,
    strength_iterations: int,
) -> ShapedGlobalLUT:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 256:
        raise ValueError("LUT fit needs aligned RGB samples")
    shaper = fit_monotone_shaper(
        source,
        target,
        knots=shaper_knots,
        minimum_increment=minimum_shaper_increment,
    )
    shaped = shaper.apply(source)
    design = _trilinear_design(shaped, lut_size)
    identity = sparse.identity(lut_size**3, format="csr")
    differences = _difference_design(lut_size)
    system = sparse.vstack(
        (
            design,
            np.sqrt(identity_regularization) * identity,
            np.sqrt(difference_regularization) * differences,
        ),
        format="csr",
    )
    residual_target = target - shaped
    zeros = np.zeros(lut_size**3 + differences.shape[0], dtype=np.float64)
    residual = np.empty((lut_size**3, 3), dtype=np.float64)
    for channel in range(3):
        right = np.concatenate((residual_target[:, channel], zeros))
        residual[:, channel] = lsqr(
            system,
            right,
            atol=1e-8,
            btol=1e-8,
            iter_lim=lsqr_iteration_limit,
            show=False,
        )[0]
    axis = np.linspace(0.0, 1.0, lut_size, dtype=np.float64)
    identity_values = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    )
    raw_values = identity_values + residual.reshape(lut_size, lut_size, lut_size, 3)
    strength, lut = _safe_strength(
        raw_values,
        output_margin=output_margin,
        minimum_determinant=minimum_determinant,
        iterations=strength_iterations,
    )
    return ShapedGlobalLUT(shaper, lut, strength)


__all__ = [
    "MonotoneRGBShaper",
    "ShapedGlobalLUT",
    "fit_monotone_shaper",
    "fit_shaped_global_lut",
]
