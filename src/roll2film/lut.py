"""Deterministic dense 3D-LUT baking and trilinear evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ColorOperator(Protocol):
    def apply(self, rgb: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DenseLUT3D:
    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        minimum = np.asarray(self.domain_min, dtype=np.float64)
        maximum = np.asarray(self.domain_max, dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != 3 or len(set(values.shape[:3])) != 1:
            raise ValueError("LUT values must have shape (N, N, N, 3)")
        if values.shape[0] < 2 or minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("LUT domain must contain three channels and at least two grid points")
        if np.any(maximum <= minimum) or not np.all(np.isfinite(values)):
            raise ValueError("LUT values/domain must be finite with a positive domain extent")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "domain_min", minimum)
        object.__setattr__(self, "domain_max", maximum)

    @property
    def size(self) -> int:
        return int(self.values.shape[0])

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        inputs = np.asarray(rgb, dtype=np.float64)
        if inputs.ndim < 2 or inputs.shape[-1] != 3 or not np.all(np.isfinite(inputs)):
            raise ValueError("RGB values must be finite with shape (..., 3)")
        if np.any(inputs < self.domain_min) or np.any(inputs > self.domain_max):
            raise ValueError("LUT input falls outside the declared domain")
        coordinates = (inputs - self.domain_min) / (self.domain_max - self.domain_min)
        coordinates *= self.size - 1
        lower = np.floor(coordinates).astype(np.int64)
        lower = np.minimum(lower, self.size - 2)
        fraction = coordinates - lower
        result = np.zeros_like(inputs)
        for red in (0, 1):
            for green in (0, 1):
                for blue in (0, 1):
                    weight = (
                        (fraction[..., 0] if red else 1.0 - fraction[..., 0])
                        * (fraction[..., 1] if green else 1.0 - fraction[..., 1])
                        * (fraction[..., 2] if blue else 1.0 - fraction[..., 2])
                    )
                    sample = self.values[
                        lower[..., 0] + red,
                        lower[..., 1] + green,
                        lower[..., 2] + blue,
                    ]
                    result += weight[..., None] * sample
        return result


def bake_dense_lut(
    operator: ColorOperator,
    size: int,
    *,
    domain_min: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> DenseLUT3D:
    if size < 2:
        raise ValueError("LUT size must be at least two")
    minimum = np.asarray(domain_min, dtype=np.float64)
    maximum = np.asarray(domain_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
        raise ValueError("LUT domain must contain three positive extents")
    axes = [np.linspace(minimum[channel], maximum[channel], size) for channel in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return DenseLUT3D(operator.apply(grid), minimum, maximum)
