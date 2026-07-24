"""Deterministic dense 3D-LUT baking and explicit interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


LUT_SCHEMA = "roll2film.dense_lut_3d.v1"
SHAPER_SCHEMA = "roll2film.log1p_shaper.v1"
SHAPED_LUT_SCHEMA = "roll2film.shaped_lut_3d.v1"
_INTERPOLATIONS = frozenset({"trilinear", "tetrahedral"})


class ColorOperator(Protocol):
    def apply(self, rgb: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LogShaperSpec:
    """Analytic nonnegative linear/HDR log1p input shaper."""

    domain_max: float
    compression: float

    def __post_init__(self) -> None:
        maximum = float(self.domain_max)
        compression = float(self.compression)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("shaper domain maximum must be finite and positive")
        if not np.isfinite(compression) or compression <= 0.0:
            raise ValueError("shaper compression must be finite and positive")
        object.__setattr__(self, "domain_max", maximum)
        object.__setattr__(self, "compression", compression)

    @property
    def denominator(self) -> float:
        return float(np.log1p(self.compression * self.domain_max))

    def apply(self, linear: np.ndarray) -> np.ndarray:
        values = np.asarray(linear, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("shaper input must be finite")
        if np.any(values < 0.0) or np.any(values > self.domain_max):
            raise ValueError("shaper input falls outside the declared linear domain")
        return np.log1p(self.compression * values) / self.denominator

    def inverse(self, shaped: np.ndarray) -> np.ndarray:
        values = np.asarray(shaped, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("inverse shaper input must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("inverse shaper input falls outside [0, 1]")
        return np.expm1(values * self.denominator) / self.compression

    def derivative(self, linear: np.ndarray) -> np.ndarray:
        values = np.asarray(linear, dtype=np.float64)
        self.apply(values)
        return self.compression / (
            self.denominator * (1.0 + self.compression * values)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHAPER_SCHEMA,
            "domain_max": self.domain_max,
            "compression": self.compression,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogShaperSpec":
        if payload.get("schema") != SHAPER_SCHEMA:
            raise ValueError(f"unsupported shaper schema: {payload.get('schema')!r}")
        return cls(float(payload["domain_max"]), float(payload["compression"]))


@dataclass(frozen=True)
class DenseLUT3D:
    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray
    interpolation: str = "trilinear"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        minimum = np.asarray(self.domain_min, dtype=np.float64)
        maximum = np.asarray(self.domain_max, dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != 3 or len(set(values.shape[:3])) != 1:
            raise ValueError("LUT values must have shape (N, N, N, 3)")
        if values.shape[0] < 2 or minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("LUT domain must contain three channels and at least two grid points")
        if (
            np.any(maximum <= minimum)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
        ):
            raise ValueError("LUT values/domain must be finite with a positive domain extent")
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(f"unsupported LUT interpolation: {self.interpolation!r}")
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
        if self.interpolation == "tetrahedral":
            return self._apply_tetrahedral(inputs, lower, fraction)
        return self._apply_trilinear(inputs, lower, fraction)

    def _apply_trilinear(
        self,
        inputs: np.ndarray,
        lower: np.ndarray,
        fraction: np.ndarray,
    ) -> np.ndarray:
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

    def _apply_tetrahedral(
        self,
        inputs: np.ndarray,
        lower: np.ndarray,
        fraction: np.ndarray,
    ) -> np.ndarray:
        flat_lower = lower.reshape(-1, 3)
        flat_fraction = fraction.reshape(-1, 3)
        r_index, g_index, b_index = flat_lower.T
        c000 = self.values[r_index, g_index, b_index]
        c100 = self.values[r_index + 1, g_index, b_index]
        c010 = self.values[r_index, g_index + 1, b_index]
        c001 = self.values[r_index, g_index, b_index + 1]
        c110 = self.values[r_index + 1, g_index + 1, b_index]
        c101 = self.values[r_index + 1, g_index, b_index + 1]
        c011 = self.values[r_index, g_index + 1, b_index + 1]
        c111 = self.values[r_index + 1, g_index + 1, b_index + 1]
        red, green, blue = flat_fraction.T

        result = np.empty_like(c000)
        rgb = (red >= green) & (green >= blue)
        rbg = (red >= blue) & (blue > green)
        brg = (blue > red) & (red >= green)
        grb = (green > red) & (red >= blue)
        gbr = (green >= blue) & (blue > red)
        bgr = (blue > green) & (green > red)

        result[rgb] = (
            c000[rgb]
            + red[rgb, None] * (c100[rgb] - c000[rgb])
            + green[rgb, None] * (c110[rgb] - c100[rgb])
            + blue[rgb, None] * (c111[rgb] - c110[rgb])
        )
        result[rbg] = (
            c000[rbg]
            + red[rbg, None] * (c100[rbg] - c000[rbg])
            + blue[rbg, None] * (c101[rbg] - c100[rbg])
            + green[rbg, None] * (c111[rbg] - c101[rbg])
        )
        result[brg] = (
            c000[brg]
            + blue[brg, None] * (c001[brg] - c000[brg])
            + red[brg, None] * (c101[brg] - c001[brg])
            + green[brg, None] * (c111[brg] - c101[brg])
        )
        result[grb] = (
            c000[grb]
            + green[grb, None] * (c010[grb] - c000[grb])
            + red[grb, None] * (c110[grb] - c010[grb])
            + blue[grb, None] * (c111[grb] - c110[grb])
        )
        result[gbr] = (
            c000[gbr]
            + green[gbr, None] * (c010[gbr] - c000[gbr])
            + blue[gbr, None] * (c011[gbr] - c010[gbr])
            + red[gbr, None] * (c111[gbr] - c011[gbr])
        )
        result[bgr] = (
            c000[bgr]
            + blue[bgr, None] * (c001[bgr] - c000[bgr])
            + green[bgr, None] * (c011[bgr] - c001[bgr])
            + red[bgr, None] * (c111[bgr] - c011[bgr])
        )
        if not np.all(rgb | rbg | brg | grb | gbr | bgr):
            raise RuntimeError("tetrahedral interpolation did not partition the LUT cell")
        return result.reshape(inputs.shape)

    def tetrahedron_jacobian_determinants(self) -> np.ndarray:
        """Return exact affine Jacobian determinants for all six cell tetrahedra."""
        c000 = self.values[:-1, :-1, :-1]
        c100 = self.values[1:, :-1, :-1]
        c010 = self.values[:-1, 1:, :-1]
        c001 = self.values[:-1, :-1, 1:]
        c110 = self.values[1:, 1:, :-1]
        c101 = self.values[1:, :-1, 1:]
        c011 = self.values[:-1, 1:, 1:]
        c111 = self.values[1:, 1:, 1:]
        steps = (self.domain_max - self.domain_min) / (self.size - 1)

        columns = (
            (c100 - c000, c110 - c100, c111 - c110),
            (c100 - c000, c111 - c101, c101 - c100),
            (c101 - c001, c111 - c101, c001 - c000),
            (c110 - c010, c010 - c000, c111 - c110),
            (c111 - c011, c010 - c000, c011 - c010),
            (c111 - c011, c011 - c001, c001 - c000),
        )
        determinants = []
        for red, green, blue in columns:
            jacobian = np.stack(
                (red / steps[0], green / steps[1], blue / steps[2]),
                axis=-1,
            )
            determinants.append(np.linalg.det(jacobian))
        return np.stack(determinants, axis=-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LUT_SCHEMA,
            "interpolation": self.interpolation,
            "domain_min": self.domain_min.tolist(),
            "domain_max": self.domain_max.tolist(),
            "values": self.values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseLUT3D":
        if payload.get("schema") != LUT_SCHEMA:
            raise ValueError(f"unsupported LUT schema: {payload.get('schema')!r}")
        return cls(
            values=np.asarray(payload["values"], dtype=np.float64),
            domain_min=np.asarray(payload["domain_min"], dtype=np.float64),
            domain_max=np.asarray(payload["domain_max"], dtype=np.float64),
            interpolation=str(payload["interpolation"]),
        )


@dataclass(frozen=True)
class ShapedLUT3D:
    """Dense LUT with an explicit analytic input shaper and linear output."""

    shaper: LogShaperSpec
    lut: DenseLUT3D
    working_space: str = "linear_srgb"

    def __post_init__(self) -> None:
        if not isinstance(self.shaper, LogShaperSpec):
            raise ValueError("shaped LUT requires a LogShaperSpec")
        if not isinstance(self.lut, DenseLUT3D):
            raise ValueError("shaped LUT requires a DenseLUT3D")
        if not np.array_equal(self.lut.domain_min, np.zeros(3)) or not np.array_equal(
            self.lut.domain_max, np.ones(3)
        ):
            raise ValueError("shaped LUT dense domain must be exactly [0, 1]^3")
        if not isinstance(self.working_space, str) or not self.working_space:
            raise ValueError("working space must be a non-empty string")

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(linear_rgb, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != 3:
            raise ValueError("RGB values must have shape (..., 3)")
        return self.lut.apply(self.shaper.apply(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHAPED_LUT_SCHEMA,
            "working_space": self.working_space,
            "shaper": self.shaper.to_dict(),
            "lut": self.lut.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShapedLUT3D":
        if payload.get("schema") != SHAPED_LUT_SCHEMA:
            raise ValueError(f"unsupported shaped LUT schema: {payload.get('schema')!r}")
        return cls(
            LogShaperSpec.from_dict(payload["shaper"]),
            DenseLUT3D.from_dict(payload["lut"]),
            str(payload["working_space"]),
        )


def bake_dense_lut(
    operator: ColorOperator,
    size: int,
    *,
    domain_min: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolation: str = "trilinear",
) -> DenseLUT3D:
    if size < 2:
        raise ValueError("LUT size must be at least two")
    minimum = np.asarray(domain_min, dtype=np.float64)
    maximum = np.asarray(domain_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
        raise ValueError("LUT domain must contain three positive extents")
    axes = [np.linspace(minimum[channel], maximum[channel], size) for channel in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return DenseLUT3D(operator.apply(grid), minimum, maximum, interpolation)


def bake_shaped_lut(
    operator: ColorOperator,
    size: int,
    shaper: LogShaperSpec,
    *,
    interpolation: str = "trilinear",
    working_space: str = "linear_srgb",
) -> ShapedLUT3D:
    """Bake an operator on a uniform shaper grid without output clamping."""

    if size < 2:
        raise ValueError("LUT size must be at least two")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    shaped_grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    linear_grid = shaper.inverse(shaped_grid)
    lut = DenseLUT3D(
        np.asarray(operator.apply(linear_grid), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        interpolation,
    )
    return ShapedLUT3D(shaper, lut, working_space)
