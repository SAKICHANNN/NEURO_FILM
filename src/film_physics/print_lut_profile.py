"""Compact FP16 tetrahedral profiles for explicit density-to-print operators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.roll2film.lut import DenseLUT3D
from src.roll2film.sensitometry_print import DensityToPrintInterpretation

FP16_PRINT_LUT_SCHEMA = "neuro_film.film_physics.fp16_print_lut.v1"


@dataclass(frozen=True)
class FP16PrintLUTProfile:
    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float16)
        minimum = np.asarray(self.domain_min, dtype=np.float32)
        maximum = np.asarray(self.domain_max, dtype=np.float32)
        if (
            values.ndim != 4
            or values.shape[-1] != 3
            or len(set(values.shape[:3])) != 1
            or values.shape[0] < 2
        ):
            raise ValueError("FP16 print LUT must have shape (N, N, N, 3)")
        if (
            minimum.shape != (3,)
            or maximum.shape != (3,)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or np.any(maximum <= minimum)
        ):
            raise ValueError("FP16 print LUT domain is invalid")
        for owned in (values, minimum, maximum):
            owned.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "domain_min", minimum)
        object.__setattr__(self, "domain_max", maximum)

    @property
    def size(self) -> int:
        return int(self.values.shape[0])

    @property
    def storage_bytes(self) -> int:
        return int(self.values.nbytes)

    @property
    def values_sha256(self) -> str:
        return hashlib.sha256(self.values.tobytes(order="C")).hexdigest()

    def descriptor(self) -> dict[str, object]:
        payload = {
            "schema": FP16_PRINT_LUT_SCHEMA,
            "size": self.size,
            "storage_dtype": "float16",
            "runtime_dtype": "float32",
            "interpolation": "tetrahedral",
            "domain_min": self.domain_min.tolist(),
            "domain_max": self.domain_max.tolist(),
            "storage_bytes": self.storage_bytes,
            "values_sha256": self.values_sha256,
        }
        payload["profile_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload

    def apply(self, density: np.ndarray) -> np.ndarray:
        inputs = np.asarray(density, dtype=np.float32)
        if (
            inputs.ndim < 2
            or inputs.shape[-1] != 3
            or not np.all(np.isfinite(inputs))
            or np.any(inputs < self.domain_min)
            or np.any(inputs > self.domain_max)
        ):
            raise ValueError("density falls outside the FP16 print LUT domain")
        coordinates = (inputs - self.domain_min) / (
            self.domain_max - self.domain_min
        )
        coordinates *= np.float32(self.size - 1)
        lower = np.floor(coordinates).astype(np.int64)
        lower = np.minimum(lower, self.size - 2)
        fraction = coordinates - lower.astype(np.float32)
        flat_lower = lower.reshape(-1, 3)
        flat_fraction = fraction.reshape(-1, 3)
        red_index, green_index, blue_index = flat_lower.T
        values = self.values.astype(np.float32)
        c000 = values[red_index, green_index, blue_index]
        c100 = values[red_index + 1, green_index, blue_index]
        c010 = values[red_index, green_index + 1, blue_index]
        c001 = values[red_index, green_index, blue_index + 1]
        c110 = values[red_index + 1, green_index + 1, blue_index]
        c101 = values[red_index + 1, green_index, blue_index + 1]
        c011 = values[red_index, green_index + 1, blue_index + 1]
        c111 = values[red_index + 1, green_index + 1, blue_index + 1]
        red, green, blue = flat_fraction.T
        result = np.empty_like(c000, dtype=np.float32)
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
            raise RuntimeError("tetrahedral interpolation did not partition the cell")
        return result.reshape(inputs.shape)

    def minimum_tetrahedral_jacobian_determinant(self) -> float:
        reference = DenseLUT3D(
            self.values.astype(np.float64),
            self.domain_min.astype(np.float64),
            self.domain_max.astype(np.float64),
            "tetrahedral",
        )
        return float(np.min(reference.tetrahedron_jacobian_determinants()))


def compile_fp16_print_lut(
    operator: DensityToPrintInterpretation,
    size: int,
) -> FP16PrintLUTProfile:
    if not isinstance(operator, DensityToPrintInterpretation):
        raise TypeError("operator must be DensityToPrintInterpretation")
    if size < 2:
        raise ValueError("print LUT size must be at least two")
    axes = [
        np.linspace(
            operator.black_reference_density[channel],
            operator.white_reference_density[channel],
            size,
            dtype=np.float64,
        )
        for channel in range(3)
    ]
    density = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = operator.apply(density)
    return FP16PrintLUTProfile(
        values.astype(np.float16),
        operator.black_reference_density,
        operator.white_reference_density,
    )


__all__ = [
    "FP16_PRINT_LUT_SCHEMA",
    "FP16PrintLUTProfile",
    "compile_fp16_print_lut",
]
