"""Explicit non-duplicative sensitometry to print/display interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .density_domain import (
    _readonly_matrix,
    _readonly_vector,
    density_to_print_reflectance,
)
from .operators import _validate_rgb
from .sensitometry import RGBSensitometryOperator


PRINT_INTERPRETATION_SCHEMA = "roll2film.density_to_print_interpretation.v1"
SENSITOMETRY_PRINT_SCHEMA = "roll2film.sensitometry_print.v1"


@dataclass(frozen=True)
class DensityToPrintInterpretation:
    dye_absorption_matrix: np.ndarray
    print_matrix: np.ndarray
    paper_midpoints: np.ndarray
    paper_slopes: np.ndarray
    paper_maximum_densities: np.ndarray
    black_reference_density: np.ndarray
    white_reference_density: np.ndarray
    exposure_floor: float = 2.0**-16
    matrix_minimum_determinant: float = 0.2

    def __post_init__(self) -> None:
        for field in ("dye_absorption_matrix", "print_matrix"):
            matrix = _readonly_matrix(getattr(self, field), field)
            if np.any(matrix < 0.0):
                raise ValueError(f"{field} must be non-negative")
            if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-12:
                raise ValueError(f"{field} must be row-stochastic")
            if float(np.linalg.det(matrix)) < self.matrix_minimum_determinant:
                raise ValueError(f"{field} determinant is below the contract")
            object.__setattr__(self, field, matrix)
        for field in (
            "paper_midpoints",
            "paper_slopes",
            "paper_maximum_densities",
            "black_reference_density",
            "white_reference_density",
        ):
            object.__setattr__(self, field, _readonly_vector(getattr(self, field), field))
        if self.exposure_floor != 2.0**-16:
            raise ValueError("v1 exposure floor must equal 2^-16")
        if not 0.0 < self.matrix_minimum_determinant < 1.0:
            raise ValueError("matrix determinant floor must be in (0, 1)")
        if np.any(self.white_reference_density <= self.black_reference_density):
            raise ValueError("white reference density must exceed black reference density")
        if np.any(self.paper_slopes < 0.2) or np.any(self.paper_slopes > 4.0):
            raise ValueError("paper slopes must be in [0.2, 4.0]")
        if np.any(self.paper_maximum_densities < 0.2) or np.any(
            self.paper_maximum_densities > 4.0
        ):
            raise ValueError("paper maximum densities must be in [0.2, 4.0]")
        raw_black, raw_white = self._raw_endpoints()
        if np.any(raw_white - raw_black <= 0.0):
            raise ValueError("print interpretation endpoints must have positive span")

    def _raw(self, density: np.ndarray) -> np.ndarray:
        return density_to_print_reflectance(
            density,
            self.dye_absorption_matrix,
            self.print_matrix,
            self.paper_midpoints,
            self.paper_slopes,
            self.paper_maximum_densities,
            exposure_floor=self.exposure_floor,
        )

    def _raw_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        values = self._raw(
            np.stack((self.black_reference_density, self.white_reference_density), axis=0)
        )
        return values[0], values[1]

    def apply(self, layer_density: np.ndarray) -> np.ndarray:
        density = _validate_rgb(layer_density)
        if np.any(density < self.black_reference_density - 1e-12) or np.any(
            density > self.white_reference_density + 1e-12
        ):
            raise ValueError("layer density falls outside declared references")
        raw_black, raw_white = self._raw_endpoints()
        output = (self._raw(density) - raw_black) / (raw_white - raw_black)
        if np.any(output < -1e-12) or np.any(output > 1.0 + 1e-12):
            raise RuntimeError("print interpretation escaped normalized endpoints")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRINT_INTERPRETATION_SCHEMA,
            "exposure_floor": self.exposure_floor,
            "matrix_minimum_determinant": self.matrix_minimum_determinant,
            "dye_absorption_matrix": self.dye_absorption_matrix.tolist(),
            "print_matrix": self.print_matrix.tolist(),
            "paper_midpoints": self.paper_midpoints.tolist(),
            "paper_slopes": self.paper_slopes.tolist(),
            "paper_maximum_densities": self.paper_maximum_densities.tolist(),
            "black_reference_density": self.black_reference_density.tolist(),
            "white_reference_density": self.white_reference_density.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DensityToPrintInterpretation":
        if payload.get("schema") != PRINT_INTERPRETATION_SCHEMA:
            raise ValueError("unsupported print interpretation schema")
        return cls(
            np.asarray(payload["dye_absorption_matrix"], dtype=np.float64),
            np.asarray(payload["print_matrix"], dtype=np.float64),
            np.asarray(payload["paper_midpoints"], dtype=np.float64),
            np.asarray(payload["paper_slopes"], dtype=np.float64),
            np.asarray(payload["paper_maximum_densities"], dtype=np.float64),
            np.asarray(payload["black_reference_density"], dtype=np.float64),
            np.asarray(payload["white_reference_density"], dtype=np.float64),
            float(payload["exposure_floor"]),
            float(payload["matrix_minimum_determinant"]),
        )


@dataclass(frozen=True)
class SensitometryPrintOperator:
    sensitometry: RGBSensitometryOperator
    interpretation: DensityToPrintInterpretation

    def __post_init__(self) -> None:
        if not isinstance(self.sensitometry, RGBSensitometryOperator):
            raise ValueError("composition requires RGBSensitometryOperator")
        if not isinstance(self.interpretation, DensityToPrintInterpretation):
            raise ValueError("composition requires DensityToPrintInterpretation")

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        rgb = _validate_rgb(linear_rgb)
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("composition input must be finite [0, 1] RGB")
        return self.interpretation.apply(self.sensitometry.apply(rgb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SENSITOMETRY_PRINT_SCHEMA,
            "sensitometry": self.sensitometry.to_dict(),
            "interpretation": self.interpretation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensitometryPrintOperator":
        if payload.get("schema") != SENSITOMETRY_PRINT_SCHEMA:
            raise ValueError("unsupported sensitometry-print schema")
        return cls(
            RGBSensitometryOperator.from_dict(payload["sensitometry"]),
            DensityToPrintInterpretation.from_dict(payload["interpretation"]),
        )

