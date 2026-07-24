"""Clean-room continuous negative-to-print density-domain colour operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DENSITY_OPERATOR_SCHEMA = "roll2film.density_domain_negative_print.v1"


def logistic_density(
    log2_exposure: np.ndarray,
    midpoint: np.ndarray,
    slope: np.ndarray,
    maximum_density: np.ndarray,
) -> np.ndarray:
    exposure = np.asarray(log2_exposure, dtype=np.float64)
    midpoint_values = np.asarray(midpoint, dtype=np.float64)
    slope_values = np.asarray(slope, dtype=np.float64)
    density_values = np.asarray(maximum_density, dtype=np.float64)
    argument = slope_values * (exposure - midpoint_values)
    return density_values / (1.0 + np.exp(-argument))


def density_to_print_reflectance(
    negative_density: np.ndarray,
    dye_absorption_matrix: np.ndarray,
    print_matrix: np.ndarray,
    paper_midpoints: np.ndarray,
    paper_slopes: np.ndarray,
    paper_maximum_densities: np.ndarray,
    *,
    exposure_floor: float,
) -> np.ndarray:
    """Apply only dye, print-exposure and paper-density interpretation stages."""

    density = np.asarray(negative_density, dtype=np.float64)
    if density.ndim < 2 or density.shape[-1] != 3 or not np.all(np.isfinite(density)):
        raise ValueError("negative density must be finite with shape (..., 3)")
    absorption_density = density @ np.asarray(dye_absorption_matrix, dtype=np.float64).T
    negative_transmission = np.power(10.0, -absorption_density)
    print_exposure = negative_transmission @ np.asarray(print_matrix, dtype=np.float64).T
    paper_log_exposure = np.log2(print_exposure + float(exposure_floor))
    paper_density = logistic_density(
        paper_log_exposure,
        paper_midpoints,
        paper_slopes,
        paper_maximum_densities,
    )
    return np.power(10.0, -paper_density)


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    array = array.copy()
    array.setflags(write=False)
    return array


def _readonly_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class DensityDomainNegativePrintOperator:
    capture_matrix: np.ndarray
    negative_midpoints: np.ndarray
    negative_slopes: np.ndarray
    negative_maximum_densities: np.ndarray
    dye_absorption_matrix: np.ndarray
    print_matrix: np.ndarray
    paper_midpoints: np.ndarray
    paper_slopes: np.ndarray
    paper_maximum_densities: np.ndarray
    exposure_floor: float = 2.0**-16
    working_space: str = "linear_srgb_d65"
    matrix_minimum_determinant: float = 0.2
    minimum_endpoint_span: float = 0.05

    def __post_init__(self) -> None:
        for field in ("capture_matrix", "dye_absorption_matrix", "print_matrix"):
            matrix = _readonly_matrix(getattr(self, field), field)
            if np.any(matrix < 0.0):
                raise ValueError(f"{field} must be non-negative")
            if np.max(np.abs(matrix.sum(axis=1) - 1.0)) > 1e-12:
                raise ValueError(f"{field} must be row-stochastic")
            if float(np.linalg.det(matrix)) < self.matrix_minimum_determinant:
                raise ValueError(f"{field} determinant is below the contract")
            object.__setattr__(self, field, matrix)
        for field in (
            "negative_midpoints",
            "negative_slopes",
            "negative_maximum_densities",
            "paper_midpoints",
            "paper_slopes",
            "paper_maximum_densities",
        ):
            object.__setattr__(
                self,
                field,
                _readonly_vector(getattr(self, field), field),
            )
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 density operator requires linear_srgb_d65")
        if not np.isfinite(self.exposure_floor) or self.exposure_floor != 2.0**-16:
            raise ValueError("v1 exposure_floor must equal 2^-16")
        if not 0.0 < self.matrix_minimum_determinant < 1.0:
            raise ValueError("matrix_minimum_determinant must be in (0, 1)")
        if self.minimum_endpoint_span <= 0.0:
            raise ValueError("minimum_endpoint_span must be positive")
        for midpoint in (self.negative_midpoints, self.paper_midpoints):
            if np.any(midpoint < -12.0) or np.any(midpoint > 2.0):
                raise ValueError("log2 midpoints must be in [-12, 2]")
        for slope in (self.negative_slopes, self.paper_slopes):
            if np.any(slope < 0.2) or np.any(slope > 4.0):
                raise ValueError("logistic slopes must be in [0.2, 4.0]")
        for density in (
            self.negative_maximum_densities,
            self.paper_maximum_densities,
        ):
            if np.any(density < 0.2) or np.any(density > 4.0):
                raise ValueError("maximum densities must be in [0.2, 4.0]")
        black, white = self._raw_endpoints()
        span = white - black
        if np.any(span < self.minimum_endpoint_span):
            raise ValueError(
                f"theoretical endpoint span is below the contract: {span.tolist()}"
            )

    def _raw_reflectance(self, linear_rgb: np.ndarray) -> np.ndarray:
        layer_exposure = linear_rgb @ self.capture_matrix.T
        negative_log_exposure = np.log2(layer_exposure + self.exposure_floor)
        negative_density = logistic_density(
            negative_log_exposure,
            self.negative_midpoints,
            self.negative_slopes,
            self.negative_maximum_densities,
        )
        return density_to_print_reflectance(
            negative_density,
            self.dye_absorption_matrix,
            self.print_matrix,
            self.paper_midpoints,
            self.paper_slopes,
            self.paper_maximum_densities,
            exposure_floor=self.exposure_floor,
        )

    def _raw_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        endpoints = self._raw_reflectance(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        )
        return endpoints[0], endpoints[1]

    @property
    def endpoint_span(self) -> np.ndarray:
        black, white = self._raw_endpoints()
        return white - black

    def apply(self, linear_rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        rgb = np.asarray(linear_rgb, dtype=np.float64)
        if (
            rgb.ndim < 2
            or rgb.shape[-1] != 3
            or not np.all(np.isfinite(rgb))
            or np.any(rgb < 0.0)
            or np.any(rgb > 1.0)
        ):
            raise ValueError("linear_rgb must be finite [0, 1] data with shape (..., 3)")
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        black, white = self._raw_endpoints()
        full = (self._raw_reflectance(rgb) - black) / (white - black)
        if np.any(full < -1e-12) or np.any(full > 1.0 + 1e-12):
            raise RuntimeError("density operator escaped theoretical endpoints")
        if strength == 1.0:
            return full
        if strength == 0.0:
            return rgb.copy()
        return rgb + strength * (full - rgb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DENSITY_OPERATOR_SCHEMA,
            "working_space": self.working_space,
            "exposure_floor": self.exposure_floor,
            "matrix_minimum_determinant": self.matrix_minimum_determinant,
            "minimum_endpoint_span": self.minimum_endpoint_span,
            "capture_matrix": self.capture_matrix.tolist(),
            "negative_midpoints": self.negative_midpoints.tolist(),
            "negative_slopes": self.negative_slopes.tolist(),
            "negative_maximum_densities": self.negative_maximum_densities.tolist(),
            "dye_absorption_matrix": self.dye_absorption_matrix.tolist(),
            "print_matrix": self.print_matrix.tolist(),
            "paper_midpoints": self.paper_midpoints.tolist(),
            "paper_slopes": self.paper_slopes.tolist(),
            "paper_maximum_densities": self.paper_maximum_densities.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DensityDomainNegativePrintOperator":
        if payload.get("schema") != DENSITY_OPERATOR_SCHEMA:
            raise ValueError("unsupported density-domain operator schema")
        return cls(
            capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
            negative_midpoints=np.asarray(payload["negative_midpoints"], dtype=np.float64),
            negative_slopes=np.asarray(payload["negative_slopes"], dtype=np.float64),
            negative_maximum_densities=np.asarray(
                payload["negative_maximum_densities"], dtype=np.float64
            ),
            dye_absorption_matrix=np.asarray(
                payload["dye_absorption_matrix"], dtype=np.float64
            ),
            print_matrix=np.asarray(payload["print_matrix"], dtype=np.float64),
            paper_midpoints=np.asarray(payload["paper_midpoints"], dtype=np.float64),
            paper_slopes=np.asarray(payload["paper_slopes"], dtype=np.float64),
            paper_maximum_densities=np.asarray(
                payload["paper_maximum_densities"], dtype=np.float64
            ),
            exposure_floor=float(payload["exposure_floor"]),
            working_space=str(payload["working_space"]),
            matrix_minimum_determinant=float(payload["matrix_minimum_determinant"]),
            minimum_endpoint_span=float(payload["minimum_endpoint_span"]),
        )


def operator_from_config(
    payload: dict[str, Any],
    *,
    exposure_floor: float = 2.0**-16,
    matrix_minimum_determinant: float = 0.2,
    minimum_endpoint_span: float = 0.05,
) -> DensityDomainNegativePrintOperator:
    return DensityDomainNegativePrintOperator(
        capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
        negative_midpoints=np.asarray(payload["negative_midpoints"], dtype=np.float64),
        negative_slopes=np.asarray(payload["negative_slopes"], dtype=np.float64),
        negative_maximum_densities=np.asarray(
            payload["negative_maximum_densities"], dtype=np.float64
        ),
        dye_absorption_matrix=np.asarray(
            payload["dye_absorption_matrix"], dtype=np.float64
        ),
        print_matrix=np.asarray(payload["print_matrix"], dtype=np.float64),
        paper_midpoints=np.asarray(payload["paper_midpoints"], dtype=np.float64),
        paper_slopes=np.asarray(payload["paper_slopes"], dtype=np.float64),
        paper_maximum_densities=np.asarray(
            payload["paper_maximum_densities"], dtype=np.float64
        ),
        exposure_floor=exposure_floor,
        matrix_minimum_determinant=matrix_minimum_determinant,
        minimum_endpoint_span=minimum_endpoint_span,
    )


def finite_difference_jacobians(
    operator: DensityDomainNegativePrintOperator,
    points: np.ndarray,
    *,
    step: float,
    strength: float = 1.0,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or step <= 0.0
        or np.any(values < step)
        or np.any(values > 1.0 - step)
    ):
        raise ValueError("points must be finite interior RGB rows for the given step")
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        forward = operator.apply(values + offset, strength=strength)
        backward = operator.apply(values - offset, strength=strength)
        columns.append((forward - backward) / (2.0 * step))
    return np.stack(columns, axis=-1)
