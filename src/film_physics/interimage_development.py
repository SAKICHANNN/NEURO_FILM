"""Bounded cross-layer interimage development in optical-density space.

This is a generic physical-inspired representation.  It does not contain a
measured stock, process, or scanner profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


INTERIMAGE_DEVELOPMENT_SCHEMA = (
    "neuro_film.physical_interimage_development_operator.v1"
)
_OFF_DIAGONAL = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)


def _vector3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite length-three vector")
    return array


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    output = np.empty_like(value, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


@dataclass(frozen=True)
class InterimageDevelopmentOperator:
    density_min: tuple[float, float, float]
    density_max: tuple[float, float, float]
    slope: tuple[float, float, float]
    midpoint: tuple[float, float, float]
    coupling: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    maximum_coupling: float = 0.8

    def __post_init__(self) -> None:
        minimum = _vector3(self.density_min, "density_min")
        maximum = _vector3(self.density_max, "density_max")
        slope = _vector3(self.slope, "slope")
        midpoint = _vector3(self.midpoint, "midpoint")
        coupling = np.asarray(self.coupling, dtype=np.float64)
        cap = float(self.maximum_coupling)
        if np.any(minimum < 0.0) or np.any(maximum <= minimum):
            raise ValueError("density bounds must be ordered and nonnegative")
        if np.any(slope <= 0.0):
            raise ValueError("slopes must be positive")
        if coupling.shape != (3, 3) or not np.all(np.isfinite(coupling)):
            raise ValueError("coupling must be a finite 3x3 matrix")
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("maximum_coupling must be finite and positive")
        if not np.array_equal(np.diag(coupling), np.zeros(3)):
            raise ValueError("coupling diagonal must be exactly zero")
        if np.any(coupling < 0.0) or np.any(coupling > cap):
            raise ValueError("coupling must be nonnegative and bounded")
        object.__setattr__(self, "density_min", tuple(float(x) for x in minimum))
        object.__setattr__(self, "density_max", tuple(float(x) for x in maximum))
        object.__setattr__(self, "slope", tuple(float(x) for x in slope))
        object.__setattr__(self, "midpoint", tuple(float(x) for x in midpoint))
        object.__setattr__(
            self,
            "coupling",
            tuple(tuple(float(x) for x in row) for row in coupling),
        )
        object.__setattr__(self, "maximum_coupling", cap)

    def apply_log2_exposure(self, log2_exposure: np.ndarray) -> np.ndarray:
        values = np.asarray(log2_exposure, dtype=np.float64)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("log2 exposure must be a finite (...,3) array")
        slope = np.asarray(self.slope)
        midpoint = np.asarray(self.midpoint)
        activation = slope * (values - midpoint)
        developed_fraction = _sigmoid(activation)
        inhibition = developed_fraction @ np.asarray(self.coupling).T
        response = _sigmoid(activation - inhibition)
        minimum = np.asarray(self.density_min)
        span = np.asarray(self.density_max) - minimum
        return minimum + span * response

    def apply_relative_exposure(self, exposure: np.ndarray) -> np.ndarray:
        values = np.asarray(exposure, dtype=np.float64)
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("relative exposure must be finite and positive")
        return self.apply_log2_exposure(np.log2(values))

    def derivative_log2_exposure(
        self, log2_exposure: np.ndarray
    ) -> np.ndarray:
        """Return the analytical output-layer by input-layer Jacobian."""
        values = np.asarray(log2_exposure, dtype=np.float64)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("log2 exposure must be a finite (...,3) array")
        slope = np.asarray(self.slope)
        activation = slope * (values - np.asarray(self.midpoint))
        activated = _sigmoid(activation)
        coupling = np.asarray(self.coupling)
        inhibited = activation - activated @ coupling.T
        response = _sigmoid(inhibited)
        span = np.asarray(self.density_max) - np.asarray(self.density_min)
        response_derivative = span * response * (1.0 - response)
        activated_derivative = activated * (1.0 - activated) * slope
        shape = values.shape[:-1] + (3, 3)
        jacobian = np.empty(shape, dtype=np.float64)
        for output in range(3):
            for source in range(3):
                if output == source:
                    latent_derivative = slope[output]
                else:
                    latent_derivative = (
                        -coupling[output, source]
                        * activated_derivative[..., source]
                    )
                jacobian[..., output, source] = (
                    response_derivative[..., output] * latent_derivative
                )
        return jacobian

    def with_coupling_vector(
        self, values: np.ndarray
    ) -> "InterimageDevelopmentOperator":
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (6,):
            raise ValueError("coupling vector must have six entries")
        coupling = np.zeros((3, 3), dtype=np.float64)
        for value, (row, column) in zip(vector, _OFF_DIAGONAL, strict=True):
            coupling[row, column] = value
        return InterimageDevelopmentOperator(
            self.density_min,
            self.density_max,
            self.slope,
            self.midpoint,
            tuple(tuple(float(x) for x in row) for row in coupling),
            self.maximum_coupling,
        )

    def coupling_vector(self) -> np.ndarray:
        coupling = np.asarray(self.coupling)
        return np.asarray(
            [coupling[row, column] for row, column in _OFF_DIAGONAL],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERIMAGE_DEVELOPMENT_SCHEMA,
            "density_min": list(self.density_min),
            "density_max": list(self.density_max),
            "slope": list(self.slope),
            "midpoint": list(self.midpoint),
            "coupling": [list(row) for row in self.coupling],
            "maximum_coupling": self.maximum_coupling,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterimageDevelopmentOperator":
        if payload.get("schema") != INTERIMAGE_DEVELOPMENT_SCHEMA:
            raise ValueError("unsupported interimage development schema")
        return cls(
            tuple(payload["density_min"]),
            tuple(payload["density_max"]),
            tuple(payload["slope"]),
            tuple(payload["midpoint"]),
            tuple(tuple(row) for row in payload["coupling"]),
            float(payload["maximum_coupling"]),
        )


def independent_development_operator(
    operator: InterimageDevelopmentOperator,
) -> InterimageDevelopmentOperator:
    return operator.with_coupling_vector(np.zeros(6, dtype=np.float64))

