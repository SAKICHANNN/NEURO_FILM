"""Amplitude-matched saturating donor for generic interimage development.

This independent physical-inspired primitive contains no measured stock,
process, scanner, or external profile parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interimage_development import InterimageDevelopmentOperator


def _vector3(value: object, label: str, *, allow_infinite: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(array)
    valid = finite | (allow_infinite & np.isposinf(array))
    if array.shape != (3,) or not np.all(valid):
        qualifier = "positive-finite-or-infinite" if allow_infinite else "finite"
        raise ValueError(f"{label} must be a {qualifier} length-three vector")
    return array


def _sigmoid(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class LangmuirDonorProfile:
    density_capacity: tuple[float, float, float]
    knee_fraction: tuple[float, float, float]
    match_fraction: tuple[float, float, float]

    def __post_init__(self) -> None:
        capacity = _vector3(self.density_capacity, "density_capacity")
        knee = _vector3(self.knee_fraction, "knee_fraction", allow_infinite=True)
        match = _vector3(self.match_fraction, "match_fraction")
        if np.any(capacity <= 0.0) or np.any(knee <= 0.0):
            raise ValueError("density capacity and knee fraction must be positive")
        if np.any(match <= 0.0) or np.any(match >= 1.0):
            raise ValueError("match fraction must lie strictly inside zero and one")
        object.__setattr__(self, "density_capacity", tuple(float(x) for x in capacity))
        object.__setattr__(self, "knee_fraction", tuple(float(x) for x in knee))
        object.__setattr__(self, "match_fraction", tuple(float(x) for x in match))

    @property
    def knee(self) -> np.ndarray:
        return np.asarray(self.knee_fraction) * np.asarray(self.density_capacity)

    @property
    def match_density(self) -> np.ndarray:
        return np.asarray(self.match_fraction) * np.asarray(self.density_capacity)

    def apply(self, density: np.ndarray) -> np.ndarray:
        values = np.asarray(density, dtype=np.float64)
        capacity = np.asarray(self.density_capacity)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("donor density must be a finite (...,3) array")
        if np.any(values < 0.0) or np.any(values > capacity):
            raise ValueError("donor density lies outside its declared capacity")
        knee = self.knee
        finite = np.isfinite(knee)
        safe_knee = np.where(finite, knee, 1.0)
        saturated = values * (safe_knee + self.match_density) / (safe_knee + values)
        return np.where(finite, saturated, values)

    def derivative(self, density: np.ndarray) -> np.ndarray:
        values = np.asarray(density, dtype=np.float64)
        self.apply(values)
        knee = self.knee
        finite = np.isfinite(knee)
        safe_knee = np.where(finite, knee, 1.0)
        derivative = (
            safe_knee * (safe_knee + self.match_density) / (safe_knee + values) ** 2
        )
        return np.where(finite, derivative, 1.0)


def apply_langmuir_interimage_development(
    operator: InterimageDevelopmentOperator,
    log2_exposure: np.ndarray,
    donor: LangmuirDonorProfile,
) -> np.ndarray:
    values = np.asarray(log2_exposure, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("log2 exposure must be a finite (...,3) array")
    slope = np.asarray(operator.slope)
    activation = slope * (values - np.asarray(operator.midpoint))
    developed_fraction = _sigmoid(activation)
    inhibition = donor.apply(developed_fraction) @ np.asarray(operator.coupling).T
    response = _sigmoid(activation - inhibition)
    minimum = np.asarray(operator.density_min)
    span = np.asarray(operator.density_max) - minimum
    return minimum + span * response


def langmuir_interimage_jacobian(
    operator: InterimageDevelopmentOperator,
    log2_exposure: np.ndarray,
    donor: LangmuirDonorProfile,
) -> np.ndarray:
    values = np.asarray(log2_exposure, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("log2 exposure must be a finite (...,3) array")
    slope = np.asarray(operator.slope)
    activation = slope * (values - np.asarray(operator.midpoint))
    developed = _sigmoid(activation)
    donor_values = donor.apply(developed)
    coupling = np.asarray(operator.coupling)
    latent = activation - donor_values @ coupling.T
    response = _sigmoid(latent)
    span = np.asarray(operator.density_max) - np.asarray(operator.density_min)
    response_derivative = span * response * (1.0 - response)
    donor_derivative = donor.derivative(developed)
    developed_derivative = developed * (1.0 - developed) * slope
    jacobian = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    for output in range(3):
        for source in range(3):
            latent_derivative = slope[output] if output == source else 0.0
            latent_derivative -= (
                coupling[output, source]
                * donor_derivative[..., source]
                * developed_derivative[..., source]
            )
            jacobian[..., output, source] = (
                response_derivative[..., output] * latent_derivative
            )
    return jacobian


__all__ = [
    "LangmuirDonorProfile",
    "apply_langmuir_interimage_development",
    "langmuir_interimage_jacobian",
]
