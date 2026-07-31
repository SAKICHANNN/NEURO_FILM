"""Exposure-domain Poisson variance propagated through a characteristic curve."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HillCharacteristicProfile:
    """Generic analytic characteristic-curve witness, not a stock profile."""

    density_min: float
    density_span: float
    half_exposure: float
    gamma: float

    def __post_init__(self) -> None:
        values = (
            self.density_min,
            self.density_span,
            self.half_exposure,
            self.gamma,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.density_min < 0.0
            or self.density_span <= 0.0
            or self.half_exposure <= 0.0
            or self.gamma <= 0.5
        ):
            raise ValueError("invalid Hill characteristic profile")

    def density(self, exposure: np.ndarray) -> np.ndarray:
        values = _positive_exposure(exposure)
        ratio = np.power(values / self.half_exposure, self.gamma)
        return self.density_min + self.density_span * ratio / (1.0 + ratio)

    def derivative(self, exposure: np.ndarray) -> np.ndarray:
        values = _positive_exposure(exposure)
        ratio = np.power(values / self.half_exposure, self.gamma)
        return (
            self.density_span
            * self.gamma
            * ratio
            / (values * np.square(1.0 + ratio))
        )

    @property
    def variance_peak_exposure(self) -> float:
        ratio = (2.0 * self.gamma - 1.0) / (2.0 * self.gamma + 1.0)
        return self.half_exposure * math.pow(ratio, 1.0 / self.gamma)


def _positive_exposure(exposure: np.ndarray) -> np.ndarray:
    values = np.asarray(exposure, dtype=np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or values.size == 0
    ):
        raise ValueError("exposure must be finite, nonempty and positive")
    return values


def developed_density_variance(
    exposure: np.ndarray,
    profile: HillCharacteristicProfile,
    *,
    photon_scale: float,
) -> np.ndarray:
    """Delta-method variance for Poisson exposure counts before development."""

    values = _positive_exposure(exposure)
    scale = float(photon_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("photon scale must be finite and positive")
    derivative = profile.derivative(values)
    variance = values * np.square(derivative) / scale
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise RuntimeError("developed-density variance escaped its domain")
    return variance


__all__ = [
    "HillCharacteristicProfile",
    "developed_density_variance",
]
