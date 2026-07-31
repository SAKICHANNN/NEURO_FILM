"""Explicit density-domain Callier illumination-geometry primitive."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CallierProfile:
    """Synthetic, non-calibrated parameters for a bounded Callier Q family."""

    profile_id: str
    peak_diffuse_density: float
    peak_q_increment: float
    wavelength_weights_rgb: tuple[float, float, float]
    maximum_component_density: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("Callier profile_id must be non-empty")
        if (
            not math.isfinite(self.peak_diffuse_density)
            or self.peak_diffuse_density <= 0.0
        ):
            raise ValueError("Callier peak density must be finite and positive")
        if (
            not math.isfinite(self.peak_q_increment)
            or self.peak_q_increment < 0.0
            or self.peak_q_increment > 1.0
        ):
            raise ValueError("Callier peak Q increment must be in [0, 1]")
        if (
            len(self.wavelength_weights_rgb) != 3
            or any(
                not math.isfinite(value) or value <= 0.0 or value > 1.0
                for value in self.wavelength_weights_rgb
            )
            or not (
                self.wavelength_weights_rgb[0]
                < self.wavelength_weights_rgb[1]
                < self.wavelength_weights_rgb[2]
            )
        ):
            raise ValueError(
                "Callier wavelength weights must satisfy 0 < red < green < blue <= 1"
            )
        if (
            not math.isfinite(self.maximum_component_density)
            or self.maximum_component_density <= self.peak_diffuse_density
        ):
            raise ValueError("Callier maximum density must exceed peak density")


def _validate_collimation(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0
    ):
        raise ValueError("Callier collimation must be finite in [0, 1]")
    return float(value)


def _validate_density(
    values: np.ndarray, profile: CallierProfile, *, name: str
) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must be float32 or float64")
    if source.ndim < 1 or source.shape[-1] != 3:
        raise ValueError(f"{name} must end in three RGB density channels")
    if (
        not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > profile.maximum_component_density)
    ):
        raise ValueError(
            f"{name} must be finite in [0, maximum_component_density]"
        )
    return np.asarray(source, dtype=np.float64)


def callier_q_factor(
    silver_diffuse_density: np.ndarray,
    profile: CallierProfile,
    *,
    collimation: float,
) -> np.ndarray:
    """Return Q>=1 for explicit metallic-silver diffuse density."""

    density = _validate_density(
        silver_diffuse_density, profile, name="silver diffuse density"
    )
    geometry = _validate_collimation(collimation)
    x = density / profile.peak_diffuse_density
    weights = np.asarray(profile.wavelength_weights_rgb, dtype=np.float64)
    response = x * np.exp(1.0 - x)
    return 1.0 + geometry * profile.peak_q_increment * weights * response


def apply_callier_components(
    dye_diffuse_density: np.ndarray,
    silver_diffuse_density: np.ndarray,
    profile: CallierProfile,
    *,
    collimation: float,
) -> np.ndarray:
    """Map separate dye and silver components to directed total density."""

    dye = _validate_density(dye_diffuse_density, profile, name="dye density")
    silver = _validate_density(
        silver_diffuse_density, profile, name="silver diffuse density"
    )
    if dye.shape != silver.shape:
        raise ValueError("Callier dye and silver density shapes must match")
    q = callier_q_factor(silver, profile, collimation=collimation)
    return dye + silver * q


def invert_callier_components(
    directed_total_density: np.ndarray,
    dye_density: np.ndarray,
    profile: CallierProfile,
    *,
    collimation: float,
    iterations: int = 64,
) -> np.ndarray:
    """Recover silver diffuse density for a known invariant dye component."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Callier inverse iterations must be a positive integer")
    dye = _validate_density(dye_density, profile, name="dye density")
    directed = np.asarray(directed_total_density)
    if directed.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("directed total density must be float32 or float64")
    if directed.shape != dye.shape:
        raise ValueError("directed total and dye density shapes must match")
    directed = np.asarray(directed, dtype=np.float64)
    if not np.all(np.isfinite(directed)) or np.any(directed < dye):
        raise ValueError("directed total density must be finite and at least dye density")
    geometry = _validate_collimation(collimation)
    target = directed - dye
    maximum = np.full_like(target, profile.maximum_component_density)
    maximum_directed = maximum * callier_q_factor(
        maximum, profile, collimation=geometry
    )
    tolerance = np.finfo(np.float64).eps * np.maximum(1.0, maximum_directed)
    if np.any(target > maximum_directed + tolerance):
        raise ValueError("directed total density exceeds invertible Callier support")
    if geometry == 0.0:
        return target.copy()
    low = np.zeros_like(target)
    high = maximum
    for _ in range(iterations):
        middle = (low + high) * 0.5
        mapped = middle * callier_q_factor(
            middle, profile, collimation=geometry
        )
        choose_upper = mapped < target
        low = np.where(choose_upper, middle, low)
        high = np.where(choose_upper, high, middle)
    result = (low + high) * 0.5
    result[target == 0.0] = 0.0
    return result


__all__ = [
    "CallierProfile",
    "apply_callier_components",
    "callier_q_factor",
    "invert_callier_components",
]
