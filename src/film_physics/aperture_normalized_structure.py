"""Generic spatial hypotheses normalized to a stated RMS measurement aperture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class ApertureNormalizedSpatialHypothesis:
    """One explicit generic density-noise shape and measurement aperture."""

    hypothesis_id: str
    family: str
    gaussian_sigma_micrometres: float
    sample_pitch_micrometres: float
    aperture_diameter_micrometres: float
    gaussian_truncate_sigma: float = 4.0

    def __post_init__(self) -> None:
        if self.family not in {"delta", "gaussian"}:
            raise ValueError("unsupported spatial hypothesis family")
        if not self.hypothesis_id or any(
            not character.islower() and not character.isdigit() and character != "_"
            for character in self.hypothesis_id
        ):
            raise ValueError("invalid spatial hypothesis identity")
        values = (
            self.gaussian_sigma_micrometres,
            self.sample_pitch_micrometres,
            self.aperture_diameter_micrometres,
            self.gaussian_truncate_sigma,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("spatial hypothesis values must be finite")
        if (
            self.sample_pitch_micrometres <= 0.0
            or self.aperture_diameter_micrometres <= 0.0
            or self.gaussian_truncate_sigma <= 0.0
            or self.gaussian_sigma_micrometres < 0.0
            or (self.family == "delta" and self.gaussian_sigma_micrometres != 0.0)
            or (self.family == "gaussian" and self.gaussian_sigma_micrometres <= 0.0)
        ):
            raise ValueError("invalid spatial hypothesis geometry")

    @property
    def spatial_radius_samples(self) -> int:
        if self.family == "delta":
            return 0
        sigma_samples = self.gaussian_sigma_micrometres / self.sample_pitch_micrometres
        return int(self.gaussian_truncate_sigma * sigma_samples + 0.5)

    @property
    def aperture_radius_samples(self) -> int:
        return round(
            0.5
            * self.aperture_diameter_micrometres
            / self.sample_pitch_micrometres
        )

    def spatial_kernel(self) -> np.ndarray:
        if self.family == "delta":
            result = np.ones((1, 1), dtype=np.float64)
        else:
            radius = self.spatial_radius_samples
            sigma_samples = (
                self.gaussian_sigma_micrometres / self.sample_pitch_micrometres
            )
            coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
            one_dimensional = np.exp(
                -0.5 * np.square(coordinates / sigma_samples)
            )
            one_dimensional /= np.sum(one_dimensional, dtype=np.float64)
            result = np.outer(one_dimensional, one_dimensional)
        result.setflags(write=False)
        return result

    def aperture_kernel(self) -> np.ndarray:
        radius = self.aperture_radius_samples
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        physical_radius = 0.5 * self.aperture_diameter_micrometres
        result = (
            (xx * self.sample_pitch_micrometres) ** 2
            + (yy * self.sample_pitch_micrometres) ** 2
            <= physical_radius * physical_radius
        ).astype(np.float64)
        result /= np.sum(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def combined_measurement_kernel(self) -> np.ndarray:
        result = fftconvolve(
            self.spatial_kernel(), self.aperture_kernel(), mode="full"
        )
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def measurement_energy(self) -> float:
        combined = self.combined_measurement_kernel()
        energy = float(np.sum(np.square(combined), dtype=np.float64))
        if not math.isfinite(energy) or energy <= 0.0:
            raise RuntimeError("invalid spatial measurement energy")
        return energy

    def innovation_sigma(self, target_sigma_d: float) -> float:
        target = float(target_sigma_d)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("target Sigma-D must be positive")
        return target / math.sqrt(self.measurement_energy())

    def analytic_aperture_sigma(self, innovation_sigma: float) -> float:
        scale = float(innovation_sigma)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("innovation sigma must be positive")
        return scale * math.sqrt(self.measurement_energy())

    def power_transfer(self, cycles_per_micrometre: float) -> float:
        frequency = float(cycles_per_micrometre)
        if not math.isfinite(frequency) or frequency < 0.0:
            raise ValueError("spatial frequency must be finite and nonnegative")
        kernel = self.spatial_kernel()
        coordinates = (
            np.arange(kernel.shape[1], dtype=np.float64) - kernel.shape[1] // 2
        ) * self.sample_pitch_micrometres
        transfer = np.sum(
            kernel
            * np.exp(-2j * np.pi * frequency * coordinates)[None, :],
            dtype=np.complex128,
        )
        return float(abs(transfer) ** 2)


__all__ = ["ApertureNormalizedSpatialHypothesis"]
