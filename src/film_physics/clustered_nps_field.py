"""Deterministic periodic fields for an explicit Thomas radial spectrum."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from src.film_physics.structure_compiler import counter_normal_region
from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps


@dataclass(frozen=True)
class ClusteredNPSField:
    values: np.ndarray
    target_spectrum: np.ndarray
    ifft_imaginary_residual: float

    def field_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.values).tobytes()).hexdigest()


def normalized_thomas_spectrum_grid(
    shape: tuple[int, int],
    *,
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
) -> np.ndarray:
    """Return a positive radial target with zero DC and unit discrete mean."""
    if len(shape) != 2 or any(not isinstance(value, int) or value <= 1 for value in shape):
        raise ValueError("invalid clustered field shape")
    for value in (particle_sigma_pixels, cluster_sigma_pixels, mean_offspring):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("clustered field parameters must be finite and positive")
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    target = thomas_cluster_gaussian_mark_nps(
        frequency,
        float(particle_sigma_pixels),
        float(cluster_sigma_pixels),
        float(mean_offspring),
    )
    target[0, 0] = 0.0
    mean = float(np.mean(target, dtype=np.float64))
    if not mean > 0.0:
        raise RuntimeError("clustered spectrum has zero non-DC energy")
    target = np.ascontiguousarray(target / mean, dtype=np.float64)
    target.setflags(write=False)
    return target


def synthesize_clustered_nps_field(
    *,
    shape: tuple[int, int],
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
    seed: int,
) -> ClusteredNPSField:
    """Synthesize a fixed-amplitude random-phase unit-variance reference."""
    if not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    target = normalized_thomas_spectrum_grid(
        shape,
        particle_sigma_pixels=particle_sigma_pixels,
        cluster_sigma_pixels=cluster_sigma_pixels,
        mean_offspring=mean_offspring,
    )
    white = counter_normal_region(shape, origin_yx=(0, 0), shape=shape, seed=seed)
    white_fourier = np.fft.fft2(white)
    magnitude = np.abs(white_fourier)
    phase = np.divide(
        white_fourier,
        magnitude,
        out=np.zeros_like(white_fourier),
        where=magnitude > 0.0,
    )
    coefficients = phase * np.sqrt(target * target.size)
    coefficients[0, 0] = 0.0
    complex_field = np.fft.ifft2(coefficients)
    values = np.ascontiguousarray(complex_field.real, dtype=np.float64)
    values.setflags(write=False)
    return ClusteredNPSField(
        values=values,
        target_spectrum=target,
        ifft_imaginary_residual=float(np.max(np.abs(complex_field.imag))),
    )


def discrete_periodogram(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field must be finite and two-dimensional")
    return np.square(np.abs(np.fft.fft2(values))) / values.size


__all__ = [
    "ClusteredNPSField",
    "discrete_periodogram",
    "normalized_thomas_spectrum_grid",
    "synthesize_clustered_nps_field",
]
