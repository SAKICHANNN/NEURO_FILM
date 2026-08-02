"""Offline random-phase fields for typed historical measured spectra."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.structure_compiler import counter_normal_region


@dataclass(frozen=True)
class MeasuredNPSFieldPair:
    """Intrinsic and source-aperture-observed periodic reference fields."""

    intrinsic: np.ndarray
    aperture_observed: np.ndarray
    intrinsic_ifft_imaginary_residual: float
    aperture_ifft_imaginary_residual: float
    construction_fourier_outside_band_exact_zero: bool

    def field_hashes(self) -> tuple[str, str]:
        return (
            hashlib.sha256(np.ascontiguousarray(self.intrinsic).tobytes()).hexdigest(),
            hashlib.sha256(
                np.ascontiguousarray(self.aperture_observed).tobytes()
            ).hexdigest(),
        )


def physical_frequency_grids(
    shape: tuple[int, int], sample_pitch_millimetres: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        len(shape) != 2
        or any(not isinstance(value, int) or value <= 1 for value in shape)
        or not math.isfinite(sample_pitch_millimetres)
        or sample_pitch_millimetres <= 0.0
    ):
        raise ValueError("invalid physical FFT grid")
    fy_axis = np.fft.fftfreq(shape[0], d=sample_pitch_millimetres)
    fx_axis = np.fft.fftfreq(shape[1], d=sample_pitch_millimetres)
    fy = np.broadcast_to(fy_axis[:, None], shape)
    fx = np.broadcast_to(fx_axis[None, :], shape)
    return fy, fx, np.hypot(fx, fy)


def target_nps_grids(
    profile: HistoricalBWNoiseSpectrumProfile,
    shape: tuple[int, int],
    sample_pitch_millimetres: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy, fx, radius = physical_frequency_grids(shape, sample_pitch_millimetres)
    supported = radius <= profile.maximum_intrinsic_frequency_lines_per_mm
    intrinsic = np.zeros(shape, dtype=np.float64)
    aperture_observed = np.zeros(shape, dtype=np.float64)
    non_dc = supported & (radius > 0.0)
    intrinsic[non_dc] = profile.intrinsic_2d(fx[non_dc], fy[non_dc])
    aperture_observed[non_dc] = profile.aperture_convolved_2d(
        fx[non_dc], fy[non_dc]
    )
    return intrinsic, aperture_observed, supported


def synthesize_measured_nps_field_pair(
    profile: HistoricalBWNoiseSpectrumProfile,
    *,
    shape: tuple[int, int],
    sample_pitch_millimetres: float,
    seed: int,
) -> MeasuredNPSFieldPair:
    """Synthesize one fixed-amplitude random-phase second-order reference."""

    if not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    intrinsic_target, observed_target, supported = target_nps_grids(
        profile, shape, sample_pitch_millimetres
    )
    white = counter_normal_region(
        shape, origin_yx=(0, 0), shape=shape, seed=seed
    )
    white_fourier = np.fft.fft2(white)
    magnitude = np.abs(white_fourier)
    phase = np.divide(
        white_fourier,
        magnitude,
        out=np.zeros_like(white_fourier),
        where=magnitude > 0.0,
    )
    pixel_count = shape[0] * shape[1]
    intrinsic_fourier = (
        phase
        * np.sqrt(pixel_count * intrinsic_target)
        / sample_pitch_millimetres
    )
    intrinsic_fourier[0, 0] = 0.0
    _, _, radius = physical_frequency_grids(shape, sample_pitch_millimetres)
    aperture_mtf = profile.circular_aperture_mtf(radius)
    observed_fourier = intrinsic_fourier * aperture_mtf
    observed_fourier[0, 0] = 0.0
    construction_zero = bool(
        np.all(intrinsic_fourier[~supported] == 0.0)
        and np.all(observed_fourier[~supported] == 0.0)
        and np.all(intrinsic_target[~supported] == 0.0)
        and np.all(observed_target[~supported] == 0.0)
    )
    intrinsic_complex = np.fft.ifft2(intrinsic_fourier)
    observed_complex = np.fft.ifft2(observed_fourier)
    return MeasuredNPSFieldPair(
        intrinsic=np.ascontiguousarray(intrinsic_complex.real, dtype=np.float64),
        aperture_observed=np.ascontiguousarray(
            observed_complex.real, dtype=np.float64
        ),
        intrinsic_ifft_imaginary_residual=float(
            np.max(np.abs(intrinsic_complex.imag))
        ),
        aperture_ifft_imaginary_residual=float(
            np.max(np.abs(observed_complex.imag))
        ),
        construction_fourier_outside_band_exact_zero=construction_zero,
    )


def physical_periodogram(
    field: np.ndarray, sample_pitch_millimetres: float
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or not math.isfinite(sample_pitch_millimetres)
        or sample_pitch_millimetres <= 0.0
    ):
        raise ValueError("invalid field periodogram request")
    pixel_count = values.shape[0] * values.shape[1]
    return (
        np.square(np.abs(np.fft.fft2(values)))
        * sample_pitch_millimetres**2
        / pixel_count
    )


__all__ = [
    "MeasuredNPSFieldPair",
    "physical_frequency_grids",
    "physical_periodogram",
    "synthesize_measured_nps_field_pair",
    "target_nps_grids",
]

