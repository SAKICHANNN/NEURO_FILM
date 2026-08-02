"""Compose a typed Sigma-D profile with a scanner-grid Thomas field."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.finite_support_thomas import (
    gaussian_kernel_1d,
    kernel_variance_2d,
)
from src.film_physics.granularity_amplitude import (
    CHANNELS,
    GranularityAmplitudeProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import (
    ThomasDcReceipt,
    render_dc_projected_thomas_region,
)

PROFILE_SCHEMA = "neuro_film.density_conditioned_thomas_profile.v1"


def binary_circular_aperture_kernel(
    sample_pitch_micrometres: float, aperture_diameter_micrometres: float
) -> np.ndarray:
    pitch = float(sample_pitch_micrometres)
    diameter = float(aperture_diameter_micrometres)
    if (
        not math.isfinite(pitch)
        or pitch <= 0.0
        or not math.isfinite(diameter)
        or diameter <= 0.0
    ):
        raise ValueError("invalid measurement aperture geometry")
    radius = round(0.5 * diameter / pitch)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    aperture_radius = 0.5 * diameter
    kernel = (
        np.square(xx * pitch) + np.square(yy * pitch)
        <= aperture_radius * aperture_radius
    ).astype(np.float64)
    kernel /= float(np.sum(kernel, dtype=np.float64))
    kernel.setflags(write=False)
    return kernel


def thomas_aperture_measurement_energy(
    *,
    particle_sigma_samples: float,
    cluster_sigma_samples: float,
    mean_offspring: float,
    truncate: float,
    aperture_kernel: np.ndarray,
) -> float:
    """Return aperture variance for the analytically unit point-sample field."""
    if not math.isfinite(mean_offspring) or mean_offspring <= 0.0:
        raise ValueError("mean offspring must be finite and positive")
    combined_sigma = math.hypot(particle_sigma_samples, cluster_sigma_samples)
    particle_1d = gaussian_kernel_1d(particle_sigma_samples, truncate)
    combined_1d = gaussian_kernel_1d(combined_sigma, truncate)
    particle = np.outer(particle_1d, particle_1d)
    combined = np.outer(combined_1d, combined_1d)
    numerator = float(
        np.sum(
            np.square(fftconvolve(particle, aperture_kernel, mode="full")),
            dtype=np.float64,
        )
        + mean_offspring
        * np.sum(
            np.square(fftconvolve(combined, aperture_kernel, mode="full")),
            dtype=np.float64,
        )
    )
    denominator = kernel_variance_2d(
        particle_sigma_samples, truncate
    ) + mean_offspring * kernel_variance_2d(combined_sigma, truncate)
    energy = numerator / denominator
    if not math.isfinite(energy) or energy <= 0.0 or energy > 1.0:
        raise RuntimeError("invalid Thomas aperture measurement energy")
    return energy


@dataclass(frozen=True)
class DensityConditionedThomasProfile:
    amplitude_profile: GranularityAmplitudeProfile
    spatial_profile_id: str
    particle_sigma_samples: float
    cluster_sigma_samples: float
    mean_offspring: float
    component_seeds: tuple[int, int]
    sample_pitch_micrometres: float = 6.35
    measurement_aperture_diameter_micrometres: float = 48.0
    truncate: float = 4.0
    input_domain: str = "relative_layer_log_exposure"
    output_domain: str = (
        "status_m_developed_density_perturbation_on_4000dpi_scanner_grid"
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.amplitude_profile, GranularityAmplitudeProfile)
            or not self.spatial_profile_id
            or len(self.component_seeds) != 2
            or self.sample_pitch_micrometres != 6.35
            or self.measurement_aperture_diameter_micrometres != 48.0
            or self.truncate != 4.0
            or self.input_domain != "relative_layer_log_exposure"
            or self.output_domain
            != "status_m_developed_density_perturbation_on_4000dpi_scanner_grid"
        ):
            raise ValueError("invalid density-conditioned Thomas profile")
        _ = self.measurement_energy()

    def aperture_kernel(self) -> np.ndarray:
        return binary_circular_aperture_kernel(
            self.sample_pitch_micrometres,
            self.measurement_aperture_diameter_micrometres,
        )

    def measurement_energy(self) -> float:
        return thomas_aperture_measurement_energy(
            particle_sigma_samples=self.particle_sigma_samples,
            cluster_sigma_samples=self.cluster_sigma_samples,
            mean_offspring=self.mean_offspring,
            truncate=self.truncate,
            aperture_kernel=self.aperture_kernel(),
        )

    def target_sigma_d(
        self,
        prior: ManufacturerCharacteristicPrior,
        channel: str,
        relative_log_exposure: float,
    ) -> float:
        result = self.amplitude_profile.evaluate_channel(
            prior, channel, np.asarray([relative_log_exposure], dtype=np.float64)
        )
        return float(result[0])

    def point_scale(
        self,
        prior: ManufacturerCharacteristicPrior,
        channel: str,
        relative_log_exposure: float,
    ) -> float:
        return self.target_sigma_d(
            prior, channel, relative_log_exposure
        ) / math.sqrt(self.measurement_energy())

    def analytic_aperture_sigma(
        self,
        prior: ManufacturerCharacteristicPrior,
        channel: str,
        relative_log_exposure: float,
    ) -> float:
        return self.point_scale(
            prior, channel, relative_log_exposure
        ) * math.sqrt(self.measurement_energy())

    def _validate_receipt(self, receipt: ThomasDcReceipt) -> None:
        if (
            receipt.profile_id != self.spatial_profile_id
            or receipt.particle_sigma_pixels != self.particle_sigma_samples
            or receipt.cluster_sigma_pixels != self.cluster_sigma_samples
            or receipt.mean_offspring != self.mean_offspring
            or receipt.component_seeds != self.component_seeds
            or receipt.truncate != self.truncate
        ):
            raise ValueError("Thomas DC receipt does not match density profile")

    def render_developed_density_region(
        self,
        receipt: ThomasDcReceipt,
        prior: ManufacturerCharacteristicPrior,
        *,
        channel: str,
        relative_log_exposure: float,
        origin_yx: tuple[int, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        self._validate_receipt(receipt)
        if channel not in CHANNELS:
            raise ValueError("unsupported density layer")
        index = CHANNELS.index(channel)
        density_mean = float(
            prior.curves[index].apply(
                np.asarray([relative_log_exposure], dtype=np.float64)
            )[0]
        )
        unit = render_dc_projected_thomas_region(
            receipt, origin_yx=origin_yx, shape=shape
        )
        result = np.ascontiguousarray(
            density_mean
            + self.point_scale(prior, channel, relative_log_exposure) * unit,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("developed density field is non-finite")
        result.setflags(write=False)
        return result

    def render_nonstationary_developed_density_region(
        self,
        receipt: ThomasDcReceipt,
        prior: ManufacturerCharacteristicPrior,
        *,
        channel: str,
        full_relative_log_exposure: np.ndarray,
        origin_yx: tuple[int, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Render one region from a full coordinate-bound layer exposure field."""
        self._validate_receipt(receipt)
        if channel not in CHANNELS:
            raise ValueError("unsupported density layer")
        exposure = np.asarray(full_relative_log_exposure, dtype=np.float64)
        if exposure.shape != receipt.full_shape or not np.all(np.isfinite(exposure)):
            raise ValueError("nonstationary exposure must match the receipt full shape")
        y0, x0 = origin_yx
        height, width = shape
        if not (
            0 <= y0 < y0 + height <= exposure.shape[0]
            and 0 <= x0 < x0 + width <= exposure.shape[1]
        ):
            raise ValueError("nonstationary density region is outside full field")
        selected = exposure[y0 : y0 + height, x0 : x0 + width]
        index = CHANNELS.index(channel)
        density_mean = prior.curves[index].apply(selected)
        sigma_d = self.amplitude_profile.evaluate_channel(
            prior, channel, selected
        )
        unit = render_dc_projected_thomas_region(
            receipt, origin_yx=origin_yx, shape=shape
        )
        result = np.ascontiguousarray(
            density_mean + sigma_d * unit / math.sqrt(self.measurement_energy()),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("nonstationary developed density is non-finite")
        result.setflags(write=False)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "input_domain": self.input_domain,
            "output_domain": self.output_domain,
            "channel_order": list(CHANNELS),
            "amplitude_profile": self.amplitude_profile.to_dict(),
            "spatial_profile_id": self.spatial_profile_id,
            "particle_sigma_samples": self.particle_sigma_samples,
            "cluster_sigma_samples": self.cluster_sigma_samples,
            "mean_offspring": self.mean_offspring,
            "component_seeds": list(self.component_seeds),
            "sample_pitch_micrometres": self.sample_pitch_micrometres,
            "measurement_aperture_diameter_micrometres": (
                self.measurement_aperture_diameter_micrometres
            ),
            "truncate": self.truncate,
            "measurement_energy": self.measurement_energy(),
            "amplitude_semantics": "48um diffuse-rms Status-M density",
            "spatial_semantics": "generic same-scanner effective Thomas spectrum",
            "composition_status": "explicit_hybrid_not_calibrated_stock_spatial_profile",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DensityConditionedThomasProfile:
        if (
            payload.get("schema") != PROFILE_SCHEMA
            or payload.get("channel_order") != list(CHANNELS)
            or payload.get("amplitude_semantics")
            != "48um diffuse-rms Status-M density"
            or payload.get("spatial_semantics")
            != "generic same-scanner effective Thomas spectrum"
            or payload.get("composition_status")
            != "explicit_hybrid_not_calibrated_stock_spatial_profile"
        ):
            raise ValueError("unsupported density-conditioned Thomas profile schema")
        profile = cls(
            amplitude_profile=GranularityAmplitudeProfile.from_dict(
                payload["amplitude_profile"]
            ),
            spatial_profile_id=str(payload["spatial_profile_id"]),
            particle_sigma_samples=float(payload["particle_sigma_samples"]),
            cluster_sigma_samples=float(payload["cluster_sigma_samples"]),
            mean_offspring=float(payload["mean_offspring"]),
            component_seeds=tuple(int(value) for value in payload["component_seeds"]),
            sample_pitch_micrometres=float(payload["sample_pitch_micrometres"]),
            measurement_aperture_diameter_micrometres=float(
                payload["measurement_aperture_diameter_micrometres"]
            ),
            truncate=float(payload["truncate"]),
            input_domain=str(payload["input_domain"]),
            output_domain=str(payload["output_domain"]),
        )
        if payload.get("measurement_energy") != profile.measurement_energy():
            raise ValueError("density-conditioned Thomas measurement energy drift")
        return profile

    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


__all__ = [
    "PROFILE_SCHEMA",
    "DensityConditionedThomasProfile",
    "binary_circular_aperture_kernel",
    "thomas_aperture_measurement_energy",
]
