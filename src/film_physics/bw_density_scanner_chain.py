"""Typed neutral B&W developed-density to scanner-linear chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    density_to_transmittance,
)
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_scanner_mtf,
)


@dataclass(frozen=True)
class BWNeutralDensityScannerResult:
    developed_density: PhysicalDomainArray
    transmittance: PhysicalDomainArray
    scan_linear: PhysicalDomainArray


def build_typed_neutral_density_scanner_chain(
    neutral_density: np.ndarray,
    scanner_profile: SpatialResponseProfile,
) -> BWNeutralDensityScannerResult:
    density = np.asarray(neutral_density)
    if (
        density.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or density.ndim != 2
        or density.size == 0
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise ValueError("neutral density must be a finite nonnegative float field")
    density_rgb = np.ascontiguousarray(np.repeat(density[..., None], 3, axis=-1))
    scale = PhysicalScale(scanner_profile.pixel_pitch_um)
    developed = PhysicalDomainArray.adopt(
        density_rgb,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        ("neutral", "neutral", "neutral"),
        scale,
    )
    transmittance = density_to_transmittance(developed)
    scan_values = np.ascontiguousarray(
        apply_scanner_mtf(transmittance.values, scanner_profile),
        dtype=transmittance.values.dtype,
    )
    scan_linear = PhysicalDomainArray.adopt(
        scan_values,
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("neutral", "neutral", "neutral"),
        scale,
    )
    return BWNeutralDensityScannerResult(developed, transmittance, scan_linear)


def interpret_bw_negative_direct_scan(
    scan_linear: PhysicalDomainArray,
) -> PhysicalDomainArray:
    """Apply a neutral, endpoint-free B&W negative direct-scan polarity."""
    state = scan_linear.require(PhysicalDomain.SCAN_LINEAR)
    values = np.ascontiguousarray(
        state.values.dtype.type(1.0) - state.values,
        dtype=state.values.dtype,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("B&W direct-scan interpretation left display-linear domain")
    return PhysicalDomainArray.adopt(
        values,
        PhysicalDomain.DISPLAY_LINEAR,
        PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
        state.channels,
        state.scale,
    )


__all__ = [
    "BWNeutralDensityScannerResult",
    "build_typed_neutral_density_scanner_chain",
    "interpret_bw_negative_direct_scan",
]
