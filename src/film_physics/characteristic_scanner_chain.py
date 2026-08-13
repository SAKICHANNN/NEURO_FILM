"""Typed characteristic-density, structure and compact-scanner composition."""

from __future__ import annotations

import numpy as np

from src.film_physics.bounded_dye_amount_direction import (
    apply_bounded_dye_amount_direction,
)
from src.film_physics.compact_log_scanner_compiler import (
    CompactLogScannerCompiler,
    apply_compact_log_scanner,
    invert_compact_log_scanner,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.relative_display_characteristic_ingress import (
    relative_display_to_finite_density_transmittance,
)


def _normalize_density_rgb(
    density_rgb: np.ndarray, prior: ManufacturerCharacteristicPrior
) -> np.ndarray:
    density = np.asarray(density_rgb, dtype=np.float64)
    if density.ndim < 1 or density.shape[-1] != 3 or not np.all(np.isfinite(density)):
        raise ValueError("characteristic density must be finite RGB")
    normalized = np.empty_like(density)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.density_bounds
        normalized[..., channel] = (density[..., channel] - lower) / (upper - lower)
    tolerance = 4.0 * np.finfo(np.float32).eps
    if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
        raise ValueError("characteristic density is outside the observed bounds")
    normalized = np.where(
        np.abs(normalized) <= tolerance,
        0.0,
        np.where(np.abs(normalized - 1.0) <= tolerance, 1.0, normalized),
    )
    return normalized


def render_characteristic_scanner_positive(
    relative_display_linear: np.ndarray,
    structured_transmittance_rgb: np.ndarray,
    *,
    prior: ManufacturerCharacteristicPrior,
    compiler: CompactLogScannerCompiler,
) -> tuple[np.ndarray, dict[str, float]]:
    """Interpret one structured negative through normalized dye amounts."""

    base_density, base_transmittance, _ = (
        relative_display_to_finite_density_transmittance(relative_display_linear, prior)
    )
    structured = np.asarray(structured_transmittance_rgb)
    if (
        structured.shape != base_transmittance.shape
        or structured.dtype != np.float32
        or not np.all(np.isfinite(structured))
        or np.any(structured <= 0.0)
        or np.any(structured > 1.0)
    ):
        raise ValueError("structured transmittance must be matching positive float32 RGB")
    structured_density = -np.log10(structured.astype(np.float64))
    base_amount = _normalize_density_rgb(base_density, prior)
    candidate_amount = _normalize_density_rgb(structured_density, prior)
    bounded_amount, envelope = apply_bounded_dye_amount_direction(
        base_amount, candidate_amount
    )
    dye_yellow_magenta_cyan = np.ascontiguousarray(
        bounded_amount[..., [2, 1, 0]], dtype=np.float32
    )
    scan = apply_compact_log_scanner(dye_yellow_magenta_cyan, compiler)
    recovered = invert_compact_log_scanner(scan, compiler)
    positive = np.ascontiguousarray(recovered[..., [2, 1, 0]], dtype=np.float32)
    tolerance = np.float32(4e-6)
    if (
        not np.all(np.isfinite(positive))
        or np.any(positive < -tolerance)
        or np.any(positive > 1.0 + tolerance)
    ):
        raise RuntimeError("characteristic scanner positive left relative display domain")
    return positive, {
        **envelope,
        "base_density_minimum": float(np.min(base_density)),
        "base_density_maximum": float(np.max(base_density)),
        "structured_density_minimum": float(np.min(structured_density)),
        "structured_density_maximum": float(np.max(structured_density)),
        "scanner_positive_minimum": float(np.min(positive)),
        "scanner_positive_maximum": float(np.max(positive)),
    }


__all__ = ["render_characteristic_scanner_positive"]
