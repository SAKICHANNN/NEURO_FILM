"""Analytical common-domain envelope for fixed measured-MTF hypotheses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

PLACEMENT_ARMS = (
    "exposure_domain",
    "developed_density_domain",
    "scan_transmittance_domain",
)


@dataclass(frozen=True)
class MinimaxTransmittanceEnvelope:
    lower: np.ndarray
    upper: np.ndarray
    transmittance: np.ndarray
    density: np.ndarray
    uncertainty_half_range: np.ndarray


def build_minimax_transmittance_envelope(
    transmittance_arms: Mapping[str, np.ndarray],
) -> MinimaxTransmittanceEnvelope:
    """Return the scalar Chebyshev center and retained placement uncertainty."""

    if tuple(transmittance_arms) != PLACEMENT_ARMS:
        raise ValueError("measured-MTF envelope requires the three ordered fixed arms")
    arrays = [np.asarray(transmittance_arms[name], dtype=np.float64) for name in PLACEMENT_ARMS]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("measured-MTF transmittance arms must have identical shapes")
    if any(
        not np.all(np.isfinite(array)) or np.any(array <= 0.0) or np.any(array > 1.0)
        for array in arrays
    ):
        raise ValueError("measured-MTF transmittance arms must be finite inside (0, 1]")
    stacked = np.stack(arrays, axis=0)
    lower = np.min(stacked, axis=0)
    upper = np.max(stacked, axis=0)
    transmittance = 0.5 * (lower + upper)
    uncertainty = 0.5 * (upper - lower)
    density = -np.log10(transmittance)
    return MinimaxTransmittanceEnvelope(
        lower=lower,
        upper=upper,
        transmittance=transmittance,
        density=density,
        uncertainty_half_range=uncertainty,
    )


__all__ = [
    "PLACEMENT_ARMS",
    "MinimaxTransmittanceEnvelope",
    "build_minimax_transmittance_envelope",
]
