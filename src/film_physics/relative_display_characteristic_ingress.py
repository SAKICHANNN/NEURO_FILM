"""Typed generic ingress from relative display code to finite film density."""

from __future__ import annotations

import numpy as np

from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)


def relative_display_to_finite_density_transmittance(
    relative_display_linear: np.ndarray,
    prior: ManufacturerCharacteristicPrior,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Map a bounded proxy coordinate through each observed characteristic curve."""

    values = np.asarray(relative_display_linear, dtype=np.float64)
    if (
        values.ndim < 1
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("relative display ingress must be finite RGB in [0,1]")
    exposure = np.empty_like(values)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        exposure[..., channel] = lower + values[..., channel] * (upper - lower)
    density64 = prior.apply(exposure)
    transmittance64 = np.power(10.0, -density64)
    density = np.ascontiguousarray(density64, dtype=np.float32)
    transmittance = np.ascontiguousarray(transmittance64, dtype=np.float32)
    if (
        not np.all(np.isfinite(density))
        or not np.all(np.isfinite(transmittance))
        or np.any(transmittance <= 0.0)
        or np.any(transmittance > 1.0)
    ):
        raise RuntimeError("finite characteristic ingress escaped its typed domains")
    return (
        density,
        transmittance,
        {
            "input_domain": "relative-display-linear-proxy-coordinate",
            "exposure_domain": "observed-relative-layer-log-exposure",
            "density_domain": prior.output_domain,
            "transmittance_domain": "strictly-positive-relative-film-transmittance",
            "zero_input_transmittance_minimum": float(
                np.min(
                    np.power(
                        10.0, -np.array([c.density_bounds[0] for c in prior.curves])
                    )
                )
            ),
            "one_input_transmittance_minimum": float(
                np.min(
                    np.power(
                        10.0, -np.array([c.density_bounds[1] for c in prior.curves])
                    )
                )
            ),
            "calibrated_exposure_claimed": False,
        },
    )


__all__ = ["relative_display_to_finite_density_transmittance"]
