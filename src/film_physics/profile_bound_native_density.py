"""Compose P4HN uniforms and P4HP density inversion under a profile bundle."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from src.film_physics.bounded_photographic_profile import (
    BoundedPhotographicRuntimeComponents,
)
from src.film_physics.calibrated_native_histogram_copula import (
    apply_source_observable_calibrated_copula,
)
from src.film_physics.native_fast_gamma_density import apply_native_fast_gamma_density
from src.film_physics.native_hybrid_gamma_density import (
    apply_native_hybrid_gamma_density,
)


def apply_profile_bound_native_density(
    copula_library: ctypes.CDLL,
    gamma_library: ctypes.CDLL,
    base: np.ndarray,
    raw_fields: np.ndarray,
    *,
    components: BoundedPhotographicRuntimeComponents,
    copula_iterations: int,
    gamma_inverse_iterations: int,
    high_shape_threshold: float,
    fast_newton_iterations: int | None = None,
    fast_direct_shape_upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image = np.asarray(base, dtype=np.float64)
    fields = np.asarray(raw_fields, dtype=np.float32)
    if (
        image.ndim != 3
        or image.shape[-1] != 3
        or fields.shape != (image.shape[0] * image.shape[1], 3)
        or not np.all(np.isfinite(image))
        or np.any(image < 0.0)
        or np.any(image > 1.0)
    ):
        raise ValueError("invalid profile-bound native density request")
    uniforms, copula_receipt = apply_source_observable_calibrated_copula(
        copula_library,
        fields,
        profile_correlation=components.correlation_matrix,
        rank_bins=components.rank_bins,
        iterations=copula_iterations,
    )
    flat_base = image.reshape(-1, 3)
    density = np.zeros_like(flat_base)
    sigma = np.zeros_like(flat_base)
    for index, channel in enumerate(("red", "green", "blue")):
        values = flat_base[:, index]
        lower, upper = components.prior.curves[index].domain
        exposure = lower + values * (upper - lower)
        sigma_d = components.profile.amplitude_profile.evaluate_channel(
            components.prior, channel, exposure
        )
        sigma[:, index] = sigma_d * (4.0 * values * (1.0 - values))
        positive = values > 0.0
        density[positive, index] = -np.log10(values[positive])
    active = (density > np.finfo(np.float64).eps) & (sigma > np.finfo(np.float64).tiny)
    shape = np.square(density[active] / sigma[active])
    scale = np.square(sigma[active]) / density[active]
    active_uniforms = uniforms.reshape(-1)[active.reshape(-1)]
    if fast_newton_iterations is None and fast_direct_shape_upper is None:
        developed, gamma_receipt = apply_native_hybrid_gamma_density(
            gamma_library,
            active_uniforms,
            shape,
            scale,
            inverse_iterations=gamma_inverse_iterations,
            high_shape_threshold=high_shape_threshold,
        )
    elif fast_newton_iterations is not None and fast_direct_shape_upper is not None:
        developed, gamma_receipt = apply_native_fast_gamma_density(
            gamma_library,
            active_uniforms,
            shape,
            scale,
            direct_iterations=gamma_inverse_iterations,
            newton_iterations=fast_newton_iterations,
            direct_shape_upper=fast_direct_shape_upper,
            newton_shape_upper=high_shape_threshold,
        )
    else:
        raise ValueError("incomplete fast Gamma configuration")
    delta = np.zeros_like(flat_base)
    delta[active] = developed - density[active]
    developed_density = density.copy()
    developed_density[active] = developed
    output = (
        (flat_base * np.power(10.0, -delta)).reshape(image.shape).astype(np.float32)
    )
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("profile-bound native density escaped unit interval")
    return (
        output,
        developed_density.reshape(image.shape),
        {
            "copula": copula_receipt,
            "gamma": gamma_receipt,
            "active_fraction": float(np.count_nonzero(active) / active.size),
            "minimum_shape": float(np.min(shape)),
            "maximum_shape": float(np.max(shape)),
            "minimum_developed_density": float(np.min(developed)),
            "hard_clipping_used": False,
        },
    )


__all__ = ["apply_profile_bound_native_density"]
