"""Independent zero-DC density NPS texture for display-scale development."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.structure_compiler import counter_normal_region

_KERNEL = np.asarray([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float64)


def _aperture(values: np.ndarray) -> np.ndarray:
    padded_x = np.pad(values, ((0, 0), (2, 2)), mode="edge")
    horizontal = sum(
        weight * padded_x[:, offset : offset + values.shape[1]]
        for offset, weight in enumerate(_KERNEL)
    )
    padded_y = np.pad(horizontal, ((2, 2), (0, 0)), mode="edge")
    return sum(
        weight * padded_y[offset : offset + values.shape[0], :]
        for offset, weight in enumerate(_KERNEL)
    )


def synthesize_independent_density_nps(
    shape: tuple[int, int], *, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Return one exact-zero-DC, exact-unit-RMS counter-seeded NPS field."""

    if (
        len(shape) != 2
        or any(not isinstance(value, int) or value < 5 for value in shape)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("invalid independent density NPS request")
    white = counter_normal_region(shape, origin_yx=(0, 0), shape=shape, seed=seed)
    field = white - _aperture(white)
    field -= float(np.mean(field, dtype=np.float64))
    standard_deviation = float(np.std(field, dtype=np.float64))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise RuntimeError("independent density NPS field is degenerate")
    field /= standard_deviation
    field = np.ascontiguousarray(field, dtype=np.float64)
    return field, {
        "field_mean": float(np.mean(field, dtype=np.float64)),
        "field_std": float(np.std(field, dtype=np.float64)),
    }


def apply_independent_density_nps(
    neutral_base: np.ndarray, *, density_sigma: float, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a bounded common density NPS field with an analytic cube guard."""

    base = np.asarray(neutral_base, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or not np.isfinite(density_sigma)
        or density_sigma <= 0.0
    ):
        raise ValueError("invalid independent density NPS input")
    field, diagnostics = synthesize_independent_density_nps(base.shape[:2], seed=seed)
    finite_tail = 3.0 * np.tanh(field / 3.0)
    luminance = 0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2]
    visibility = 4.0 * luminance * (1.0 - luminance)
    requested_density = density_sigma * visibility * finite_tail

    scale = np.ones_like(requested_density)
    brightening = requested_density < 0.0
    if np.any(brightening):
        positive_base = base > 0.0
        allowed = np.full_like(base, np.inf)
        allowed[positive_base] = np.log10(
            (1.0 - 16.0 * np.finfo(np.float64).eps) / base[positive_base]
        )
        allowed_brightening = np.min(allowed, axis=-1)
        scale[brightening] = np.minimum(
            1.0,
            allowed_brightening[brightening]
            / np.maximum(-requested_density[brightening], np.finfo(np.float64).tiny),
        )
    scale = np.clip(scale, 0.0, 1.0)
    output64 = base * np.power(10.0, -(requested_density * scale))[..., None]
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("independent density NPS escaped its output domain")
    residual = output.astype(np.float64) - base
    diagnostics.update(
        {
            "finite_tail_mean": float(np.mean(finite_tail, dtype=np.float64)),
            "finite_tail_std": float(np.std(finite_tail, dtype=np.float64)),
            "mean_density_visibility": float(np.mean(visibility, dtype=np.float64)),
            "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "minimum_residual_scale": float(np.min(scale)),
            "limited_fraction": float(np.mean(scale < 1.0)),
            "hard_clipping_used": 0.0,
        }
    )
    return output, diagnostics


def compile_conservative_shared_sigma_d(
    neutral_base: np.ndarray,
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project retained layer Sigma-D onto one conservative achromatic field.

    Relative display luminance is only a normalized coordinate across each
    retained layer curve's observed exposure domain.  It is not interpreted as
    calibrated scene exposure.  A shared density component cannot exceed any
    layer's total Sigma-D, so the layer minimum is the no-fit upper bound.
    """

    base = np.asarray(neutral_base, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or not isinstance(profile, DensityConditionedThomasProfile)
        or not isinstance(prior, ManufacturerCharacteristicPrior)
    ):
        raise ValueError("invalid density-compiled independent NPS input")
    luminance = (
        0.2126 * base[..., 0]
        + 0.7152 * base[..., 1]
        + 0.0722 * base[..., 2]
    )
    layer_sigma: list[np.ndarray] = []
    for index, channel in enumerate(("red", "green", "blue")):
        lower, upper = prior.curves[index].domain
        exposure = lower + luminance * (upper - lower)
        layer_sigma.append(
            profile.amplitude_profile.evaluate_channel(
                prior, channel, exposure
            )
        )
    stacked = np.stack(layer_sigma, axis=-1)
    shared = np.ascontiguousarray(np.min(stacked, axis=-1), dtype=np.float64)
    if not np.all(np.isfinite(shared)) or np.any(shared <= 0.0):
        raise RuntimeError("compiled shared Sigma-D is invalid")
    return shared, {
        "minimum_compiled_sigma_d": float(np.min(shared)),
        "maximum_compiled_sigma_d": float(np.max(shared)),
        "mean_compiled_sigma_d": float(np.mean(shared, dtype=np.float64)),
        "amplitude_multiplier": 1.0,
    }


def apply_density_compiled_independent_nps(
    neutral_base: np.ndarray,
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply P4GW topology with the retained conservative Sigma-D envelope."""

    base = np.asarray(neutral_base, dtype=np.float64)
    sigma_d, amplitude_diagnostics = compile_conservative_shared_sigma_d(
        base, profile=profile, prior=prior
    )
    field, diagnostics = synthesize_independent_density_nps(
        base.shape[:2], seed=seed
    )
    finite_tail = 3.0 * np.tanh(field / 3.0)
    luminance = (
        0.2126 * base[..., 0]
        + 0.7152 * base[..., 1]
        + 0.0722 * base[..., 2]
    )
    visibility = 4.0 * luminance * (1.0 - luminance)
    requested_density = sigma_d * visibility * finite_tail

    scale = np.ones_like(requested_density)
    brightening = requested_density < 0.0
    if np.any(brightening):
        positive_base = base > 0.0
        allowed = np.full_like(base, np.inf)
        allowed[positive_base] = np.log10(
            (1.0 - 16.0 * np.finfo(np.float64).eps) / base[positive_base]
        )
        allowed_brightening = np.min(allowed, axis=-1)
        scale[brightening] = np.minimum(
            1.0,
            allowed_brightening[brightening]
            / np.maximum(
                -requested_density[brightening], np.finfo(np.float64).tiny
            ),
        )
    scale = np.clip(scale, 0.0, 1.0)
    output64 = base * np.power(10.0, -(requested_density * scale))[..., None]
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("density-compiled independent NPS escaped its output domain")
    residual = output.astype(np.float64) - base
    diagnostics.update(amplitude_diagnostics)
    diagnostics.update(
        {
            "finite_tail_mean": float(np.mean(finite_tail, dtype=np.float64)),
            "finite_tail_std": float(np.std(finite_tail, dtype=np.float64)),
            "mean_density_visibility": float(np.mean(visibility, dtype=np.float64)),
            "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "minimum_residual_scale": float(np.min(scale)),
            "limited_fraction": float(np.mean(scale < 1.0)),
            "hard_clipping_used": 0.0,
        }
    )
    return output, diagnostics


def _exact_empirical_midranks(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(len(flat), dtype=np.int64)
    ranks[order] = np.arange(len(flat), dtype=np.int64)
    result = (ranks.astype(np.float64) + 0.5) / float(len(flat))
    return np.ascontiguousarray(result.reshape(values.shape), dtype=np.float64)


def apply_support_matched_gamma_density(
    neutral_base: np.ndarray,
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a moment-matched shared density variable with physical support.

    The shifted Gamma marginal has mean zero, target variance and a lower bound
    equal to the negative available optical density.  Exact-white/support-
    degenerate pixels remain identity because no nonzero zero-mean perturbation
    can satisfy that one-sided support.
    """

    base = np.asarray(neutral_base, dtype=np.float64)
    sigma_d, amplitude_diagnostics = compile_conservative_shared_sigma_d(
        base, profile=profile, prior=prior
    )
    field, field_diagnostics = synthesize_independent_density_nps(
        base.shape[:2], seed=seed
    )
    # Preserve the field ordering/correlation while eliminating marginal
    # sampling error before the inverse Gamma transform.
    uniform = _exact_empirical_midranks(field)
    luminance = (
        0.2126 * base[..., 0]
        + 0.7152 * base[..., 1]
        + 0.0722 * base[..., 2]
    )
    target_sigma = sigma_d * (4.0 * luminance * (1.0 - luminance))
    maximum_channel = np.max(base, axis=-1)
    available_density = np.full(base.shape[:2], np.inf, dtype=np.float64)
    positive = maximum_channel > 0.0
    available_density[positive] = -np.log10(maximum_channel[positive])
    nondegenerate = (
        available_density > np.finfo(np.float64).eps
    ) & (target_sigma > np.finfo(np.float64).tiny)
    delta_density = np.zeros(base.shape[:2], dtype=np.float64)
    if np.any(nondegenerate):
        density = available_density[nondegenerate]
        sigma = target_sigma[nondegenerate]
        shape = np.square(density / sigma)
        scale = np.square(sigma) / density
        quantiles = gamma.ppf(
            uniform[nondegenerate], a=shape, scale=scale
        )
        delta_density[nondegenerate] = quantiles - density
    if (
        not np.all(np.isfinite(delta_density))
        or np.any(delta_density[nondegenerate] < -available_density[nondegenerate])
    ):
        raise RuntimeError("support-matched Gamma density is invalid")
    output64 = base * np.power(10.0, -delta_density)[..., None]
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("support-matched Gamma density escaped the unit cube")
    residual = output.astype(np.float64) - base
    diagnostics = {
        **field_diagnostics,
        **amplitude_diagnostics,
        "minimum_available_density": float(np.min(available_density)),
        "minimum_delta_density": float(np.min(delta_density)),
        "maximum_delta_density": float(np.max(delta_density)),
        "support_degenerate_fraction": float(np.mean(~nondegenerate)),
        "support_degenerate_residual_absolute": float(
            np.max(np.abs(delta_density[~nondegenerate]), initial=0.0)
        ),
        "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "limited_fraction": 0.0,
        "hard_clipping_used": 0.0,
    }
    return output, diagnostics


def apply_layer_support_matched_gamma_density(
    neutral_base: np.ndarray,
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    layer_seeds: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply independent support- and moment-matched RGB layer densities."""

    base = np.asarray(neutral_base, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or not isinstance(profile, DensityConditionedThomasProfile)
        or not isinstance(prior, ManufacturerCharacteristicPrior)
        or len(layer_seeds) != 3
        or any(
            not isinstance(seed, int) or seed < 0 or seed >= 2**64
            for seed in layer_seeds
        )
    ):
        raise ValueError("invalid layer support-matched Gamma request")

    delta_density = np.zeros_like(base, dtype=np.float64)
    target_sigma = np.zeros_like(base, dtype=np.float64)
    degenerate = np.zeros_like(base, dtype=bool)
    field_means: list[float] = []
    field_stds: list[float] = []
    for index, channel in enumerate(("red", "green", "blue")):
        values = base[..., index]
        lower, upper = prior.curves[index].domain
        exposure = lower + values * (upper - lower)
        sigma_d = profile.amplitude_profile.evaluate_channel(
            prior, channel, exposure
        )
        sigma = sigma_d * (4.0 * values * (1.0 - values))
        target_sigma[..., index] = sigma
        available = np.full(values.shape, np.inf, dtype=np.float64)
        positive = values > 0.0
        available[positive] = -np.log10(values[positive])
        active = (available > np.finfo(np.float64).eps) & (
            sigma > np.finfo(np.float64).tiny
        )
        degenerate[..., index] = ~active
        field, field_diagnostics = synthesize_independent_density_nps(
            values.shape, seed=layer_seeds[index]
        )
        field_means.append(field_diagnostics["field_mean"])
        field_stds.append(field_diagnostics["field_std"])
        uniform = _exact_empirical_midranks(field)
        if np.any(active):
            selected_density = available[active]
            selected_sigma = sigma[active]
            shape = np.square(selected_density / selected_sigma)
            scale = np.square(selected_sigma) / selected_density
            quantiles = gamma.ppf(
                uniform[active], a=shape, scale=scale
            )
            delta_density[..., index][active] = quantiles - selected_density
        if np.any(delta_density[..., index][active] < -available[active]):
            raise RuntimeError("layer Gamma density left its physical support")

    output64 = base * np.power(10.0, -delta_density)
    output = np.ascontiguousarray(output64, dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("layer support-matched Gamma escaped the unit cube")
    residual = output.astype(np.float64) - base
    return output, {
        "minimum_target_sigma_d": float(np.min(target_sigma)),
        "maximum_target_sigma_d": float(np.max(target_sigma)),
        "support_degenerate_fraction": float(np.mean(degenerate)),
        "support_degenerate_residual_absolute": float(
            np.max(np.abs(delta_density[degenerate]), initial=0.0)
        ),
        "minimum_developed_density": float(
            np.min(
                np.where(
                    base > 0.0,
                    -np.log10(np.maximum(base, np.finfo(np.float64).tiny))
                    + delta_density,
                    np.inf,
                )
            )
        ),
        "bounded_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "maximum_field_mean_absolute": max(abs(value) for value in field_means),
        "maximum_field_std_absolute_error": max(
            abs(value - 1.0) for value in field_stds
        ),
        "limited_fraction": 0.0,
        "hard_clipping_used": 0.0,
    }


__all__ = [
    "apply_density_compiled_independent_nps",
    "apply_independent_density_nps",
    "apply_layer_support_matched_gamma_density",
    "apply_support_matched_gamma_density",
    "compile_conservative_shared_sigma_d",
    "synthesize_independent_density_nps",
]
