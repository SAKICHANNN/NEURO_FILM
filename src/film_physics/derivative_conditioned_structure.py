"""Positive developed-density structure driven by sensitometry derivatives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import gammaincinv, ndtr

from src.roll2film.sensitometry import RGBSensitometryOperator

from .structure_compiler import (
    balanced_correlated_normal_region,
    correlated_normal_region,
    zero_dc_dog_normal_region,
)


@dataclass(frozen=True)
class DerivativeConditionedStructureProfile:
    peak_density_variance: float
    correlation_sigma_pixels: float
    layer_seeds: tuple[int, int, int]
    exposure_minimum: float
    exposure_maximum: float
    normalization_samples: int

    def __post_init__(self) -> None:
        values = (
            self.peak_density_variance,
            self.correlation_sigma_pixels,
            self.exposure_minimum,
            self.exposure_maximum,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.peak_density_variance <= 0.0
            or self.correlation_sigma_pixels < 0.0
            or self.exposure_minimum <= 0.0
            or self.exposure_maximum <= self.exposure_minimum
            or not isinstance(self.normalization_samples, int)
            or self.normalization_samples < 1025
            or len(self.layer_seeds) != 3
            or any(
                not isinstance(seed, int) or seed < 0 or seed >= 2**64
                for seed in self.layer_seeds
            )
        ):
            raise ValueError("invalid derivative-conditioned structure profile")


@dataclass(frozen=True)
class DerivativeConditionedStructureResult:
    density: np.ndarray
    transmittance: np.ndarray

    def __post_init__(self) -> None:
        density = np.asarray(self.density)
        transmittance = np.asarray(self.transmittance)
        if (
            density.dtype != np.float32
            or transmittance.dtype != np.float32
            or density.shape != transmittance.shape
            or density.ndim != 3
            or density.shape[-1] != 3
            or not np.all(np.isfinite(density))
            or not np.all(np.isfinite(transmittance))
            or np.any(density < 0.0)
            or np.any(transmittance <= 0.0)
            or np.any(transmittance > 1.0)
        ):
            raise ValueError("invalid derivative-conditioned structure result")
        density.setflags(write=False)
        transmittance.setflags(write=False)


def gaussian_block_mean_variance_scale(
    correlation_sigma_pixels: float,
    pixel_size_factor: int,
    *,
    truncate: float = 4.0,
) -> float:
    """Return the exact linear-Gaussian variance scale of a square block mean."""

    sigma = float(correlation_sigma_pixels)
    if (
        not math.isfinite(sigma)
        or sigma < 0.0
        or not isinstance(pixel_size_factor, int)
        or pixel_size_factor < 1
        or not math.isfinite(truncate)
        or truncate <= 0.0
    ):
        raise ValueError("invalid Gaussian block-mean variance inputs")
    if sigma == 0.0:
        return 1.0 / float(pixel_size_factor * pixel_size_factor)
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma))
    kernel /= np.sum(kernel, dtype=np.float64)
    autocorrelation = np.correlate(kernel, kernel, mode="full")
    center = kernel.size - 1
    variance = float(autocorrelation[center])
    factor = pixel_size_factor
    one_dimensional = sum(
        (factor - abs(offset))
        * float(autocorrelation[center + offset])
        / variance
        for offset in range(-(factor - 1), factor)
        if 0 <= center + offset < autocorrelation.size
    ) / float(factor * factor)
    scale = one_dimensional * one_dimensional
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise RuntimeError("invalid Gaussian block-mean variance scale")
    return scale


def _validate_exposure(
    linear_exposure: np.ndarray,
    profile: DerivativeConditionedStructureProfile,
) -> np.ndarray:
    values = np.asarray(linear_exposure, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > profile.exposure_maximum)
    ):
        raise ValueError("linear exposure must be finite RGB within profile domain")
    return values


def derivative_variance_shape(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target density and globally normalized H*(dD/dH)^2."""

    values = _validate_exposure(linear_exposure, profile)
    log_exposure = operator.encoder.apply(values)
    encoder_derivative = operator.encoder.derivative(values)
    density = operator.apply(values)
    shape = np.zeros_like(values, dtype=np.float64)
    normalization_exposure = np.geomspace(
        profile.exposure_minimum,
        profile.exposure_maximum,
        profile.normalization_samples,
        dtype=np.float64,
    )
    normalization_log = operator.encoder.apply(normalization_exposure)
    normalization_encoder_derivative = operator.encoder.derivative(
        normalization_exposure
    )
    for channel, curve in enumerate(operator.curves):
        derivative = curve.derivative(log_exposure[..., channel])
        derivative *= encoder_derivative[..., channel]
        raw = values[..., channel] * np.square(derivative)
        reference_derivative = curve.derivative(normalization_log)
        reference_derivative *= normalization_encoder_derivative
        reference = normalization_exposure * np.square(reference_derivative)
        peak = float(np.max(reference))
        if not math.isfinite(peak) or peak <= 0.0:
            raise RuntimeError("invalid derivative variance normalization")
        shape[..., channel] = raw / peak
    if (
        not np.all(np.isfinite(density))
        or not np.all(np.isfinite(shape))
        or np.any(density <= 0.0)
        or np.any(shape < 0.0)
        or np.any(shape > 1.0 + 1e-12)
    ):
        raise RuntimeError("derivative-conditioned target escaped its domain")
    shape = np.minimum(shape, 1.0)
    return density, shape


def gamma_mean_transmittance_density_offset(
    mean_density: np.ndarray,
    density_variance: np.ndarray,
) -> np.ndarray:
    """Return the analytic density offset that preserves mean transmittance.

    For ``X ~ Gamma(m^2/v, v/m)``, this returns ``c`` such that
    ``E[10**(-(X+c))] == 10**(-m)``.  The correction is pointwise and does not
    use rendered samples, so it cannot introduce image-dependent normalization.
    """

    mean = np.asarray(mean_density, dtype=np.float64)
    variance = np.asarray(density_variance, dtype=np.float64)
    if (
        mean.shape != variance.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(variance))
        or np.any(mean <= 0.0)
        or np.any(variance < 0.0)
    ):
        raise ValueError("invalid gamma density moments")
    offset = np.zeros_like(mean)
    active = variance > 0.0
    if np.any(active):
        shape = np.square(mean[active]) / variance[active]
        scale = variance[active] / mean[active]
        log_ten = math.log(10.0)
        offset[active] = mean[active] - shape * np.log1p(log_ten * scale) / log_ten
    if not np.all(np.isfinite(offset)) or np.any(offset < 0.0):
        raise RuntimeError("invalid gamma mean-transmittance correction")
    return offset


def render_derivative_conditioned_structure_region(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DerivativeConditionedStructureResult:
    values = _validate_exposure(linear_exposure, profile)
    full_shape = values.shape[:2]
    y0, x0 = origin_yx
    height, width = shape
    if (
        y0 < 0
        or x0 < 0
        or height <= 0
        or width <= 0
        or y0 + height > full_shape[0]
        or x0 + width > full_shape[1]
    ):
        raise ValueError("requested structure region is outside the full field")
    target, normalized_shape = derivative_variance_shape(values, operator, profile)
    target = target[y0 : y0 + height, x0 : x0 + width]
    desired_variance = (
        profile.peak_density_variance
        * normalized_shape[y0 : y0 + height, x0 : x0 + width]
    )
    output = np.empty_like(target, dtype=np.float64)
    for channel, seed in enumerate(profile.layer_seeds):
        mean = target[..., channel]
        variance = desired_variance[..., channel]
        active = variance > 0.0
        layer = mean.copy()
        if np.any(active):
            normal = correlated_normal_region(
                full_shape,
                origin_yx=origin_yx,
                shape=shape,
                sigma=profile.correlation_sigma_pixels,
                seed=seed,
            )
            uniform = ndtr(normal)
            if np.any(uniform <= 0.0) or np.any(uniform >= 1.0):
                raise RuntimeError("gamma copula escaped the open unit interval")
            gamma_shape = np.square(mean[active]) / variance[active]
            gamma_scale = variance[active] / mean[active]
            layer[active] = gammaincinv(gamma_shape, uniform[active]) * gamma_scale
        output[..., channel] = layer
    density = np.asarray(output, dtype=np.float32)
    transmittance = np.asarray(
        np.power(10.0, -density.astype(np.float64)), dtype=np.float32
    )
    return DerivativeConditionedStructureResult(density, transmittance)


def render_derivative_conditioned_structure(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
) -> DerivativeConditionedStructureResult:
    values = _validate_exposure(linear_exposure, profile)
    return render_derivative_conditioned_structure_region(
        values,
        operator,
        profile,
        origin_yx=(0, 0),
        shape=values.shape[:2],
    )


def render_transmittance_corrected_structure_region(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DerivativeConditionedStructureResult:
    """Render P4AF structure with an analytic mean-transmittance correction."""

    values = _validate_exposure(linear_exposure, profile)
    base = render_derivative_conditioned_structure_region(
        values,
        operator,
        profile,
        origin_yx=origin_yx,
        shape=shape,
    )
    target, normalized_shape = derivative_variance_shape(values, operator, profile)
    y0, x0 = origin_yx
    height, width = shape
    target = target[y0 : y0 + height, x0 : x0 + width]
    variance = profile.peak_density_variance * normalized_shape[
        y0 : y0 + height, x0 : x0 + width
    ]
    offset = gamma_mean_transmittance_density_offset(target, variance)
    density = np.asarray(base.density.astype(np.float64) + offset, dtype=np.float32)
    transmittance = np.asarray(
        np.power(10.0, -density.astype(np.float64)), dtype=np.float32
    )
    return DerivativeConditionedStructureResult(density, transmittance)


def render_transmittance_corrected_structure(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
) -> DerivativeConditionedStructureResult:
    values = _validate_exposure(linear_exposure, profile)
    return render_transmittance_corrected_structure_region(
        values,
        operator,
        profile,
        origin_yx=(0, 0),
        shape=values.shape[:2],
    )


def render_balanced_derivative_structure_region(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> DerivativeConditionedStructureResult:
    """Render the P4AF Gamma marginal from a zero-sum 2x2 Gaussian source."""

    values = _validate_exposure(linear_exposure, profile)
    full_shape = values.shape[:2]
    y0, x0 = origin_yx
    height, width = shape
    if y0 < 0 or x0 < 0 or height <= 0 or width <= 0 or y0 + height > full_shape[0] or x0 + width > full_shape[1]:
        raise ValueError("requested balanced structure region is outside the field")
    target, normalized_shape = derivative_variance_shape(values, operator, profile)
    target = target[y0 : y0 + height, x0 : x0 + width]
    variance = profile.peak_density_variance * normalized_shape[y0 : y0 + height, x0 : x0 + width]
    output = np.empty_like(target)
    for channel, seed in enumerate(profile.layer_seeds):
        mean = target[..., channel]
        active = variance[..., channel] > 0.0
        layer = mean.copy()
        if np.any(active):
            normal = balanced_correlated_normal_region(full_shape, origin_yx=origin_yx, shape=shape, sigma=profile.correlation_sigma_pixels, seed=seed)
            uniform = ndtr(normal)
            if np.any(uniform <= 0.0) or np.any(uniform >= 1.0):
                raise RuntimeError("balanced gamma copula escaped the open unit interval")
            gamma_shape = np.square(mean[active]) / variance[..., channel][active]
            gamma_scale = variance[..., channel][active] / mean[active]
            layer[active] = gammaincinv(gamma_shape, uniform[active]) * gamma_scale
        output[..., channel] = layer
    density = np.asarray(output, dtype=np.float32)
    transmittance = np.asarray(np.power(10.0, -density.astype(np.float64)), dtype=np.float32)
    return DerivativeConditionedStructureResult(density, transmittance)


def render_balanced_derivative_structure(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
) -> DerivativeConditionedStructureResult:
    values = _validate_exposure(linear_exposure, profile)
    return render_balanced_derivative_structure_region(values, operator, profile, origin_yx=(0, 0), shape=values.shape[:2])


def render_zero_dc_dog_derivative_structure_region(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    narrow_sigma: float,
    broad_sigma: float,
    broad_weight: float = 1.0,
) -> DerivativeConditionedStructureResult:
    """Render the P4AF Gamma marginal from a fixed Gaussian band-pass field."""

    values = _validate_exposure(linear_exposure, profile)
    full_shape = values.shape[:2]
    y0, x0 = origin_yx
    height, width = shape
    if (
        y0 < 0
        or x0 < 0
        or height <= 0
        or width <= 0
        or y0 + height > full_shape[0]
        or x0 + width > full_shape[1]
    ):
        raise ValueError("requested band-pass structure region is outside the field")
    target, normalized_shape = derivative_variance_shape(values, operator, profile)
    target = target[y0 : y0 + height, x0 : x0 + width]
    variance = profile.peak_density_variance * normalized_shape[
        y0 : y0 + height, x0 : x0 + width
    ]
    output = np.empty_like(target)
    for channel, seed in enumerate(profile.layer_seeds):
        mean = target[..., channel]
        active = variance[..., channel] > 0.0
        layer = mean.copy()
        if np.any(active):
            normal = zero_dc_dog_normal_region(
                full_shape,
                origin_yx=origin_yx,
                shape=shape,
                narrow_sigma=narrow_sigma,
                broad_sigma=broad_sigma,
                broad_weight=broad_weight,
                seed=seed,
            )
            uniform = ndtr(normal)
            if np.any(uniform <= 0.0) or np.any(uniform >= 1.0):
                raise RuntimeError("band-pass gamma copula escaped the unit interval")
            gamma_shape = np.square(mean[active]) / variance[..., channel][active]
            gamma_scale = variance[..., channel][active] / mean[active]
            layer[active] = gammaincinv(gamma_shape, uniform[active]) * gamma_scale
        output[..., channel] = layer
    density = np.asarray(output, dtype=np.float32)
    transmittance = np.asarray(
        np.power(10.0, -density.astype(np.float64)), dtype=np.float32
    )
    return DerivativeConditionedStructureResult(density, transmittance)


def render_zero_dc_dog_derivative_structure(
    linear_exposure: np.ndarray,
    operator: RGBSensitometryOperator,
    profile: DerivativeConditionedStructureProfile,
    *,
    narrow_sigma: float,
    broad_sigma: float,
    broad_weight: float = 1.0,
) -> DerivativeConditionedStructureResult:
    values = _validate_exposure(linear_exposure, profile)
    return render_zero_dc_dog_derivative_structure_region(
        values,
        operator,
        profile,
        origin_yx=(0, 0),
        shape=values.shape[:2],
        narrow_sigma=narrow_sigma,
        broad_sigma=broad_sigma,
        broad_weight=broad_weight,
    )


__all__ = [
    "DerivativeConditionedStructureProfile",
    "DerivativeConditionedStructureResult",
    "derivative_variance_shape",
    "gamma_mean_transmittance_density_offset",
    "gaussian_block_mean_variance_scale",
    "render_derivative_conditioned_structure",
    "render_derivative_conditioned_structure_region",
    "render_balanced_derivative_structure",
    "render_balanced_derivative_structure_region",
    "render_zero_dc_dog_derivative_structure",
    "render_zero_dc_dog_derivative_structure_region",
    "render_transmittance_corrected_structure",
    "render_transmittance_corrected_structure_region",
]
