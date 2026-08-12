"""Coordinate-stable shared-component Poisson dye-layer structure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .density_conditioned_structure import counter_poisson_rate_field
from .developed_structure import DevelopedStructureContext


@dataclass(frozen=True)
class CrossLayerPoissonProfile:
    """Seven-component positive multivariate-Poisson layer profile."""

    marginal_rates_cmy: tuple[float, float, float]
    shared_all_rate: float
    shared_pair_rates_cm_cy_my: tuple[float, float, float]
    mark_optical_density_cmy: tuple[float, float, float]
    seed: int
    component_seed_stride: int = 1009

    def __post_init__(self) -> None:
        values = (
            *self.marginal_rates_cmy,
            self.shared_all_rate,
            *self.shared_pair_rates_cm_cy_my,
            *self.mark_optical_density_cmy,
        )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in values)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
            or not isinstance(self.component_seed_stride, int)
            or self.component_seed_stride <= 0
        ):
            raise ValueError("invalid cross-layer Poisson profile")
        if any(value <= 0.0 for value in self.independent_rates_cmy):
            raise ValueError("shared rates exceed a marginal rate")

    @property
    def independent_rates_cmy(self) -> tuple[float, float, float]:
        all_rate = self.shared_all_rate
        cm, cy, my = self.shared_pair_rates_cm_cy_my
        c, m, y = self.marginal_rates_cmy
        return (c - all_rate - cm - cy, m - all_rate - cm - my, y - all_rate - cy - my)

    def analytic_correlation(self) -> np.ndarray:
        all_rate = self.shared_all_rate
        cm, cy, my = self.shared_pair_rates_cm_cy_my
        rates = np.asarray(self.marginal_rates_cmy, dtype=np.float64)
        covariance = np.asarray(
            [
                [rates[0], all_rate + cm, all_rate + cy],
                [all_rate + cm, rates[1], all_rate + my],
                [all_rate + cy, all_rate + my, rates[2]],
            ],
            dtype=np.float64,
        )
        scale = np.sqrt(rates[:, None] * rates[None, :])
        result = covariance / scale
        np.fill_diagonal(result, 1.0)
        result.setflags(write=False)
        return result


def sample_cross_layer_poisson_region(
    profile: CrossLayerPoissonProfile,
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return uint16 C/M/Y counts with shared and independent components."""

    all_rate = profile.shared_all_rate
    cm, cy, my = profile.shared_pair_rates_cm_cy_my
    ic, im, iy = profile.independent_rates_cmy
    rates = (all_rate, cm, cy, my, ic, im, iy)
    components = []
    for index, rate in enumerate(rates):
        field = np.full(shape, rate, dtype=np.float64)
        seed = (profile.seed + index * profile.component_seed_stride) % (2**64)
        components.append(
            counter_poisson_rate_field(
                field, full_shape, origin_yx=origin_yx, seed=seed, maximum_rate=rate
            ).astype(np.uint32)
        )
    (
        shared_all,
        pair_cm,
        pair_cy,
        pair_my,
        independent_c,
        independent_m,
        independent_y,
    ) = components
    counts = np.stack(
        (
            shared_all + pair_cm + pair_cy + independent_c,
            shared_all + pair_cm + pair_my + independent_m,
            shared_all + pair_cy + pair_my + independent_y,
        ),
        axis=-1,
    )
    if np.any(counts > np.iinfo(np.uint16).max):
        raise RuntimeError("cross-layer Poisson count overflow")
    output = counts.astype(np.uint16)
    output.setflags(write=False)
    return output


def cross_layer_counts_to_density(
    profile: CrossLayerPoissonProfile, counts: np.ndarray
) -> np.ndarray:
    values = np.asarray(counts)
    if values.dtype != np.uint16 or values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("cross-layer counts must be uint16 HxWx3")
    density = values.astype(np.float32) * np.asarray(
        profile.mark_optical_density_cmy, dtype=np.float32
    )
    density.setflags(write=False)
    return density


@dataclass(frozen=True)
class CrossLayerCloudGeometry:
    """Developed context plus exact component centers for audit."""

    context: DevelopedStructureContext
    component_centers: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.component_centers) != 7:
            raise ValueError("cross-layer geometry requires seven components")
        owned = []
        for centers in self.component_centers:
            values = np.asarray(centers, dtype=np.float64)
            if (
                values.ndim != 2
                or values.shape[1:] != (2,)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError("invalid cross-layer component centers")
            copied = np.array(values, copy=True)
            copied.setflags(write=False)
            owned.append(copied)
        object.__setattr__(self, "component_centers", tuple(owned))


def _centers_from_counts(counts: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = int(np.sum(counts, dtype=np.int64))
    centers = np.empty((total, 2), dtype=np.float64)
    cursor = 0
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = int(counts[row, column])
            if count:
                centers[cursor : cursor + count, 0] = row + rng.random(count)
                centers[cursor : cursor + count, 1] = column + rng.random(count)
                cursor += count
    return centers


def build_cross_layer_cloud_geometry(
    profile: CrossLayerPoissonProfile,
    input_shape: tuple[int, int],
    *,
    radius_um_cmy: tuple[float, float, float],
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
) -> CrossLayerCloudGeometry:
    """Materialize shared Poisson components as exact shared cloud centers."""

    if any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError("input shape must be positive")
    all_rate = profile.shared_all_rate
    cm, cy, my = profile.shared_pair_rates_cm_cy_my
    ic, im, iy = profile.independent_rates_cmy
    rates = (all_rate, cm, cy, my, ic, im, iy)
    components = []
    for index, rate in enumerate(rates):
        count_seed = (profile.seed + index * profile.component_seed_stride) % (2**64)
        counts = counter_poisson_rate_field(
            np.full(input_shape, rate, dtype=np.float64),
            input_shape,
            origin_yx=(0, 0),
            seed=count_seed,
            maximum_rate=rate,
        )
        coordinate_seed = (count_seed ^ 0xD2B74407B1CE6E93) % (2**64)
        components.append(_centers_from_counts(counts, coordinate_seed))
    (
        shared_all,
        pair_cm,
        pair_cy,
        pair_my,
        independent_c,
        independent_m,
        independent_y,
    ) = components
    layers = (
        np.concatenate((shared_all, pair_cm, pair_cy, independent_c)),
        np.concatenate((shared_all, pair_cm, pair_my, independent_m)),
        np.concatenate((shared_all, pair_cy, pair_my, independent_y)),
    )
    if (
        not isinstance(output_zoom, int)
        or output_zoom <= 0
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
        or not math.isfinite(output_pixel_pitch_um)
        or output_pixel_pitch_um <= 0.0
        or any(not math.isfinite(value) or value <= 0.0 for value in radius_um_cmy)
    ):
        raise ValueError("invalid cross-layer cloud render geometry")
    input_pitch_um = output_zoom * output_pixel_pitch_um
    radii_input = tuple(float(value) / input_pitch_um for value in radius_um_cmy)
    offset_rng = np.random.default_rng((profile.seed ^ 0xA24BAED4963EE407) % (2**64))
    offsets = offset_rng.uniform(-0.5, 0.5, size=(monte_carlo_samples, 2))
    context = DevelopedStructureContext(
        "colour-dye-cloud",
        input_shape,
        output_zoom,
        output_pixel_pitch_um,
        layers,
        radii_input,
        profile.mark_optical_density_cmy,
        offsets,
        profile.seed,
    )
    return CrossLayerCloudGeometry(context, tuple(components))


def build_conditioned_total_cloud_geometry(
    profile: CrossLayerPoissonProfile,
    input_shape: tuple[int, int],
    *,
    radius_um_cmy: tuple[float, float, float],
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
) -> CrossLayerCloudGeometry:
    """Materialize a finite-window binomial process with exact totals."""

    if any(not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError("input shape must be positive")
    area = input_shape[0] * input_shape[1]
    all_rate = profile.shared_all_rate
    cm, cy, my = profile.shared_pair_rates_cm_cy_my
    ic, im, iy = profile.independent_rates_cmy
    rates = (all_rate, cm, cy, my, ic, im, iy)
    components = []
    for index, rate in enumerate(rates):
        count = round(rate * area)
        if not math.isclose(count, rate * area, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("conditioned total is not integral")
        seed = (profile.seed + index * profile.component_seed_stride) % (2**64)
        rng = np.random.default_rng(seed ^ 0xD2B74407B1CE6E93)
        centers = np.empty((count, 2), dtype=np.float64)
        centers[:, 0] = rng.random(count) * input_shape[0]
        centers[:, 1] = rng.random(count) * input_shape[1]
        components.append(centers)
    (
        shared_all,
        pair_cm,
        pair_cy,
        pair_my,
        independent_c,
        independent_m,
        independent_y,
    ) = components
    layers = (
        np.concatenate((shared_all, pair_cm, pair_cy, independent_c)),
        np.concatenate((shared_all, pair_cm, pair_my, independent_m)),
        np.concatenate((shared_all, pair_cy, pair_my, independent_y)),
    )
    if (
        not isinstance(output_zoom, int)
        or output_zoom <= 0
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
        or not math.isfinite(output_pixel_pitch_um)
        or output_pixel_pitch_um <= 0.0
        or any(not math.isfinite(value) or value <= 0.0 for value in radius_um_cmy)
    ):
        raise ValueError("invalid conditioned-total render geometry")
    input_pitch_um = output_zoom * output_pixel_pitch_um
    radii_input = tuple(float(value) / input_pitch_um for value in radius_um_cmy)
    offset_rng = np.random.default_rng((profile.seed ^ 0xA24BAED4963EE407) % (2**64))
    offsets = offset_rng.uniform(-0.5, 0.5, size=(monte_carlo_samples, 2))
    context = DevelopedStructureContext(
        "colour-dye-cloud",
        input_shape,
        output_zoom,
        output_pixel_pitch_um,
        layers,
        radii_input,
        profile.mark_optical_density_cmy,
        offsets,
        profile.seed,
    )
    return CrossLayerCloudGeometry(context, tuple(components))


__all__ = [
    "CrossLayerCloudGeometry",
    "CrossLayerPoissonProfile",
    "build_conditioned_total_cloud_geometry",
    "build_cross_layer_cloud_geometry",
    "cross_layer_counts_to_density",
    "sample_cross_layer_poisson_region",
]
