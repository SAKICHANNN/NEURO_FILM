"""Coordinate-stable shared-component Poisson dye-layer structure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .density_conditioned_structure import counter_poisson_rate_field


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


__all__ = [
    "CrossLayerPoissonProfile",
    "cross_layer_counts_to_density",
    "sample_cross_layer_poisson_region",
]
