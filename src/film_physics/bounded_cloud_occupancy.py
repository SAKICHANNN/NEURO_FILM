"""Finite-capacity overlapping-cloud occupancy fields.

The source field is a standardized binomial count, not an unbounded Gaussian.
Positive finite-support cloud kernels preserve that bounded innovation before a
canonical whole-frame DC projection.  This is a synthetic scanner-grid model;
the capacity cells are not interpreted as microscopic grains.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .finite_support_thomas import kernel_variance_2d
from .structure_compiler import counter_uniform_region

RECEIPT_SCHEMA = "neuro_film.bounded_cloud_dc_receipt.v1"
REDUCER = "float64-neumaier-v1"


class BoundedCloudOccupancyError(ValueError):
    """Raised when a bounded-cloud request or receipt is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def counter_binomial_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    trials: int,
    probability: float,
    seed: int,
) -> np.ndarray:
    """Return coordinate-stable binomial counts from one open uniform per cell."""
    if (
        not isinstance(trials, int)
        or trials < 1
        or trials > 256
        or not math.isfinite(probability)
        or probability <= 0.0
        or probability >= 1.0
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise BoundedCloudOccupancyError("invalid binomial occupancy request")
    uniform = counter_uniform_region(
        full_shape, origin_yx=origin_yx, shape=shape, seed=seed
    )
    failure_probability = 1.0 - probability
    mass = np.full(shape, failure_probability**trials, dtype=np.float64)
    cumulative = mass.copy()
    counts = np.zeros(shape, dtype=np.uint16)
    active = uniform > cumulative
    outcome = 0
    odds = probability / failure_probability
    while np.any(active):
        outcome += 1
        if outcome > trials:
            raise RuntimeError("binomial occupancy recurrence did not converge")
        counts[active] += np.uint16(1)
        mass *= ((trials - outcome + 1) / outcome) * odds
        cumulative += mass
        active = uniform > cumulative
    counts.setflags(write=False)
    return counts


def _bounded_correlated_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    sigma: float,
    trials: int,
    probability: float,
    seed: int,
    truncate: float,
) -> np.ndarray:
    if (
        not math.isfinite(sigma)
        or sigma <= 0.0
        or not math.isfinite(truncate)
        or truncate <= 0.0
    ):
        raise BoundedCloudOccupancyError("invalid bounded cloud kernel")
    halo = int(truncate * sigma + 0.5)
    y0 = max(0, origin_yx[0] - halo)
    x0 = max(0, origin_yx[1] - halo)
    y1 = min(full_shape[0], origin_yx[0] + shape[0] + halo)
    x1 = min(full_shape[1], origin_yx[1] + shape[1] + halo)
    counts = counter_binomial_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        trials=trials,
        probability=probability,
        seed=seed,
    ).astype(np.float64)
    innovation = (counts - trials * probability) / math.sqrt(
        trials * probability * (1.0 - probability)
    )
    filtered = gaussian_filter(
        innovation,
        sigma=sigma,
        order=0,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    ) / math.sqrt(kernel_variance_2d(sigma, truncate))
    crop_y = origin_yx[0] - y0
    crop_x = origin_yx[1] - x0
    return filtered[crop_y : crop_y + shape[0], crop_x : crop_x + shape[1]]


def render_bounded_cloud_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
    component_seeds: tuple[int, int],
    realization_seed: int,
    trials: int,
    probability: float,
    truncate: float,
) -> np.ndarray:
    """Render a two-scale unit-reference field from bounded occupancy counts."""
    if (
        len(component_seeds) != 2
        or not math.isfinite(cluster_sigma_pixels)
        or cluster_sigma_pixels <= 0.0
        or not math.isfinite(mean_offspring)
        or mean_offspring <= 0.0
        or not isinstance(realization_seed, int)
        or realization_seed < 0
        or realization_seed >= 2**64
    ):
        raise BoundedCloudOccupancyError("invalid bounded cloud profile")
    combined_sigma = math.hypot(particle_sigma_pixels, cluster_sigma_pixels)
    particle_variance = kernel_variance_2d(particle_sigma_pixels, truncate)
    combined_variance = kernel_variance_2d(combined_sigma, truncate)
    first = _bounded_correlated_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        sigma=particle_sigma_pixels,
        trials=trials,
        probability=probability,
        seed=int(component_seeds[0]) ^ realization_seed,
        truncate=truncate,
    )
    second = _bounded_correlated_region(
        full_shape,
        origin_yx=origin_yx,
        shape=shape,
        sigma=combined_sigma,
        trials=trials,
        probability=probability,
        seed=int(component_seeds[1]) ^ realization_seed,
        truncate=truncate,
    )
    normalization = math.sqrt(particle_variance + mean_offspring * combined_variance)
    field = (
        math.sqrt(particle_variance) * first
        + math.sqrt(mean_offspring * combined_variance) * second
    ) / normalization
    output = np.ascontiguousarray(field, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("bounded cloud field is non-finite")
    output.setflags(write=False)
    return output


def _neumaier_add(
    total: float, compensation: float, value: float
) -> tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        compensation += (total - updated) + value
    else:
        compensation += (value - updated) + total
    return updated, compensation


@dataclass(frozen=True)
class BoundedCloudDcReceipt:
    schema: str
    reducer: str
    canonical_row_block_height: int
    full_shape: tuple[int, int]
    profile_id: str
    particle_sigma_pixels: float
    cluster_sigma_pixels: float
    mean_offspring: float
    component_seeds: tuple[int, int]
    realization_seed: int
    trials: int
    probability: float
    truncate: float
    raw_field_sha256: str
    sample_count: int
    raw_sum: float
    raw_mean: float
    receipt_id: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reducer": self.reducer,
            "canonical_row_block_height": self.canonical_row_block_height,
            "full_shape": list(self.full_shape),
            "profile_id": self.profile_id,
            "particle_sigma_pixels": self.particle_sigma_pixels,
            "cluster_sigma_pixels": self.cluster_sigma_pixels,
            "mean_offspring": self.mean_offspring,
            "component_seeds": list(self.component_seeds),
            "realization_seed": self.realization_seed,
            "trials": self.trials,
            "probability": self.probability,
            "truncate": self.truncate,
            "raw_field_sha256": self.raw_field_sha256,
            "sample_count": self.sample_count,
            "raw_sum": self.raw_sum,
            "raw_mean": self.raw_mean,
        }

    def validate_identity(self) -> None:
        expected = hashlib.sha256(_canonical_bytes(self.identity_payload())).hexdigest()
        if self.receipt_id != expected:
            raise BoundedCloudOccupancyError("bounded cloud receipt identity mismatch")


def build_bounded_cloud_dc_receipt(
    full_shape: tuple[int, int],
    *,
    profile_id: str,
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
    component_seeds: tuple[int, int],
    realization_seed: int,
    trials: int,
    probability: float,
    truncate: float,
    canonical_row_block_height: int,
) -> BoundedCloudDcReceipt:
    if (
        len(full_shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in full_shape)
        or not profile_id
        or not isinstance(canonical_row_block_height, int)
        or canonical_row_block_height <= 0
    ):
        raise BoundedCloudOccupancyError("invalid bounded cloud receipt request")
    kwargs = {
        "particle_sigma_pixels": particle_sigma_pixels,
        "cluster_sigma_pixels": cluster_sigma_pixels,
        "mean_offspring": mean_offspring,
        "component_seeds": component_seeds,
        "realization_seed": realization_seed,
        "trials": trials,
        "probability": probability,
        "truncate": truncate,
    }
    digest = hashlib.sha256()
    total = 0.0
    compensation = 0.0
    count = 0
    for y0 in range(0, full_shape[0], canonical_row_block_height):
        height = min(canonical_row_block_height, full_shape[0] - y0)
        block = render_bounded_cloud_region(
            full_shape, origin_yx=(y0, 0), shape=(height, full_shape[1]), **kwargs
        )
        little_endian = np.ascontiguousarray(block, dtype="<f8")
        digest.update(little_endian.tobytes())
        for value in little_endian.reshape(-1):
            total, compensation = _neumaier_add(total, compensation, float(value))
            count += 1
    raw_sum = total + compensation
    raw_mean = raw_sum / count
    provisional = BoundedCloudDcReceipt(
        schema=RECEIPT_SCHEMA,
        reducer=REDUCER,
        canonical_row_block_height=canonical_row_block_height,
        full_shape=full_shape,
        profile_id=profile_id,
        raw_field_sha256=digest.hexdigest(),
        sample_count=count,
        raw_sum=raw_sum,
        raw_mean=raw_mean,
        receipt_id="",
        **kwargs,
    )
    receipt_id = hashlib.sha256(
        _canonical_bytes(provisional.identity_payload())
    ).hexdigest()
    return replace(provisional, receipt_id=receipt_id)


def render_dc_projected_bounded_cloud_region(
    receipt: BoundedCloudDcReceipt,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    receipt.validate_identity()
    if (
        receipt.schema != RECEIPT_SCHEMA
        or receipt.reducer != REDUCER
        or receipt.sample_count != receipt.full_shape[0] * receipt.full_shape[1]
        or not math.isfinite(receipt.raw_mean)
    ):
        raise BoundedCloudOccupancyError("invalid bounded cloud receipt")
    raw = render_bounded_cloud_region(
        receipt.full_shape,
        origin_yx=origin_yx,
        shape=shape,
        particle_sigma_pixels=receipt.particle_sigma_pixels,
        cluster_sigma_pixels=receipt.cluster_sigma_pixels,
        mean_offspring=receipt.mean_offspring,
        component_seeds=receipt.component_seeds,
        realization_seed=receipt.realization_seed,
        trials=receipt.trials,
        probability=receipt.probability,
        truncate=receipt.truncate,
    )
    output = np.ascontiguousarray(raw - receipt.raw_mean, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("projected bounded cloud field is non-finite")
    output.setflags(write=False)
    return output


__all__ = [
    "RECEIPT_SCHEMA",
    "REDUCER",
    "BoundedCloudDcReceipt",
    "BoundedCloudOccupancyError",
    "build_bounded_cloud_dc_receipt",
    "counter_binomial_region",
    "render_bounded_cloud_region",
    "render_dc_projected_bounded_cloud_region",
]
