"""Canonical two-pass DC projection for finite-support Thomas fields."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.film_physics.finite_support_thomas import (
    render_finite_support_thomas_region,
)

RECEIPT_SCHEMA = "neuro_film.thomas_dc_receipt.v1"
REDUCER = "float64-neumaier-v1"


class ThomasDcProjectionError(ValueError):
    """Raised when a DC receipt does not match the requested field."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class ThomasDcReceipt:
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
            "truncate": self.truncate,
            "raw_field_sha256": self.raw_field_sha256,
            "sample_count": self.sample_count,
            "raw_sum": self.raw_sum,
            "raw_mean": self.raw_mean,
        }

    def validate_identity(self) -> None:
        expected = hashlib.sha256(_canonical_bytes(self.identity_payload())).hexdigest()
        if self.receipt_id != expected:
            raise ThomasDcProjectionError("Thomas DC receipt identity mismatch")


def _neumaier_add(total: float, compensation: float, value: float) -> tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        compensation += (total - updated) + value
    else:
        compensation += (value - updated) + total
    return updated, compensation


def build_thomas_dc_receipt(
    full_shape: tuple[int, int],
    *,
    profile_id: str,
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float,
    mean_offspring: float,
    component_seeds: tuple[int, int],
    realization_seed: int,
    truncate: float,
    canonical_row_block_height: int,
) -> ThomasDcReceipt:
    """Stream the raw field once in canonical row-major order and bind its DC."""
    if (
        len(full_shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in full_shape)
        or not profile_id
        or not isinstance(canonical_row_block_height, int)
        or canonical_row_block_height <= 0
    ):
        raise ThomasDcProjectionError("invalid Thomas DC receipt request")
    digest = hashlib.sha256()
    total = 0.0
    compensation = 0.0
    count = 0
    for y0 in range(0, full_shape[0], canonical_row_block_height):
        height = min(canonical_row_block_height, full_shape[0] - y0)
        block = render_finite_support_thomas_region(
            full_shape,
            origin_yx=(y0, 0),
            shape=(height, full_shape[1]),
            particle_sigma_pixels=particle_sigma_pixels,
            cluster_sigma_pixels=cluster_sigma_pixels,
            mean_offspring=mean_offspring,
            component_seeds=component_seeds,
            realization_seed=realization_seed,
            truncate=truncate,
        )
        little_endian = np.ascontiguousarray(block, dtype="<f8")
        digest.update(little_endian.tobytes())
        for value in little_endian.reshape(-1):
            total, compensation = _neumaier_add(
                total, compensation, float(value)
            )
            count += 1
    raw_sum = total + compensation
    raw_mean = raw_sum / count
    if not math.isfinite(raw_mean) or count != full_shape[0] * full_shape[1]:
        raise ThomasDcProjectionError("non-finite or incomplete Thomas DC receipt")
    provisional = ThomasDcReceipt(
        schema=RECEIPT_SCHEMA,
        reducer=REDUCER,
        canonical_row_block_height=canonical_row_block_height,
        full_shape=full_shape,
        profile_id=profile_id,
        particle_sigma_pixels=particle_sigma_pixels,
        cluster_sigma_pixels=cluster_sigma_pixels,
        mean_offspring=mean_offspring,
        component_seeds=component_seeds,
        realization_seed=realization_seed,
        truncate=truncate,
        raw_field_sha256=digest.hexdigest(),
        sample_count=count,
        raw_sum=raw_sum,
        raw_mean=raw_mean,
        receipt_id="",
    )
    receipt_id = hashlib.sha256(
        _canonical_bytes(provisional.identity_payload())
    ).hexdigest()
    return replace(provisional, receipt_id=receipt_id)


def render_dc_projected_thomas_region(
    receipt: ThomasDcReceipt,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Render a region and remove only the receipt-bound full-field DC value."""
    receipt.validate_identity()
    if (
        receipt.schema != RECEIPT_SCHEMA
        or receipt.reducer != REDUCER
        or receipt.sample_count != receipt.full_shape[0] * receipt.full_shape[1]
        or not math.isfinite(receipt.raw_mean)
    ):
        raise ThomasDcProjectionError("invalid Thomas DC receipt")
    raw = render_finite_support_thomas_region(
        receipt.full_shape,
        origin_yx=origin_yx,
        shape=shape,
        particle_sigma_pixels=receipt.particle_sigma_pixels,
        cluster_sigma_pixels=receipt.cluster_sigma_pixels,
        mean_offspring=receipt.mean_offspring,
        component_seeds=receipt.component_seeds,
        realization_seed=receipt.realization_seed,
        truncate=receipt.truncate,
    )
    projected = np.ascontiguousarray(raw - receipt.raw_mean, dtype=np.float64)
    if not np.all(np.isfinite(projected)):
        raise ThomasDcProjectionError("projected Thomas field is non-finite")
    projected.setflags(write=False)
    return projected


__all__ = [
    "RECEIPT_SCHEMA",
    "REDUCER",
    "ThomasDcProjectionError",
    "ThomasDcReceipt",
    "build_thomas_dc_receipt",
    "render_dc_projected_thomas_region",
]
