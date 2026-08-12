"""Caller-held-profile CPU reference for bounded photographic structure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.bounded_histogram_copula import (
    apply_bounded_multipass_histogram_copula,
)
from src.film_physics.bounded_photographic_profile import (
    BoundedPhotographicRuntimeComponents,
    canonical_profile_bytes,
    load_bounded_photographic_profile,
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.spatial_response import apply_scanner_mtf

RECEIPT_SCHEMA = "neuro-film.bounded-photographic-cpu-render-receipt.v1"


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


@dataclass(frozen=True)
class BoundedPhotographicCpuRuntime:
    bundle_sha256: str
    components: BoundedPhotographicRuntimeComponents

    @classmethod
    def load(
        cls, path: Path, *, expected_bundle_sha256: str
    ) -> BoundedPhotographicCpuRuntime:
        payload = load_bounded_photographic_profile(
            path, expected_bundle_sha256=expected_bundle_sha256
        )
        return cls(
            bundle_sha256=payload["bundle_sha256"],
            components=reconstruct_bounded_photographic_profile(payload),
        )

    def render(
        self, source: np.ndarray, *, source_index: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if (
            not isinstance(source, np.ndarray)
            or source.dtype != np.float32
            or source.ndim != 3
            or source.shape[-1] != 3
            or not source.flags.c_contiguous
            or source.size == 0
            or not np.all(np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(source > 1.0)
        ):
            raise ValueError("invalid bounded photographic CPU source")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or source_index >= 2**32
        ):
            raise ValueError("invalid bounded photographic source index")
        components = self.components
        seeds = tuple(
            value + source_index * components.field_seed_stride_per_source
            for value in components.layer_field_seeds
        )
        if any(seed >= 2**64 for seed in seeds):
            raise ValueError("bounded photographic derived seed overflow")
        input_sha = _array_sha256(source)
        physical, diagnostics = apply_bounded_multipass_histogram_copula(
            source,
            profile=components.profile,
            prior=components.prior,
            layer_seeds=seeds,
            correlation_matrix=components.correlation_matrix,
            canonical_receipt_row_block_height=components.canonical_row_block_height,
            rank_bins=components.rank_bins,
        )
        physical_sha = _array_sha256(physical)
        output = np.ascontiguousarray(
            apply_scanner_mtf(
                physical.astype(np.float64), components.scanner_profile
            ),
            dtype=np.float32,
        )
        output_sha = _array_sha256(output)
        receipt_body = {
            "schema": RECEIPT_SCHEMA,
            "bundle_sha256": self.bundle_sha256,
            "density_profile_id": components.profile.identity(),
            "manufacturer_prior_id": components.prior.identity(),
            "source_index": source_index,
            "layer_field_seeds": list(seeds),
            "input_shape": list(source.shape),
            "input_float32_sha256": input_sha,
            "physical_float32_sha256": physical_sha,
            "scanner_float32_sha256": output_sha,
            "rank_bins": components.rank_bins,
            "canonical_row_block_height": components.canonical_row_block_height,
            "diagnostics": diagnostics,
            "claim_level": "generic-physical-inspired-relative-display",
        }
        receipt = {
            **receipt_body,
            "receipt_id": hashlib.sha256(
                canonical_profile_bytes(receipt_body)
            ).hexdigest(),
        }
        return output, receipt


__all__ = ["RECEIPT_SCHEMA", "BoundedPhotographicCpuRuntime"]
