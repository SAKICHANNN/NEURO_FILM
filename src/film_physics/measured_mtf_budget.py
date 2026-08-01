"""Fail-closed ownership boundary for a measured total-film MTF budget."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.film_physics.measured_mtf import (
    CompiledPsfComponent,
    apply_compiled_positive_psf,
    apply_compiled_positive_psf_row_tiled,
)

LOD_SCHEMA = "neuro_film.measured_positive_psf_lod_bundle.v1"
BUDGET_SCHEMA = "neuro_film.measured_total_film_spatial_budget.v1"
RGB_CHANNELS = ("red", "green", "blue")
REPLACED_STAGES = ("forward_scatter", "development_adjacency", "dye_diffusion")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def compiled_rgb_from_lod_bundle(
    payload: Mapping[str, Any],
) -> tuple[tuple[CompiledPsfComponent, ...], ...]:
    if (
        payload.get("schema") != LOD_SCHEMA
        or payload.get("target_dpi") != 4000.0
        or payload.get("input_reconstruction") != "zero_order_hold_repeat"
        or set(payload.get("channels", {})) != set(RGB_CHANNELS)
    ):
        raise ValueError("unsupported measured-MTF LOD bundle")
    compiled = []
    for name in RGB_CHANNELS:
        rows = payload["channels"][name]
        if not isinstance(rows, list) or not rows:
            raise ValueError("measured-MTF channel is empty")
        compiled.append(
            tuple(
                CompiledPsfComponent(
                    weight=float(row["weight"]),
                    kernel_1d=np.asarray(row["kernel_1d"], dtype=np.float64),
                )
                for row in rows
            )
        )
    return tuple(compiled)


@dataclass(frozen=True)
class MeasuredTotalFilmSpatialBudget:
    source_lod_bundle_id: str
    compiled_rgb: tuple[tuple[CompiledPsfComponent, ...], ...]
    budget_id: str

    @classmethod
    def from_lod_bundle(
        cls, payload: Mapping[str, Any]
    ) -> MeasuredTotalFilmSpatialBudget:
        compiled = compiled_rgb_from_lod_bundle(payload)
        core = {
            "schema": BUDGET_SCHEMA,
            "mode": "measured_total_replacement",
            "source_lod_bundle_id": str(payload["lod_bundle_id"]),
            "rgb_channel_order": list(RGB_CHANNELS),
            "replaces_stages": list(REPLACED_STAGES),
            "retains_downstream_stages": ["scanner_mtf"],
        }
        return cls(
            source_lod_bundle_id=core["source_lod_bundle_id"],
            compiled_rgb=compiled,
            budget_id=hashlib.sha256(_canonical_json(core)).hexdigest(),
        )

    def validate_active_stages(self, active_stages: Sequence[str]) -> None:
        overlap = set(active_stages).intersection(REPLACED_STAGES)
        if overlap:
            raise ValueError(
                "measured total-film MTF cannot compose with replaced stages: "
                + ", ".join(sorted(overlap))
            )

    def apply(self, values: np.ndarray) -> np.ndarray:
        return apply_compiled_positive_psf(values, self.compiled_rgb)

    def apply_row_tiled(self, values: np.ndarray, *, tile_rows: int) -> np.ndarray:
        return apply_compiled_positive_psf_row_tiled(
            values, self.compiled_rgb, tile_rows=tile_rows
        )


__all__ = [
    "BUDGET_SCHEMA",
    "LOD_SCHEMA",
    "REPLACED_STAGES",
    "RGB_CHANNELS",
    "MeasuredTotalFilmSpatialBudget",
    "compiled_rgb_from_lod_bundle",
]
