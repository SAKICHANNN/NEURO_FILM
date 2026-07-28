"""Parity helpers for the U6.P6C float32 Standard scanner compiler."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.eval.physical_scanner_profile import _profile
from src.film_physics import (
    apply_scanner_profile,
    apply_scanner_profile_standard_row_tiled,
    compile_scanner_context,
    compile_scanner_standard_context,
)


def deterministic_gradient(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    low = np.float32(0.0001)
    span = np.float32(0.9999)
    output = np.empty((height, width, 3), dtype=np.float32)
    output[..., 0] = low + span * (
        np.float32(0.65) * x + np.float32(0.35) * y
    )
    output[..., 1] = low + span * (
        np.float32(0.25) * x + np.float32(0.75) * y
    )
    output[..., 2] = low + span * (
        np.float32(0.5) * x + np.float32(0.5) * y
    )
    return output


def array_sha256(values: np.ndarray) -> str:
    little = np.asarray(values, dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(little).tobytes()).hexdigest()


def evaluate_standard_parity(
    contract: dict[str, Any], p6a_contract: dict[str, Any]
) -> dict[str, Any]:
    fixture = contract["parity_fixture"]
    shape = tuple(int(value) for value in fixture["shape"])
    rng = np.random.default_rng(int(fixture["seed"]))
    values = rng.uniform(0.0001, 1.0, size=(*shape, 3)).astype(np.float32)
    profile = _profile(
        p6a_contract["profiles"][contract["candidate"]["profile"]]
    )
    reference_context = compile_scanner_context(
        values.astype(np.float64), profile
    )
    reference = apply_scanner_profile(
        values.astype(np.float64),
        profile,
        pixel_pitch_um=1.0,
        context=reference_context,
    )
    standard_context = compile_scanner_standard_context(values, profile)
    rows = []
    outputs = []
    for tile_rows in fixture["row_partitions"]:
        output = apply_scanner_profile_standard_row_tiled(
            values,
            profile,
            pixel_pitch_um=1.0,
            context=standard_context,
            tile_rows=int(tile_rows),
        )
        outputs.append(output)
        rows.append(
            {
                "tile_rows": int(tile_rows),
                "output_sha256": array_sha256(output),
                "maximum_abs_vs_float64": float(
                    np.max(np.abs(output.astype(np.float64) - reference))
                ),
                "finite_bounded": bool(
                    output.dtype == np.float32
                    and np.all(np.isfinite(output))
                    and np.all(output >= 0.0)
                    and np.all(output <= 1.0)
                ),
            }
        )
    return {
        "shape": list(shape),
        "context_id": standard_context.context_id,
        "rows": rows,
        "partitions_exact": all(
            np.array_equal(outputs[0], output) for output in outputs[1:]
        ),
        "maximum_abs_vs_float64": max(
            row["maximum_abs_vs_float64"] for row in rows
        ),
    }
