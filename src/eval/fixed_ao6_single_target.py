"""Output-exact single-target AO6 row executor for throughput development."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.fixed_global_policy_confirmation import FixedGlobalPolicyError
from src.film_physics.display_look import (
    build_source_context_display_look_row_stages,
)


def render_fixed_ao6_single_target(
    scene_linear: np.ndarray,
    artifact: Mapping[str, Any],
    component: str,
    *,
    row_chunk: int = 128,
    workers: int = 1,
) -> np.ndarray:
    """Render only the AO6 arm, bounding base rows and preserving exact pixels."""
    source = np.asarray(scene_linear)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or isinstance(row_chunk, bool)
        or not isinstance(row_chunk, int)
        or row_chunk <= 0
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
    ):
        raise FixedGlobalPolicyError("CB69 input drift")
    encoded = np.empty_like(source)
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        encoded[y0:y1] = linear_srgb_to_encoded(
            np.asarray(source[y0:y1], dtype=np.float64)
        ).astype(np.float32)
    payload = artifact["component_payloads"][component]
    apply_base_rows, apply_residual_rows = build_source_context_display_look_row_stages(
        payload, encoded, tile_rows=row_chunk
    )
    output = np.empty_like(source)

    def render_rows(bounds: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        y0, y1 = bounds
        base_rows = apply_base_rows(encoded[y0:y1])
        return y0, y1, apply_residual_rows(base_rows)

    bounds = [
        (y0, min(source.shape[0], y0 + row_chunk))
        for y0 in range(0, source.shape[0], row_chunk)
    ]
    if workers == 1:
        rendered = map(render_rows, bounds)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        rendered = executor.map(render_rows, bounds)
    try:
        for y0, y1, rows in rendered:
            output[y0:y1] = rows
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    if (
        output.shape != source.shape
        or output.dtype != np.float32
        or not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise FixedGlobalPolicyError("CB69 output drift")
    return output


__all__ = ["render_fixed_ao6_single_target"]
