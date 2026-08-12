"""Research adapter for replayable-row ingress to native Standard."""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Callable

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.native_ao6_base_profile import NativeAo6BaseContextF32V1
from src.film_physics.native_ao6_context_profile import NativeAo6ContextStateF32V1
from src.film_physics.native_standard_runtime import (
    NativeStandardRuntime,
    NativeStandardRuntimeError,
    OutputSink,
    _pointer,
)

SourceRows = Callable[[int, int], np.ndarray]


def render_native_standard_replayable_rows(
    runtime: NativeStandardRuntime,
    *,
    height: int,
    width: int,
    source_rows: SourceRows,
    expected_input_sha256: str,
    output_sink: OutputSink,
) -> dict:
    if (
        height <= 0
        or width <= 0
        or not callable(source_rows)
        or not callable(output_sink)
        or len(expected_input_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_input_sha256)
    ):
        raise ValueError("invalid native Standard replayable source")
    tile_rows = int(runtime.policy["tile_rows"])

    def read_rows(start: int, count: int) -> np.ndarray:
        rows = np.asarray(source_rows(start, count))
        if (
            rows.dtype != np.float32
            or rows.shape != (count, width, 3)
            or not rows.flags.c_contiguous
            or not np.all(np.isfinite(rows))
            or np.any(rows < 0)
            or np.any(rows > 1)
        ):
            raise NativeStandardRuntimeError("native Standard source row drift")
        return rows

    state = NativeAo6ContextStateF32V1()
    if runtime._context.nf_ao6_context_f32_init_v1(ctypes.byref(state)) != 0:
        raise NativeStandardRuntimeError("AO6 context init failed")
    input_digest = hashlib.sha256()
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        rows = read_rows(y0, y1 - y0)
        input_digest.update(memoryview(rows).cast("B"))
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(rows.astype(np.float64)), dtype=np.float32
        )
        scratch = np.empty_like(encoded)
        status = runtime._context.nf_ao6_context_f32_update_v2(
            ctypes.byref(runtime._base_profile), ctypes.byref(state),
            _pointer(encoded), encoded.shape[0] * encoded.shape[1],
            _pointer(scratch),
        )
        if status != 0:
            raise NativeStandardRuntimeError(f"AO6 context update failed: {status}")
    if input_digest.hexdigest() != expected_input_sha256:
        raise NativeStandardRuntimeError("native Standard source hash drift")
    context = NativeAo6BaseContextF32V1()
    if runtime._context.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise NativeStandardRuntimeError("AO6 context finalize failed")

    output_digest = hashlib.sha256()
    consumed = 0
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - runtime._halo)
        source_y1 = min(height, y1 + runtime._halo)
        rendered = runtime._render_physical_tile(
            read_rows(source_y0, source_y1 - source_y0)
        )
        core = np.ascontiguousarray(
            rendered[y0 - source_y0 : y1 - source_y0]
        )
        gauged = runtime._apply_gauge(core)
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(gauged.astype(np.float64)), dtype=np.float32
        )
        rows = runtime._apply_display(encoded, context=context)
        rows.flags.writeable = False
        output_sink(y0, y1, rows)
        output_digest.update(rows.tobytes())
        consumed = y1
    if consumed != height:
        raise NativeStandardRuntimeError("native Standard output incomplete")
    replay_digest = hashlib.sha256()
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        replay_digest.update(memoryview(read_rows(y0, y1 - y0)).cast("B"))
    if replay_digest.hexdigest() != expected_input_sha256:
        raise NativeStandardRuntimeError("native Standard source replay drift")
    return {
        "input_sha256": expected_input_sha256,
        "output_sha256": output_digest.hexdigest(),
        "shape": [height, width, 3],
        "source_passes": 3,
        "full_source_frame_retained": False,
        "package_sha256": runtime.package_sha256,
        "artifact_sha256": runtime.artifact_sha256,
        "production_default_changed": False,
    }


__all__ = ["render_native_standard_replayable_rows"]
