"""Compose correct-domain cloud scan rows with frozen Standard display stages."""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Callable

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.native_ao6_base_profile import NativeAo6BaseContextF32V1
from src.film_physics.native_ao6_context_profile import NativeAo6ContextStateF32V1
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from src.film_physics.native_standard_runtime import (
    NativeStandardRuntime,
    NativeStandardRuntimeError,
    OutputSink,
    _pointer,
)

SourceRows = Callable[[int, int], np.ndarray]


def render_cloud_scan_with_standard_display(
    standard: NativeStandardRuntime,
    cloud: WindowedNativeCloudScanRuntime,
    *,
    height: int,
    width: int,
    source_rows: SourceRows,
    expected_input_sha256: str,
    output_sink: OutputSink,
) -> dict:
    """Render cloud physics, then gauge/OETF/AO6 without changing domains."""

    if (
        height <= 0
        or width <= 0
        or not callable(source_rows)
        or not callable(output_sink)
        or len(expected_input_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_input_sha256)
    ):
        raise ValueError("invalid cloud Standard display source")

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
            raise NativeStandardRuntimeError("cloud Standard source row drift")
        return rows

    # AO6 context is source-owned and must be frozen before physical output.
    state = NativeAo6ContextStateF32V1()
    if standard._context.nf_ao6_context_f32_init_v1(ctypes.byref(state)) != 0:
        raise NativeStandardRuntimeError("AO6 context init failed")
    context_digest = hashlib.sha256()
    context_tile_rows = int(standard.policy["tile_rows"])
    for y0 in range(0, height, context_tile_rows):
        y1 = min(height, y0 + context_tile_rows)
        source = read_rows(y0, y1 - y0)
        context_digest.update(memoryview(source).cast("B"))
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(source.astype(np.float64)), dtype=np.float32
        )
        scratch = np.empty_like(encoded)
        status = standard._context.nf_ao6_context_f32_update_v2(
            ctypes.byref(standard._base_profile),
            ctypes.byref(state),
            _pointer(encoded),
            encoded.shape[0] * encoded.shape[1],
            _pointer(scratch),
        )
        if status != 0:
            raise NativeStandardRuntimeError(f"AO6 context update failed: {status}")
    if context_digest.hexdigest() != expected_input_sha256:
        raise NativeStandardRuntimeError("cloud Standard context source hash drift")
    context = NativeAo6BaseContextF32V1()
    if standard._context.nf_ao6_context_f32_finalize_v1(
        ctypes.byref(state), ctypes.byref(context)
    ) != 0:
        raise NativeStandardRuntimeError("AO6 context finalize failed")

    output_digest = hashlib.sha256()
    consumed = 0

    def consume_scan(y0: int, y1: int, scan_rows: np.ndarray) -> None:
        nonlocal consumed
        if y0 != consumed or scan_rows.shape != (y1 - y0, width, 3):
            raise NativeStandardRuntimeError("cloud Standard ordered output drift")
        gauged = standard._apply_gauge(scan_rows)
        encoded = np.ascontiguousarray(
            linear_srgb_to_encoded(gauged.astype(np.float64)), dtype=np.float32
        )
        displayed = np.ascontiguousarray(
            standard._apply_display(encoded, context=context), dtype=np.float32
        )
        if not np.all(np.isfinite(displayed)) or np.any(displayed < 0) or np.any(displayed > 1):
            raise NativeStandardRuntimeError("cloud Standard display output invalid")
        displayed.flags.writeable = False
        output_sink(y0, y1, displayed)
        output_digest.update(memoryview(displayed).cast("B"))
        consumed = y1

    cloud_receipt = cloud.render_rows_to_sink(
        height=height,
        width=width,
        source_rows=read_rows,
        expected_input_sha256=expected_input_sha256,
        output_sink=consume_scan,
    )
    if consumed != height:
        raise NativeStandardRuntimeError("cloud Standard display output incomplete")
    order = [
        "scene-linear-relative-exposure",
        "forward-scatter",
        "sensitometry",
        "developed-dye-cloud-density",
        "bounded-development-adjacency",
        "dye-diffusion",
        "density-interpretation",
        "scanner-mtf",
        "scan-linear",
        "neutral-gauge",
        "srgb-oetf",
        "ao6-display",
    ]
    return {
        "input_sha256": expected_input_sha256,
        "output_sha256": output_digest.hexdigest(),
        "shape": [height, width, 3],
        "source_passes": int(cloud_receipt["source_passes"]) + 1,
        "full_source_frame_retained": False,
        "physical_order": order,
        "cloud_receipt": cloud_receipt,
        "standard_package_sha256": standard.package_sha256,
        "standard_artifact_sha256": standard.artifact_sha256,
        "claim_ceiling": (
            "synthetic-correct-domain-cloud-plus-frozen-ao6-composition-"
            "not-calibrated-not-promoted"
        ),
        "production_default_changed": False,
    }


__all__ = ["render_cloud_scan_with_standard_display"]
