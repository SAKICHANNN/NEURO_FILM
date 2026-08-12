from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.eval.native_cloud_standard_display import (
    render_cloud_scan_with_standard_display,
)
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4fy_cloud_standard_display_v1.json"
P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def test_cloud_scan_composes_exactly_with_frozen_standard_display(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text())
    fixture = contract["fixture"]
    height, width = fixture["height"], fixture["width"]
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    standard = _runtime(tmp_path)
    p4fb = json.loads(P4FB.read_text())
    p4fb["fixture"]["full_height"] = height
    p4fb["fixture"]["width"] = width
    dll = _build(ROOT, tmp_path / "build", None)
    lib = _configure(dll)
    component_sha = hashlib.sha256(
        (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
    ).hexdigest()

    def provider(forward_rows, y0: int, count: int) -> np.ndarray:
        return render_physical_partition(lib, p4fb, forward_rows, y0, count)

    outputs: list[np.ndarray] = []
    receipts: list[dict] = []
    scans: list[np.ndarray] = []
    for tile_rows in fixture["tile_rows"]:
        cloud = WindowedNativeCloudScanRuntime(
            gaussian_library=dll,
            forward_scatter_profile=_profile(),
            physical_rows=provider,
            physical_component_sha256=component_sha,
            tile_rows=tile_rows,
        )
        scan_parts: list[np.ndarray] = []
        cloud.render_rows_to_sink(
            height=height,
            width=width,
            source_rows=lambda start, count: np.ascontiguousarray(
                source[start : start + count]
            ),
            expected_input_sha256=source_sha,
            output_sink=lambda _y0, _y1, rows: scan_parts.append(rows.copy()),
        )
        scan = np.concatenate(scan_parts)
        scans.append(scan)
        for _ in range(2):
            parts: list[np.ndarray] = []
            receipt = render_cloud_scan_with_standard_display(
                standard,
                cloud,
                height=height,
                width=width,
                source_rows=lambda start, count: np.ascontiguousarray(
                    source[start : start + count]
                ),
                expected_input_sha256=source_sha,
                output_sink=lambda _y0, _y1, rows: parts.append(rows.copy()),
            )
            outputs.append(np.concatenate(parts))
            receipts.append(receipt)

    assert all(np.array_equal(outputs[0], value) for value in outputs[1:])
    assert all(np.array_equal(scans[0], value) for value in scans[1:])
    assert len({item["output_sha256"] for item in receipts}) == 1
    assert all(item["source_passes"] == contract["gates"]["source_passes"] for item in receipts)
    assert all(item["physical_order"] == contract["order"] for item in receipts)
    assert np.all(np.isfinite(outputs[0])) and np.all((outputs[0] >= 0) & (outputs[0] <= 1))

    context = standard._build_source_context(source)
    manual = standard._apply_display(
        np.ascontiguousarray(
            linear_srgb_to_encoded(
                standard._apply_gauge(scans[0]).astype(np.float64)
            ),
            dtype=np.float32,
        ),
        context=context,
    )
    assert np.array_equal(outputs[0], manual)
    assert not np.array_equal(outputs[0], scans[0])
