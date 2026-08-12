from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.film_physics.native_cloud_scan_runtime import NativeCloudScanRuntimeError
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile, _source

ROOT = Path(__file__).resolve().parents[1]
P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def test_replayable_rows_match_array_ingress_and_fail_before_output(tmp_path: Path) -> None:
    contract = json.loads(P4FB.read_text())
    dll = _build(ROOT, tmp_path / "build", None)
    lib = _configure(dll)
    source = _source(
        contract["fixture"]["full_height"], contract["fixture"]["width"]
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    component = hashlib.sha256(
        (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
    ).hexdigest()

    def provider(forward_rows, y0: int, height: int) -> np.ndarray:
        return render_physical_partition(lib, contract, forward_rows, y0, height)

    runtime = WindowedNativeCloudScanRuntime(
        gaussian_library=dll,
        forward_scatter_profile=_profile(),
        physical_rows=provider,
        physical_component_sha256=component,
        tile_rows=11,
    )
    array_rows: list[np.ndarray] = []
    array_receipt = runtime.render_to_sink(
        source,
        output_sink=lambda _y0, _y1, rows: array_rows.append(rows.copy()),
    )
    replay_rows: list[np.ndarray] = []
    calls: list[tuple[int, int]] = []

    def replay(start: int, count: int) -> np.ndarray:
        calls.append((start, count))
        return np.ascontiguousarray(source[start : start + count])

    replay_receipt = runtime.render_rows_to_sink(
        height=source.shape[0],
        width=source.shape[1],
        source_rows=replay,
        expected_input_sha256=source_sha,
        output_sink=lambda _y0, _y1, rows: replay_rows.append(rows.copy()),
    )
    assert np.array_equal(np.concatenate(array_rows), np.concatenate(replay_rows))
    assert replay_receipt["output_sha256"] == array_receipt["output_sha256"]
    assert replay_receipt["full_source_frame_retained"] is False
    assert replay_receipt["source_passes"] == 3
    assert len(calls) > replay_receipt["submitted_tiles"]

    output_calls = 0

    def count_output(_y0: int, _y1: int, _rows: np.ndarray) -> None:
        nonlocal output_calls
        output_calls += 1

    with pytest.raises(NativeCloudScanRuntimeError, match="source hash drift"):
        runtime.render_rows_to_sink(
            height=source.shape[0],
            width=source.shape[1],
            source_rows=replay,
            expected_input_sha256="0" * 64,
            output_sink=count_output,
        )
    assert output_calls == 0

    calls = 0

    def drifting_replay(start: int, count: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        rows = np.ascontiguousarray(source[start : start + count])
        if calls > 16:
            rows[0, 0, 0] = np.nextafter(rows[0, 0, 0], np.float32(1.0))
        return rows

    with pytest.raises(NativeCloudScanRuntimeError, match="source replay drift"):
        runtime.render_rows_to_sink(
            height=source.shape[0],
            width=source.shape[1],
            source_rows=drifting_replay,
            expected_input_sha256=source_sha,
            output_sink=lambda _y0, _y1, _rows: None,
        )
