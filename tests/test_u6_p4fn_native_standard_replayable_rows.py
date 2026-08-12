from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.eval.native_standard_replayable_rows import (
    render_native_standard_replayable_rows,
)
from src.film_physics.native_standard_package import (
    resolve_native_standard_libraries,
)
from src.film_physics.native_standard_runtime import NativeStandardRuntime

ROOT = Path(__file__).resolve().parents[1]


def _runtime(_tmp_path: Path) -> NativeStandardRuntime:
    package = json.loads(
        (ROOT / "configs/u6_p8az_native_standard_package_v1.json").read_text()
    )
    report = json.loads(
        (ROOT / "outputs/u6_p8b_artifact_only_cpu_consumer_v1/run_a/report.json").read_text()
    )
    artifact = report["artifact"]
    canonical = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == package["profile_artifact"]["sha256"]
    builds: dict[str, Path] = {}
    for dll in (ROOT / "outputs").rglob("*.dll"):
        digest = hashlib.sha256(dll.read_bytes()).hexdigest()
        for name, component in package["components"].items():
            if digest == component["sha256"]:
                builds.setdefault(name, dll)
    if builds.keys() != package["components"].keys():
        pytest.skip("exact frozen native Standard DLL set is not local")
    resolved = resolve_native_standard_libraries(package, builds)
    return NativeStandardRuntime(package=package, artifact=artifact, resolved=resolved)


def test_standard_replayable_rows_match_array_runtime(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    height, width = 64, 96
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    baseline: list[np.ndarray] = []
    baseline_receipt = runtime.render_to_sink(
        source, output_sink=lambda _y0, _y1, rows: baseline.append(rows.copy())
    )
    candidate: list[np.ndarray] = []
    receipt = render_native_standard_replayable_rows(
        runtime,
        height=height,
        width=width,
        source_rows=lambda start, count: np.ascontiguousarray(source[start : start + count]),
        expected_input_sha256=source_sha,
        output_sink=lambda _y0, _y1, rows: candidate.append(rows.copy()),
    )
    assert np.array_equal(np.concatenate(baseline), np.concatenate(candidate))
    assert receipt["output_sha256"] == baseline_receipt["output"]["array_sha256"]
    assert receipt["source_passes"] == 3
    assert receipt["full_source_frame_retained"] is False


def test_standard_replayable_rows_reject_postrender_source_drift(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    height, width = 16, 24
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    emitted = 0
    drift = False

    def rows(start: int, count: int) -> np.ndarray:
        result = np.ascontiguousarray(source[start : start + count])
        if drift:
            result[0, 0, 0] = np.nextafter(result[0, 0, 0], np.float32(1.0))
        return result

    def sink(_y0: int, _y1: int, _rows: np.ndarray) -> None:
        nonlocal drift, emitted
        emitted += 1
        drift = True

    with pytest.raises(RuntimeError, match="source replay drift"):
        render_native_standard_replayable_rows(
            runtime,
            height=height,
            width=width,
            source_rows=rows,
            expected_input_sha256=source_sha,
            output_sink=sink,
        )
    assert emitted > 0
