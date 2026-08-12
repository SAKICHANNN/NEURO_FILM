from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_replayable_staging import (
    stage_native_standard_replayable_rows,
)
from src.film_physics.native_standard_staging import verify_native_standard_staging
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime


def test_replayable_rows_commit_and_restart_verify(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    height, width = 16, 24
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    staged = stage_native_standard_replayable_rows(
        runtime,
        height=height,
        width=width,
        source_rows=lambda start, count: np.ascontiguousarray(
            source[start : start + count]
        ),
        expected_input_sha256=source_sha,
        output_path=tmp_path / "render.f32",
        report_path=tmp_path / "render.json",
    )
    verified = verify_native_standard_staging(
        report_path=Path(staged["report_path"]),
        expected_report_sha256=staged["report_sha256"],
        expected_run_id=staged["run_id"],
    )
    assert verified["output_sha256"] == staged["output_sha256"]


def test_replayable_rows_drift_leaves_no_commit(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    height, width = 16, 24
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    drift = False

    def rows(start: int, count: int) -> np.ndarray:
        result = np.ascontiguousarray(source[start : start + count])
        if drift:
            result[0, 0, 0] = np.nextafter(result[0, 0, 0], np.float32(1.0))
        return result

    output = tmp_path / "render.f32"
    report = tmp_path / "render.json"

    def arm_drift() -> None:
        nonlocal drift
        drift = True

    original = runtime._apply_display

    def display(*args, **kwargs):
        value = original(*args, **kwargs)
        arm_drift()
        return value

    runtime._apply_display = display
    with pytest.raises(RuntimeError, match="source replay drift"):
        stage_native_standard_replayable_rows(
            runtime,
            height=height,
            width=width,
            source_rows=rows,
            expected_input_sha256=source_sha,
            output_path=output,
            report_path=report,
        )
    assert not output.exists()
    assert not report.exists()
    assert list(tmp_path.glob("*.stage")) == []
