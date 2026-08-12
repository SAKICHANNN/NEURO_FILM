from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from src.film_physics.native_standard_replayable_staging import (
    stage_native_standard_replayable_rows,
)
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime


def test_replayable_rows_commit_restart_verified_png16(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    height, width = 64, 96
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
        report_path=tmp_path / "render.raw.json",
    )
    committed = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "render.png",
        report_path=tmp_path / "render.png.json",
    )
    verified = verify_native_standard_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    assert verified["output_sha256"] == committed["output_sha256"]
    assert (tmp_path / "render.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
