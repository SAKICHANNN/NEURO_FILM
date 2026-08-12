from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_output import verify_native_standard_png16
from src.film_physics.native_standard_replayable_file import (
    render_native_standard_scene_linear_npy_to_png16,
)
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime


def test_scene_linear_npy_file_reaches_verified_png16(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    pixels = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=64, height=64, width=96), dtype=np.float32
    )
    source = tmp_path / "scene.npy"
    np.save(source, pixels, allow_pickle=False)
    result = render_native_standard_scene_linear_npy_to_png16(
        runtime,
        input_path=source,
        expected_input_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_input_pixel_sha256=hashlib.sha256(
            memoryview(pixels).cast("B")
        ).hexdigest(),
        raw_output_path=tmp_path / "render.f32",
        raw_report_path=tmp_path / "render.raw.json",
        png_output_path=tmp_path / "render.png",
        png_report_path=tmp_path / "render.png.json",
    )
    verified = verify_native_standard_png16(
        report_path=tmp_path / "render.png.json",
        expected_report_sha256=result["png_report_sha256"],
        expected_delivery_id=result["png_delivery_id"],
    )
    assert verified["output_sha256"] == result["png_output_sha256"]
    assert result["source"]["domain"] == "scene-linear-relative-exposure"
