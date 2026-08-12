"""Explicit scene-linear NPY ingress to verified native Standard PNG16."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .native_standard_output import commit_verified_native_standard_png16
from .native_standard_replayable_staging import (
    stage_native_standard_replayable_rows,
)
from .native_standard_runtime import NativeStandardRuntime
from .replayable_scene_linear_npy import ReplayableSceneLinearNpyRows


def render_native_standard_scene_linear_npy_to_png16(
    runtime: NativeStandardRuntime,
    *,
    input_path: Path,
    expected_input_file_sha256: str,
    expected_input_pixel_sha256: str,
    raw_output_path: Path,
    raw_report_path: Path,
    png_output_path: Path,
    png_report_path: Path,
) -> dict[str, Any]:
    """Render one exact mmap-backed scene-linear source transaction."""

    source = ReplayableSceneLinearNpyRows(
        input_path,
        expected_file_sha256=expected_input_file_sha256,
        expected_pixel_sha256=expected_input_pixel_sha256,
    )
    try:
        source_receipt = source.verify_file_identity()
        staged = stage_native_standard_replayable_rows(
            runtime,
            height=source.height,
            width=source.width,
            source_rows=source.rows,
            expected_input_sha256=expected_input_pixel_sha256,
            output_path=raw_output_path,
            report_path=raw_report_path,
        )
        source.verify_file_identity()
    finally:
        source.close()
    committed = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=png_output_path,
        report_path=png_report_path,
    )
    core = {
        "schema": "neuro_film.native_standard_replayable_npy_result.v1",
        "source": source_receipt,
        "raw_run_id": staged["run_id"],
        "raw_report_sha256": staged["report_sha256"],
        "png_delivery_id": committed["delivery_id"],
        "png_report_sha256": committed["report_sha256"],
        "png_output_sha256": committed["output_sha256"],
        "production_default_changed": False,
        "claim_ceiling": (
            "opt-in exact scene-linear NPY through native Standard to "
            "staged sRGB16 PNG; not camera decode, calibrated, delivered or promoted"
        ),
    }
    return {
        **core,
        "result_id": hashlib.sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }


__all__ = ["render_native_standard_scene_linear_npy_to_png16"]
