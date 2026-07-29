"""Fail-closed file ingress for the opt-in native Standard transaction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.preprocess import load_working_image

from .native_standard_output import (
    commit_verified_native_standard_png16,
)
from .native_standard_runtime import NativeStandardRuntime
from .native_standard_staging import (
    _absolute_unresolved,
    _sha256_file,
    stage_native_standard_working_image,
)


def render_native_standard_file_to_png16(
    runtime: NativeStandardRuntime,
    *,
    input_path: Path,
    raw_output_path: Path,
    raw_report_path: Path,
    png_output_path: Path,
    png_report_path: Path,
) -> dict[str, Any]:
    """Decode one supported scene-linear file and commit verified PNG16."""

    source = _absolute_unresolved(input_path)
    if not source.is_file():
        raise ValueError("native Standard input file is missing")
    source_sha = _sha256_file(source)
    working = load_working_image(source)
    if _sha256_file(source) != source_sha:
        raise ValueError("native Standard input changed during decode")
    if working.transfer_state != "scene_linear":
        raise ValueError(
            "native Standard file ingress requires scene-linear decode"
        )
    if working.working_space not in {
        "linear_srgb",
        "linear_srgb_d65",
    }:
        raise ValueError(
            "native Standard file ingress requires D65 linear sRGB"
        )
    if not working.orientation_applied:
        raise ValueError(
            "native Standard file ingress requires applied orientation"
        )

    staged = stage_native_standard_working_image(
        runtime,
        working,
        output_path=raw_output_path,
        report_path=raw_report_path,
    )
    committed = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=png_output_path,
        report_path=png_report_path,
    )
    core = {
        "schema": "neuro_film.native_standard_file_result.v1",
        "input_path": str(source),
        "input_file_sha256": source_sha,
        "working_space": working.working_space,
        "transfer_state": working.transfer_state,
        "source_profile_kind": working.source_profile.kind,
        "orientation_applied": working.orientation_applied,
        "raw_run_id": staged["run_id"],
        "raw_report_sha256": staged["report_sha256"],
        "png_delivery_id": committed["delivery_id"],
        "png_report_sha256": committed["report_sha256"],
        "png_output_sha256": committed["output_sha256"],
        "production_default_changed": False,
        "claim_ceiling": (
            "opt-in generic scene-linear file decode through native "
            "Standard to staged sRGB16 PNG; not calibrated, delivered "
            "or promoted"
        ),
    }
    return {
        **core,
        "result_id": hashlib.sha256(
            __import__("json").dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }


__all__ = ["render_native_standard_file_to_png16"]
