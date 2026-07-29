"""File ingress for the versioned native Standard display-strength path."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.preprocess import load_working_image

from .native_standard_runtime import NativeStandardRuntime
from .native_standard_staging import (
    _absolute_unresolved,
    _canonical_bytes,
    _sha256_file,
)
from .native_standard_strength_output import (
    commit_verified_native_standard_strength_png16,
)
from .native_standard_strength_staging import (
    stage_native_standard_working_image_with_strength,
)


def render_native_standard_strength_file_to_png16(
    runtime: NativeStandardRuntime,
    *,
    input_path: Path,
    strength: float,
    raw_output_path: Path,
    raw_report_path: Path,
    png_output_path: Path,
    png_report_path: Path,
) -> dict[str, Any]:
    """Decode a supported scene-linear file and commit a strength PNG16."""

    source = _absolute_unresolved(input_path)
    if not source.is_file():
        raise ValueError("native Standard strength input file is missing")
    source_sha = _sha256_file(source)
    working = load_working_image(source)
    if _sha256_file(source) != source_sha:
        raise ValueError(
            "native Standard strength input changed during decode"
        )
    if working.transfer_state != "scene_linear":
        raise ValueError(
            "native Standard strength file ingress requires scene-linear "
            "decode"
        )
    if working.working_space not in {
        "linear_srgb",
        "linear_srgb_d65",
    }:
        raise ValueError(
            "native Standard strength file ingress requires D65 "
            "linear sRGB"
        )
    if not working.orientation_applied:
        raise ValueError(
            "native Standard strength file ingress requires applied "
            "orientation"
        )

    staged = stage_native_standard_working_image_with_strength(
        runtime,
        working,
        strength=strength,
        output_path=raw_output_path,
        report_path=raw_report_path,
    )
    committed = commit_verified_native_standard_strength_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=png_output_path,
        report_path=png_report_path,
    )
    core = {
        "schema": "neuro_film.native_standard_strength_file_result.v1",
        "input_path": str(source),
        "input_file_sha256": source_sha,
        "working_space": working.working_space,
        "transfer_state": working.transfer_state,
        "source_profile_kind": working.source_profile.kind,
        "orientation_applied": working.orientation_applied,
        "strength": staged["strength"],
        "strength_domain": "encoded-display-srgb",
        "raw_run_id": staged["run_id"],
        "raw_report_sha256": staged["report_sha256"],
        "png_delivery_id": committed["delivery_id"],
        "png_report_sha256": committed["report_sha256"],
        "png_output_sha256": committed["output_sha256"],
        "production_default_changed": False,
        "claim_ceiling": (
            "opt-in generic scene-linear file decode through bounded "
            "native Standard display-look strength to staged sRGB16 PNG; "
            "not calibrated, delivered or promoted"
        ),
    }
    return {
        **core,
        "result_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


__all__ = ["render_native_standard_strength_file_to_png16"]
