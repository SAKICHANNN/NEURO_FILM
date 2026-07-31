"""Fit-forbidden real-scene validation for the fixed BL1 operator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file
from src.eval.filmmatch_validation_scene import _apply_rows, _save_rgb16_png
from src.roll2film.identity_residual_sigmoid import (
    IdentityResidualSigmoidOperator,
)


def _operator_from_report(report: Mapping[str, Any]) -> IdentityResidualSigmoidOperator:
    payload = report["final_fit"]["operator"]
    return IdentityResidualSigmoidOperator(
        capture_matrix=np.asarray(payload["capture_matrix"], dtype=np.float64),
        response_midpoints=np.asarray(
            payload["response_midpoints"], dtype=np.float64
        ),
        response_slopes=np.asarray(
            payload["response_slopes"], dtype=np.float64
        ),
        scan_matrix=np.asarray(payload["scan_matrix"], dtype=np.float64),
        nonlinear_strength=float(payload["nonlinear_strength"]),
        exposure_floor=float(payload["exposure_floor"]),
    )


def evaluate_identity_residual_validation(
    parent_report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validation = config["validation"]
    source_path = root / validation["source_relative_path"]
    target_path = root / validation["target_relative_path"]
    if sha256_file(source_path) != validation["source_sha256"]:
        raise ValueError("validation source identity drift")
    if sha256_file(target_path) != validation["target_sha256"]:
        raise ValueError("validation target identity drift")

    source = np.asarray(tifffile.memmap(source_path), dtype=np.float64) / 65535.0
    target = np.asarray(tifffile.memmap(target_path), dtype=np.float64) / 65535.0
    if source.shape != target.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("validation image shape drift")

    operator = _operator_from_report(parent_report)
    candidate = _apply_rows(
        operator, source, int(config["execution"]["row_chunk"])
    )
    boundary = np.any(
        ((candidate <= 0.0) & (source > 0.0))
        | ((candidate >= 1.0) & (source < 1.0)),
        axis=2,
    )
    diagnostics = {
        "finite": bool(np.all(np.isfinite(candidate))),
        "minimum_output": float(np.min(candidate)),
        "maximum_output": float(np.max(candidate)),
        "new_boundary_fraction": float(np.mean(boundary)),
        "style_rgb_rmse_from_source": float(
            np.sqrt(np.mean(np.square(candidate - source)))
        ),
    }
    gates = config["automatic_gate"]
    passed = bool(
        diagnostics["finite"]
        and diagnostics["minimum_output"] >= 0.0
        and diagnostics["maximum_output"] <= 1.0
        and diagnostics["new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"])
        and diagnostics["style_rgb_rmse_from_source"]
        >= float(gates["minimum_style_rgb_rmse_from_source"])
    )
    outputs = {
        "source": _save_rgb16_png(output_dir / "source.png", source),
        "candidate": _save_rgb16_png(output_dir / "candidate.png", candidate),
        "target_reference": _save_rgb16_png(
            output_dir / "target_reference.png", target
        ),
    }
    report = {
        "schema": config["schema"],
        "experiment_id": config["experiment_id"],
        "parent_stable_evidence_id": parent_report["stable_evidence_id"],
        "operator": operator.to_dict(),
        "image_shape": list(source.shape),
        "outputs": outputs,
        "automatic_diagnostics": diagnostics,
        "automatic_gate_passed": passed,
        "visual_review_opened": passed,
        "independent_confirmation_opened": False,
        "product_integration_opened": False,
        "target_pixel_metric_computed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_identity_residual_validation"]
