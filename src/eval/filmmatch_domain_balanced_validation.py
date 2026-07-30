"""Fit-forbidden real-scene validation for the AX5 fixed operator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file
from src.eval.filmmatch_validation_scene import _apply_rows, _save_rgb16_png
from src.roll2film.factorized_monotone_bernstein import (
    FactorizedMonotoneBernsteinOperator,
)


def evaluate_domain_balanced_validation(
    parent_report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    expected_champion = config.get(
        "expected_development_champion", "emissive_share075"
    )
    if parent_report.get("development_champion") != expected_champion:
        raise ValueError("development champion identity drift")
    operator = FactorizedMonotoneBernsteinOperator.from_dict(
        parent_report["final_fit"]["operator"]
    )
    validation = config["validation"]
    source_path = root / validation["source_relative_path"]
    target_path = root / validation["target_relative_path"]
    if (
        sha256_file(source_path) != validation["source_sha256"]
        or sha256_file(target_path) != validation["target_sha256"]
    ):
        raise ValueError("validation image identity drift")
    source = np.asarray(tifffile.memmap(source_path), dtype=np.float64) / 65535.0
    target = np.asarray(tifffile.memmap(target_path), dtype=np.float64) / 65535.0
    if source.shape != target.shape:
        raise ValueError("validation image shape drift")
    output = _apply_rows(
        operator, source, int(config["execution"]["row_chunk"])
    )
    output_sha = {
        "source": _save_rgb16_png(output_dir / "source.png", source),
        "candidate": _save_rgb16_png(output_dir / "candidate.png", output),
        "target_reference": _save_rgb16_png(
            output_dir / "target_reference.png", target
        ),
    }
    diagnostics = {
        "finite": bool(np.all(np.isfinite(output))),
        "minimum_output": float(np.min(output)),
        "maximum_output": float(np.max(output)),
        "new_boundary_fraction": float(
            np.mean(
                np.any(
                    ((output <= 0.0) & (source > 0.0))
                    | ((output >= 1.0) & (source < 1.0)),
                    axis=2,
                )
            )
        ),
        "style_rgb_rmse_from_source": float(
            np.sqrt(np.mean(np.square(output - source)))
        ),
    }
    gates = config["automatic_gate"]
    automatic_pass = bool(
        diagnostics["finite"]
        and diagnostics["minimum_output"] >= 0.0
        and diagnostics["maximum_output"] <= 1.0
        and diagnostics["new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"])
        and diagnostics["style_rgb_rmse_from_source"]
        >= float(gates["minimum_style_rgb_rmse_from_source"])
    )
    report = {
        "schema": config.get(
            "report_schema",
            "neuro_film.u5_r2ax6_filmmatch_domain_balanced_validation.v1",
        ),
        "experiment_id": config["experiment_id"],
        "operator": operator.to_dict(),
        "outputs": output_sha,
        "automatic_diagnostics": diagnostics,
        "automatic_gate_passed": automatic_pass,
        "visual_review_required": automatic_pass,
        "visual_review_opened": automatic_pass,
        "promotion_opened": False,
        "target_pixel_metric_forbidden_reason": (
            "validation captures have materially different framing, crop, "
            "film border and scene geometry"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_domain_balanced_validation"]
