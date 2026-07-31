"""Evaluate analytical AO6-base composition of the fixed BL1 residual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.eval.fresh_native_standard_confirmation import boundary_metrics
from src.eval.global_frontier import sha256_file
from src.preprocess import save_srgb16_png
from src.roll2film.analytical_residual_composition import (
    compose_analytical_residual,
)


SCHEMA = "neuro_film.u5_r2bl4_filmmatch_analytical_residual_ood_report.v1"
BASE_ARM = "fixed_ao6_colour_only_t15_c35"
TARGET_ARM = "fixed_bl1_identity_residual_sigmoid"


class AnalyticalResidualOODError(RuntimeError):
    """Raised when BL4 identities or image contracts drift."""


def _load_exact_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise AnalyticalResidualOODError(f"hash mismatch: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalyticalResidualOODError(f"expected JSON object: {path}")
    return payload


def _decode_rgb16(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise AnalyticalResidualOODError("parent output identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape[-1] != 3:
        raise AnalyticalResidualOODError("expected RGB16 PNG parent output")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float64) / 65535.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def run_analytical_residual_ood(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    parent_output_dir: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("BL4 output is create-only")
    parent = config["parent"]
    decision = _load_exact_json(
        root, parent["bl3_decision"], parent["bl3_decision_sha256"]
    )
    report_a = _load_exact_json(root, parent["run_a"], parent["report_sha256"])
    report_b = _load_exact_json(root, parent["run_b"], parent["report_sha256"])
    if (
        config.get("schema")
        != "neuro_film.u5_r2bl4_filmmatch_analytical_residual_ood.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or decision["decision"]["status"] != parent["required_status"]
        or report_a != report_b
        or report_a["stable_evidence_id"] != parent["stable_evidence_id"]
        or report_a["automatic_gate_pass"]
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config["candidate"]["hard_clipping_allowed"]
    ):
        raise AnalyticalResidualOODError("BL4 frozen contract drift")

    parent_rows = {(row["source_id"], row["arm_id"]): row for row in report_a["rows"]}
    source_ids = sorted({key[0] for key in parent_rows})
    candidate_config = config["candidate"]
    rows: list[dict[str, Any]] = []
    limited_fractions: list[float] = []
    retentions: list[float] = []
    styles: list[float] = []
    output_dir.mkdir(parents=True)
    for source_id in source_ids:
        base_row = parent_rows[(source_id, BASE_ARM)]
        target_row = parent_rows[(source_id, TARGET_ARM)]
        base = _decode_rgb16(
            parent_output_dir / base_row["output"], base_row["output_sha256"]
        )
        target = _decode_rgb16(
            parent_output_dir / target_row["output"], target_row["output_sha256"]
        )
        composed = compose_analytical_residual(
            base,
            target,
            hard_boundary_epsilon=float(
                candidate_config["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon=float(
                candidate_config["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        requested = target - base
        actual = composed.output - base
        requested_norm = float(np.linalg.norm(requested.reshape(-1)))
        retention = (
            float(np.linalg.norm(actual.reshape(-1))) / requested_norm
            if requested_norm > 0.0
            else 1.0
        )
        limited = float(np.mean(composed.residual_scale < 1.0 - 1e-12))
        style = float(np.sqrt(np.mean(np.square(actual))))
        path = output_dir / "renders" / f"{source_id}.png"
        save_srgb16_png(composed.output.astype(np.float32), path)
        metrics = boundary_metrics(composed.output, base)
        rows.append(
            {
                "source_id": source_id,
                "make": base_row["make"],
                "output": path.relative_to(output_dir).as_posix(),
                "output_sha256": sha256_file(path),
                "residual_l2_retention": retention,
                "safety_limited_pixel_fraction": limited,
                "style_rgb_rmse_from_ao6": style,
                "minimum_residual_scale": float(np.min(composed.residual_scale)),
                **metrics,
            }
        )
        limited_fractions.append(limited)
        retentions.append(retention)
        styles.append(style)

    gate = config["automatic_gate"]
    aggregate = {
        "output_count": len(rows),
        "maximum_output_code_boundary_fraction": max(
            row["output_code_boundary_fraction"] for row in rows
        ),
        "maximum_new_boundary_fraction_vs_ao6": max(
            row["new_boundary_fraction_vs_ao6"] for row in rows
        ),
        "median_residual_energy_retention": float(np.median(retentions)),
        "p95_safety_limited_pixel_fraction": float(
            np.quantile(limited_fractions, 0.95)
        ),
        "median_style_rgb_rmse_from_ao6": float(np.median(styles)),
    }
    passed = bool(
        aggregate["output_count"] == gate["expected_outputs"]
        and aggregate["maximum_output_code_boundary_fraction"]
        <= gate["maximum_output_code_boundary_fraction"]
        and aggregate["maximum_new_boundary_fraction_vs_ao6"]
        <= gate["maximum_new_boundary_fraction_vs_ao6"]
        and aggregate["median_residual_energy_retention"]
        >= gate["minimum_median_residual_energy_retention"]
        and aggregate["p95_safety_limited_pixel_fraction"]
        <= gate["maximum_p95_safety_limited_pixel_fraction"]
        and aggregate["median_style_rgb_rmse_from_ao6"]
        >= gate["minimum_median_style_rgb_rmse_from_ao6"]
    )
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "parent_stable_evidence_id": report_a["stable_evidence_id"],
        "rows": rows,
        "aggregate": aggregate,
        "automatic_gate_pass": passed,
        "visual_review_opened": passed,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["run_analytical_residual_ood"]
