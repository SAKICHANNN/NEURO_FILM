"""Best-basic explainability audit for fixed fresh FilmMatch renders."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.eval.filmmatch_identity_residual_ood import _encode_scene_linear
from src.eval.filmmatch_strict_interior_fresh_confirmation import (
    AO6_ARM,
    CANDIDATE_ARM,
    validate_contract as validate_bl8_contract,
)
from src.eval.global_frontier import sha256_file
from src.preprocess.raw_decode import load_raw_working_image
from src.real_film.gold_matrix_transplant import style_and_basic_residual


SCHEMA = "neuro_film.u5_r2bl9_filmmatch_fresh_nonbasic_audit_report.v1"


class FreshNonBasicAuditError(RuntimeError):
    """Raised when the fixed BL9 evaluator or its parent evidence drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _sample_aligned(source: np.ndarray, output: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != output.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise FreshNonBasicAuditError("aligned RGB shape mismatch")
    flat_source = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    flat_output = np.asarray(output, dtype=np.float32).reshape(-1, 3)
    count = min(maximum, flat_source.shape[0])
    if count < 256:
        raise FreshNonBasicAuditError("insufficient pixels for best-basic audit")
    indices = np.linspace(0, flat_source.shape[0] - 1, num=count, dtype=np.int64)
    return flat_source[indices], flat_output[indices]


def _load_rgb16(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise FreshNonBasicAuditError("render identity drift")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise FreshNonBasicAuditError("expected RGB16 PNG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != "neuro_film.u5_r2bl9_filmmatch_fresh_nonbasic_audit.v1"
        or config.get("status") != "contract_frozen_evaluation_ready"
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_default_changed")
        or int(config["evaluation"]["maximum_pixels_per_image"]) < 256
    ):
        raise FreshNonBasicAuditError("BL9 frozen contract drift")
    parent = config["parent"]
    bl8_config_path = root / parent["bl8_contract"]
    if sha256_file(bl8_config_path) != parent["bl8_contract_sha256"]:
        raise FreshNonBasicAuditError("BL8 contract identity drift")
    bl8_config = json.loads(bl8_config_path.read_text(encoding="utf-8"))
    validated = validate_bl8_contract(root, bl8_config)
    report_path = root / parent["bl8_report"]
    if sha256_file(report_path) != parent["bl8_report_sha256"]:
        raise FreshNonBasicAuditError("BL8 report identity drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["automatic_gate_pass"] or report["stable_evidence_id"] != parent["bl8_stable_evidence_id"]:
        raise FreshNonBasicAuditError("BL8 automatic gate is not open")
    records = {(str(row["source_id"]), str(row["arm_id"])): row for row in report["rows"]}
    expected = {(source_id, arm) for source_id in validated["eligible_ids"] for arm in (AO6_ARM, CANDIDATE_ARM)}
    if set(records) != expected:
        raise FreshNonBasicAuditError("BL8 render inventory drift")
    return {**validated, "records": records}


def run_nonbasic_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise FreshNonBasicAuditError("BL9 requires OMP_NUM_THREADS=1")
    if output_path.exists():
        raise FileExistsError("BL9 output is create-only")
    budget = int(config["evaluation"]["maximum_pixels_per_image"])
    rows: list[dict[str, Any]] = []
    for source_id in validated["eligible_ids"]:
        source_row = validated["source_rows"][source_id]
        raw_path = root / source_row["raw_path"]
        if sha256_file(raw_path) != source_row["raw_sha256"]:
            raise FreshNonBasicAuditError("live RAW identity drift")
        working = load_raw_working_image(raw_path)
        encoded = _encode_scene_linear(working.pixels)
        for arm_id in (AO6_ARM, CANDIDATE_ARM):
            record = validated["records"][(source_id, arm_id)]
            output = _load_rgb16(root / config["parent"]["bl8_output_dir"] / record["output"], record["output_sha256"])
            sampled_source, sampled_output = _sample_aligned(encoded, output, budget)
            style, residual = style_and_basic_residual(sampled_source, sampled_output)
            rows.append(
                {
                    "source_id": source_id,
                    "make": source_row["make"],
                    "arm_id": arm_id,
                    "sample_count": sampled_source.shape[0],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "non_basic_to_style_ratio": residual / max(style, 1e-12),
                    "output_sha256": record["output_sha256"],
                }
            )
        del working, encoded
    by_arm = {
        arm: [row for row in rows if row["arm_id"] == arm]
        for arm in (AO6_ARM, CANDIDATE_ARM)
    }
    summaries: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        summaries[arm] = {
            "median_style_delta_e76": float(np.median([row["median_style_delta_e76"] for row in arm_rows])),
            "median_non_basic_residual_delta_e76": float(np.median([row["median_non_basic_residual_delta_e76"] for row in arm_rows])),
            "median_non_basic_to_style_ratio": float(np.median([row["non_basic_to_style_ratio"] for row in arm_rows])),
            "images_above_non_basic_floor": sum(row["median_non_basic_residual_delta_e76"] >= float(config["automatic_gate"]["per_image_non_basic_delta_e76_floor"]) for row in arm_rows),
        }
    candidate = summaries[CANDIDATE_ARM]
    gate = config["automatic_gate"]
    gates = {
        "median_non_basic_floor": candidate["median_non_basic_residual_delta_e76"] >= float(gate["minimum_candidate_median_non_basic_delta_e76"]),
        "median_non_basic_ratio": candidate["median_non_basic_to_style_ratio"] >= float(gate["minimum_candidate_median_non_basic_to_style_ratio"]),
        "broad_image_support": candidate["images_above_non_basic_floor"] >= int(gate["minimum_images_above_non_basic_floor"]),
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "rows": rows,
        "summaries": summaries,
        "automatic_gates": gates,
        "automatic_gate_pass": all(gates.values()),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["run_nonbasic_audit", "validate_contract"]
