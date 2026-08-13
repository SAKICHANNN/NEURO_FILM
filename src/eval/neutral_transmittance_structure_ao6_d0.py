"""P4IP achromatic transmittance-structure ablation after colour mixing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate as evaluate_base
from src.eval.sigmoid_scanner_ao6_value_d1 import load_contract as load_base_contract

SCHEMA = "neuro-film.u6-p4ip-neutral-transmittance-structure-ao6-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ip-neutral-transmittance-structure-ao6-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mechanism = payload.get("mechanism", {})
    if (
        payload.get("schema") != SCHEMA
        or mechanism.get("coordinate") != "scan-linear-achromatic-transmittance"
        or mechanism.get("hard_clipping_allowed") is not False
        or mechanism.get("posthoc_limiting_allowed") is not False
        or mechanism.get("colour_ratio_change_allowed") is not False
        or mechanism.get("cohort_fit_allowed") is not False
    ):
        raise ValueError("unsupported P4IP contract")
    return payload


def apply_neutral_transmittance_structure(
    baseline: np.ndarray,
    source_linear: np.ndarray,
    index: int,
    amplitude: float,
) -> np.ndarray:
    del source_linear
    value = np.asarray(baseline)
    if (
        value.dtype != np.float32
        or value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
        or not np.isfinite(amplitude)
        or amplitude < 0.0
    ):
        raise ValueError("invalid P4IP input")
    height, width = value.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    field = np.sin((x + 11 * index) / 23.0) * np.cos((y - 7 * index) / 19.0)
    desired = np.power(10.0, -float(amplitude) * field)
    maximum = np.max(value.astype(np.float64), axis=-1)
    safe = np.where(maximum > 0.0, np.minimum(desired, 1.0 / maximum), desired)
    return np.ascontiguousarray(value.astype(np.float64) * safe[..., None], dtype=np.float32)


def evaluate(
    contract: Mapping[str, Any], root: Path, *, contact_path: Path | None = None
) -> dict[str, Any]:
    p4in_lock = contract["parents"]["p4in_evidence"]
    p4in_path = root / str(p4in_lock["path"])
    if not p4in_path.is_file() or _sha(p4in_path) != p4in_lock["sha256"]:
        raise ValueError("P4IP parent identity drift")
    p4in = json.loads(p4in_path.read_text(encoding="utf-8"))
    if p4in.get("decision") != p4in_lock["required_decision"]:
        raise ValueError("P4IP parent decision drift")
    base_lock = contract["parents"]["base_contract"]
    base_path = root / str(base_lock["path"])
    if not base_path.is_file() or _sha(base_path) != base_lock["sha256"]:
        raise ValueError("P4IP base contract drift")
    result = evaluate_base(
        load_base_contract(base_path),
        root,
        contact_path=contact_path,
        scan_structure_builder=apply_neutral_transmittance_structure,
    )
    passed = bool(result["automatic_pass"])
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": p4in["stable_evidence_id"],
        "ao6_bundle_sha256": result["ao6_bundle_sha256"],
        "mechanism": contract["mechanism"],
        "rows": result["rows"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "automatic_pass": passed,
        "blind_review_allowed": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    if "contact_sheet_sha256" in result:
        stable["contact_sheet_sha256"] = result["contact_sheet_sha256"]
    identity = {key: value for key, value in stable.items() if key != "contact_sheet_sha256"}
    return {**stable, "stable_evidence_id": hashlib.sha256(_canonical(identity)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["apply_neutral_transmittance_structure", "evaluate", "load_contract", "write_report"]
