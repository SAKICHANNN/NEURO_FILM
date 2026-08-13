"""Energy-matched common-mode density-structure ablation after AO6."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate as evaluate_base
from src.eval.sigmoid_scanner_ao6_value_d1 import load_contract as load_base_contract

SCHEMA = "neuro-film.u6-p4in-equal-density-structure-ao6-ablation-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4in-equal-density-structure-ao6-ablation-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IN contract")
    return value


def evaluate(
    contract: Mapping[str, Any],
    root: Path,
    *,
    contact_path: Path | None = None,
) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if _sha(parent_path) != contract["parent"]["sha256"]:
        raise ValueError("P4IN parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("P4IN parent decision drift")
    base_path = root / str(contract["base_contract"]["path"])
    if _sha(base_path) != contract["base_contract"]["sha256"]:
        raise ValueError("P4IN base contract drift")
    original = np.asarray(contract["structure"]["original_direction"], dtype=np.float64)
    candidate = np.asarray(
        contract["structure"]["candidate_equal_channel_direction"],
        dtype=np.float64,
    )
    if (
        original.shape != (3,)
        or candidate.shape != (3,)
        or not np.all(np.isfinite(original))
        or not np.all(np.isfinite(candidate))
        or not np.all(candidate == candidate[0])
        or abs(float(np.linalg.norm(original) - np.linalg.norm(candidate))) > 1e-12
        or not contract["structure"].get("require_exact_l2_energy_match")
        or contract["structure"].get("amplitude_changed")
        or contract["structure"].get("spatial_field_changed")
    ):
        raise ValueError("P4IN energy-matched structure contract drift")
    result = evaluate_base(
        load_base_contract(base_path),
        root,
        contact_path=contact_path,
        structure_direction=tuple(float(value) for value in candidate),
    )
    passed = bool(result["automatic_pass"])
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "base_ao6_bundle_sha256": result["ao6_bundle_sha256"],
        "structure_direction": candidate.tolist(),
        "direction_l2_energy": float(np.dot(candidate, candidate)),
        "rows": result["rows"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "automatic_pass": passed,
        "blind_review_allowed": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    if "contact_sheet_sha256" in result:
        core["contact_sheet_sha256"] = result["contact_sheet_sha256"]
    stable = {key: value for key, value in core.items() if key != "contact_sheet_sha256"}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

