"""U6.P7I analytical scanner-safe residual value test over frozen P7H."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.eval.native_msvc import sha256_file
from src.eval.p4hu_ao6_value import evaluate as evaluate_p7h

CONTRACT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-result.v1"


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def _path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("P7I bindings must be repository-relative")
    return root / value


def _bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = _path(root, binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P7I parent identity drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_decision" in binding:
        decision = payload.get("decision", payload.get("result", {}).get("decision"))
        if decision != binding["required_decision"]:
            raise ValueError(f"P7I parent decision drift: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P7I contract")
    return payload


def _validate(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or candidate.get("scanner_safe_residual_runtime_id") != "scanner-safe-residual-analytic-v1"
        or candidate.get("application_domain") != "post-scanner-linear-before-encode-and-ao6"
        or candidate.get("reference_arm_domain") != "matched-scanner-source-linear"
        or candidate.get("direction_change_allowed") is not False
        or candidate.get("p4hu_or_ao6_refit_allowed") is not False
    ):
        raise ValueError("P7I candidate policy drift")
    gates = contract.get("additional_gates", {})
    expected = {"maximum_limited_pixel_fraction", "minimum_median_scale", "maximum_collinearity_error"}
    if set(gates) != expected or any(
        isinstance(gates[key], bool) or not isinstance(gates[key], (int, float)) or not math.isfinite(float(gates[key]))
        for key in expected
    ):
        raise ValueError("P7I additional gate drift")
    if not (0 <= gates["maximum_limited_pixel_fraction"] <= 1 and 0 <= gates["minimum_median_scale"] <= 1 and gates["maximum_collinearity_error"] >= 0):
        raise ValueError("P7I additional gate range drift")


def evaluate(contract: dict[str, Any], *, root: Path, output_dir: Path, build_dir: Path) -> dict[str, Any]:
    _validate(contract)
    parents = contract["parents"]
    p7h_contract = _bound_json(root, parents["p7h_contract"])
    for name in ("p7h_evidence", "p6am_evidence", "p6al_evidence"):
        _bound_json(root, parents[name])
    result = evaluate_p7h(
        p7h_contract,
        root=root,
        output_dir=output_dir,
        build_dir=build_dir,
        scanner_safe_residual=True,
    )
    receipt = result["aggregates"].get("scanner_safe_residual")
    if not isinstance(receipt, dict):
        raise TypeError("P7I scanner-safe receipt missing")
    gates = contract["additional_gates"]
    additional_checks = {
        "limited_pixel_fraction": receipt["maximum_limited_pixel_fraction"] <= gates["maximum_limited_pixel_fraction"],
        "median_scale": receipt["minimum_median_scale"] >= gates["minimum_median_scale"],
        "collinearity": receipt["maximum_collinearity_error"] <= gates["maximum_collinearity_error"],
    }
    automatic_pass = result["automatic_pass"] and all(additional_checks.values())
    core = {
        **{key: value for key, value in result.items() if key != "stable_evidence_id"},
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _canonical_sha256(contract),
        "parent_p7h_contract_sha256": parents["p7h_contract"]["sha256"],
        "scanner_safe_residual_gates": additional_checks,
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = ["CONTRACT_SCHEMA", "RESULT_SCHEMA", "evaluate", "load_contract"]
