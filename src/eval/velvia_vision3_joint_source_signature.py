"""U5.R2BW1 fail-closed joint MTF/granularity source-signature preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u5_r2bw1_velvia_vision3_joint_source_signature_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bw1_velvia_vision3_joint_source_signature_report.v1"
EXPERIMENT_ID = "U5.R2BW1"


class JointSourceSignatureError(RuntimeError):
    """Raised when the frozen BW1 contract or parent evidence drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise JointSourceSignatureError("BW1 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parents = payload.get("parents", {})
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or parents.get("bw0_decision", {}).get("sha256")
        != "19703d1236f7463b6569a4c73cca305e582ad8539b20527e96c4bc453baedcba"
        or parents.get("vision3_profile_decision", {}).get("sha256")
        != "d4c318a857f788b3ea5fdef292ba44f24f2d4386a03cb02928c77a24424c95a9"
        or parents.get("characteristic_trace", {}).get("sha256")
        != "e7d9d45cd56066d1cc770c3a506117fcddc4ae52417c6d69d4088e8d0da61dfb"
        or parents.get("granularity_trace", {}).get("sha256")
        != "00b3436c9619a477bf13dcae0ab90df6fd9470bc1976e77484983a4a0029f145"
        or comparison.get("velvia_profile") != "fujichrome_velvia_50_rvp50_af3_0221e2"
        or comparison.get("velvia_granularity_sensitivity_sigma_d")
        != [0.008, 0.009, 0.01]
        or gates.get("target_density_above_minimum") != 1.0
        or gates.get("minimum_nominal_absolute_log_granularity_ratio_each_profile")
        != 0.15
        or gates.get("minimum_joint_distance_each_profile") != 0.1
        or gates.get("minimum_worst_sensitivity_joint_distance_each_profile") != 0.08
        or gates.get("maximum_characteristic_axis_residual_px") != 2.0
        or gates.get("maximum_granularity_axis_residual_px") != 1.5
    ):
        raise JointSourceSignatureError("BW1 frozen contract drift")
    for lock in parents.values():
        _relative_path(str(lock.get("path", "")))
    return payload


def _stable_identity(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("report_sha256", None)
    stable.pop("stable_evidence_id", None)
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def audit_joint_source_signature(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parent_payloads: dict[str, dict[str, Any]] = {}
    parent_hashes: dict[str, str] = {}
    for name, lock in config["parents"].items():
        path = root / _relative_path(str(lock["path"]))
        if not path.is_file() or hash_file(path) != lock["sha256"]:
            raise JointSourceSignatureError(f"BW1 parent integrity mismatch: {name}")
        parent_hashes[name] = lock["sha256"]
        if name.endswith("decision"):
            parent_payloads[name] = json.loads(path.read_text(encoding="utf-8"))

    bw0 = parent_payloads["bw0_decision"]
    vision3 = parent_payloads["vision3_profile_decision"]
    bw0_decision_ok = (
        bw0.get("decision") == config["parents"]["bw0_decision"]["required_decision"]
    )
    vision3_decision_ok = (
        vision3.get("decision")
        == config["parents"]["vision3_profile_decision"]["required_decision"]
    )
    bw0_failed_gates = tuple(bw0.get("failed_gates", ()))
    upstream_axis_calibration_ok = "axis_calibration" not in bw0_failed_gates
    admissible = bool(
        bw0_decision_ok and vision3_decision_ok and upstream_axis_calibration_ok
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "parent_hashes": parent_hashes,
        "preflight": {
            "bw0_decision_ok": bw0_decision_ok,
            "vision3_profile_decision_ok": vision3_decision_ok,
            "bw0_failed_gates": list(bw0_failed_gates),
            "upstream_axis_calibration_ok": upstream_axis_calibration_ok,
        },
        "joint_feature_rows_computed": 0,
        "granularity_unit_conversions_applied": 0,
        "passed": False,
        "decision": (
            "score_joint_source_signature"
            if admissible
            else "close_before_joint_score_on_parent_source_calibration"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    if admissible:
        raise JointSourceSignatureError(
            "BW1 scoring stage is intentionally unavailable until preflight admits it"
        )
    report["stable_evidence_id"] = _stable_identity(report)
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "JointSourceSignatureError",
    "audit_joint_source_signature",
    "canonical_json",
    "hash_file",
    "load_contract",
    "write_report",
]
