"""Compile strongest real-colour observations into stock K=1 data admissions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro-film.sf3-a3d-partial-real-observation-matrix-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a3d-partial-real-observation-matrix-report.v1"
EXPERIMENT_ID = "SF3.A3D_PARTIAL_REAL_OBSERVATION_MATRIX_V1"
STOCK_IDS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
GATE_KEYS = [
    "original_pixel_payload_available",
    "rights_clear_for_operator_fit",
    "same_scene_digital_reference",
    "independent_roll_holdout",
    "process_separated",
    "scanner_separated",
    "transferable_stock_signal_passed",
]


class PartialObservationMatrixError(RuntimeError):
    """Raised when a bound source or frozen data gate drifts."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PartialObservationMatrixError("paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bindings = payload.get("evidence_bindings")
    lanes = payload.get("observation_lanes")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("required_stock_ids") != STOCK_IDS
        or payload.get("k1_fit_gates") != {key: True for key in GATE_KEYS}
        or not isinstance(bindings, list)
        or not isinstance(lanes, list)
        or len(lanes) != len(STOCK_IDS)
    ):
        raise PartialObservationMatrixError("frozen contract drift")
    binding_ids = [str(row.get("id")) for row in bindings]
    if len(binding_ids) != len(set(binding_ids)):
        raise PartialObservationMatrixError("duplicate evidence identity")
    for binding in bindings:
        _relative(str(binding.get("path", "")))
        if len(str(binding.get("sha256", ""))) != 64 or not binding.get("required_decision"):
            raise PartialObservationMatrixError("invalid evidence binding")
    if [row.get("film_stock_id") for row in lanes] != STOCK_IDS:
        raise PartialObservationMatrixError("observation stock order drift")
    for lane in lanes:
        evidence_ids = lane.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or any(item not in binding_ids for item in evidence_ids)
            or any(key not in lane or not isinstance(lane[key], bool) for key in GATE_KEYS)
            or not lane.get("observation_id")
            or not lane.get("observation_grade")
            or not lane.get("blocking_reason")
            or not lane.get("next_required_observation")
        ):
            raise PartialObservationMatrixError("invalid observation lane")
    return payload


def _validate_evidence(root: Path, bindings: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for binding in bindings:
        path = root / _relative(str(binding["path"]))
        if not path.is_file() or _hash_file(path) != binding["sha256"]:
            raise PartialObservationMatrixError(f"evidence drift: {binding['id']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decision") != binding["required_decision"]:
            raise PartialObservationMatrixError(f"evidence decision drift: {binding['id']}")
        rows.append({key: str(binding[key]) for key in ("id", "path", "sha256")})
    return rows


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    evidence = _validate_evidence(root, list(contract["evidence_bindings"]))
    stock_rows: list[dict[str, Any]] = []
    for lane in contract["observation_lanes"]:
        checks = {key: bool(lane[key]) for key in GATE_KEYS}
        admitted = all(checks.values())
        stock_rows.append(
            {
                "film_stock_id": lane["film_stock_id"],
                "observation_id": lane["observation_id"],
                "observation_grade": lane["observation_grade"],
                "evidence_ids": lane["evidence_ids"],
                "gate_checks": checks,
                "k1_fit_admitted": admitted,
                "blocking_reason": None if admitted else lane["blocking_reason"],
                "next_required_observation": lane["next_required_observation"],
            }
        )
    all_admitted = all(row["k1_fit_admitted"] for row in stock_rows)
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "validated_evidence": evidence,
        "stock_rows": stock_rows,
        "stocks_with_any_same_scene_real_colour_observation": sum(
            row["gate_checks"]["same_scene_digital_reference"] for row in stock_rows
        ),
        "stocks_k1_fit_admitted": sum(row["k1_fit_admitted"] for row in stock_rows),
        "all_three_stocks_k1_fit_admitted": all_admitted,
        "first_data_priority": stock_rows[2]["next_required_observation"],
        "decision": contract["decision_if_all_admitted"] if all_admitted else contract["decision_if_any_blocked"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
