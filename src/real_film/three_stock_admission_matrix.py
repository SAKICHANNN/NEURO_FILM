"""Compile the current three-stock evidence into fail-closed K=1 admissions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro-film.sf3-a3-three-stock-admission-matrix-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a3-three-stock-admission-matrix-report.v1"
STOCK_IDS = [
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
]
GATE_KEYS = [
    "rights_clear",
    "pixels_available",
    "same_scene_digital_reference",
    "independent_roll_process_scanner_holdout",
    "stock_identifiability_passed",
]


class ThreeStockAdmissionError(RuntimeError):
    """Raised when the frozen admission contract or evidence drifts."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockAdmissionError("evidence paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bindings = payload.get("evidence_bindings")
    lanes = payload.get("source_lanes")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id")
        != "SF3.A3_THREE_STOCK_ADMISSION_MATRIX_V1"
        or payload.get("required_stock_ids") != STOCK_IDS
        or not isinstance(bindings, list)
        or not isinstance(lanes, list)
        or not bindings
        or not lanes
        or payload.get("k1_admission_gates") != {key: True for key in GATE_KEYS}
    ):
        raise ThreeStockAdmissionError("SF3.A3 frozen contract drift")
    binding_ids = [str(row.get("id")) for row in bindings]
    source_ids = [str(row.get("source_id")) for row in lanes]
    if len(binding_ids) != len(set(binding_ids)) or len(source_ids) != len(
        set(source_ids)
    ):
        raise ThreeStockAdmissionError("duplicate evidence or source lane identity")
    for binding in bindings:
        _relative(str(binding.get("path", "")))
        if len(str(binding.get("sha256", ""))) != 64:
            raise ThreeStockAdmissionError("invalid evidence hash")
        if not binding.get("required_decision") and not binding.get("required_status"):
            raise ThreeStockAdmissionError("evidence binding lacks a required state")
    for lane in lanes:
        stocks = lane.get("stocks")
        if (
            not isinstance(stocks, list)
            or not stocks
            or any(stock not in STOCK_IDS for stock in stocks)
            or any(key not in lane or not isinstance(lane[key], bool) for key in GATE_KEYS)
            or not lane.get("valid_role")
            or not lane.get("blocking_reason")
        ):
            raise ThreeStockAdmissionError("invalid source lane")
    if set().union(*(set(row["stocks"]) for row in lanes)) != set(STOCK_IDS):
        raise ThreeStockAdmissionError("source lanes do not cover all required stocks")
    return payload


def _validate_evidence(
    root: Path, bindings: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        path = root / _relative(str(binding["path"]))
        if not path.is_file() or _hash_file(path) != binding["sha256"]:
            raise ThreeStockAdmissionError(f"evidence drift: {binding['id']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ThreeStockAdmissionError(f"invalid evidence payload: {binding['id']}")
        for key in ("decision", "status"):
            expected = binding.get(f"required_{key}")
            if expected is not None and payload.get(key) != expected:
                raise ThreeStockAdmissionError(
                    f"evidence {key} drift: {binding['id']}"
                )
        validated.append(
            {
                "id": binding["id"],
                "path": binding["path"],
                "sha256": binding["sha256"],
            }
        )
    return validated


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Validate bound evidence and compile per-stock K=1 admission state."""

    evidence = _validate_evidence(root, list(contract["evidence_bindings"]))
    required = contract["k1_admission_gates"]
    stock_rows: list[dict[str, Any]] = []
    for stock_id in STOCK_IDS:
        options: list[dict[str, Any]] = []
        for lane in contract["source_lanes"]:
            if stock_id not in lane["stocks"]:
                continue
            checks = {key: lane[key] is required[key] for key in GATE_KEYS}
            options.append(
                {
                    "source_id": lane["source_id"],
                    "valid_role": lane["valid_role"],
                    "gate_checks": checks,
                    "k1_fit_admitted": all(checks.values()),
                    "blocking_reason": None
                    if all(checks.values())
                    else lane["blocking_reason"],
                }
            )
        admitted = [row["source_id"] for row in options if row["k1_fit_admitted"]]
        stock_rows.append(
            {
                "film_stock_id": stock_id,
                "k1_fit_admitted": bool(admitted),
                "admitted_source_lanes": admitted,
                "source_options": options,
            }
        )

    all_admitted = all(row["k1_fit_admitted"] for row in stock_rows)
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "validated_evidence": evidence,
        "stock_rows": stock_rows,
        "all_three_stocks_k1_fit_admitted": all_admitted,
        "current_baseline": {
            "experiment_id": "RF3.D0",
            "role": "development_only_three_stock_k1_proxy_matrix",
            "ao6_role": "velvia_50_author_rendered_display_proxy_look_approximation_baseline_only",
            "target_closeness_evaluable": False,
            "stock_distinguishability_evaluable": False,
        },
        "blocked_candidate_families": [
            "image_adaptive_lut_or_basis",
            "nearest_neighbour_retrieval",
            "hard_medoid_or_sparse_retrieval",
            "k_greater_than_1_or_latent_mode_routing",
        ],
        "next_executable_leaf": (
            "populate_and_validate_first_ready_sf3_a0_single_stock_manifest_"
            "then_complete_three_stock_controls"
        ),
        "decision": contract["decision_if_all_stocks_admitted"]
        if all_admitted
        else contract["decision_if_any_stock_blocked"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256(raw)
