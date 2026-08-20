"""Facts-only RF3.D1 target-evidence equalization gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class TargetEvidenceError(RuntimeError):
    """Raised when a frozen parent binding or fact contract is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_bound(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = root / spec["path"]
    raw = path.read_bytes()
    if _sha256(raw) != spec["sha256"]:
        raise TargetEvidenceError(f"parent hash mismatch: {spec['id']}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TargetEvidenceError(f"parent is not an object: {spec['id']}")
    return value


def evaluate(contract_path: Path, root: Path) -> dict[str, Any]:
    contract_raw = contract_path.read_bytes()
    contract = json.loads(contract_raw)
    parents = {spec["id"]: _load_bound(root, spec) for spec in contract["parents"]}

    rf3 = parents["rf3_d0"]
    connected = parents["sf1_0b"]
    shared = parents["sf1_3b"]
    portra = parents["portra_cham5"]
    velvia = parents["velvia_p6y"]

    matrix = {
        "fujifilm_velvia_50": {
            "paired_target_available": False,
            "explicit_pixel_reuse_rights": bool(
                velvia["results"]["explicit_pixel_reuse_rights"]
            ),
            "independent_holdout_available": False,
            "blocking_fact": velvia["decision"],
        },
        "kodak_portra_400": {
            "paired_target_available": True,
            "explicit_pixel_reuse_rights": False,
            "independent_holdout_available": False,
            "blocking_fact": portra["scientific_conclusion"],
        },
        "kodak_ektar_100": {
            "paired_target_available": False,
            "explicit_pixel_reuse_rights": True,
            "independent_holdout_available": False,
            "blocking_fact": shared["decision"],
        },
    }

    source_content_clean = (
        connected["decision"]
        != "close_current_four_cell_pools_for_stock_learning_and_open_metadata_only_full_yfcc_shared_author_gate"
        and shared["decision"] != "close_shared_author_pixel_pool_for_stock_learning"
    )
    observed = {
        "same_level_paired_targets_for_all_stocks": all(
            row["paired_target_available"] for row in matrix.values()
        ),
        "explicit_reuse_rights_for_all_target_pixels": all(
            row["explicit_pixel_reuse_rights"] for row in matrix.values()
        ),
        "independent_roll_process_scanner_holdout_for_all_stocks": all(
            row["independent_holdout_available"] for row in matrix.values()
        ),
        "zero_source_content_identifiability_failure": source_content_clean,
    }
    required = contract["admission_gates"]
    gates = {name: observed[name] is expected for name, expected in required.items()}
    automatic_pass = all(gates.values())

    if rf3["source_evidence_matrix"]["same_level_paired_stock_truth_available"]:
        raise TargetEvidenceError(
            "RF3.D0 contradicts the frozen missing-target premise"
        )
    if portra["operator_fit_allowed"]:
        raise TargetEvidenceError("Portra parent unexpectedly admits operator fitting")

    core = {
        "schema": "neuro-film.rf3-d1-three-stock-target-evidence-equalization-report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "parent_sha256": {spec["id"]: spec["sha256"] for spec in contract["parents"]},
        "stock_matrix": matrix,
        "observed_admission_facts": observed,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "pixel_reads": 0,
        "operator_fits": 0,
        "renders": 0,
        "training_rows": 0,
        "forbidden_rescues": contract["forbidden_rescues"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    core["stable_evidence_id"] = _sha256(_canonical(core))
    return core
