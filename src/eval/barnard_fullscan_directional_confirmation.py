"""U6.P6AY same-plate acquisition-scale confirmation of P6AX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.rotated_plate_coherence_d0 import _load_u16
from src.eval.rotated_plate_directional_attribution_d0 import (
    _correlation,
    _directional_profile,
)

SCHEMA = "neuro-film.u6-p6ay-barnard-fullscan-directional-confirmation-d0-contract.v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P6AY contract")
    return value


def _load_profile(
    root: Path, source: dict[str, Any], analysis: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    values, facts = _load_u16(root / source["path"], source)
    result = _directional_profile(values, analysis)
    return result, {
        **facts,
        "patches": result["patches"],
        "valid_frequency_bins": result["valid_frequency_bins"],
        "residual_standard_deviation": result["residual_standard_deviation"],
        "profile_sha256": result["profile_sha256"],
    }


def _score(
    target_reference: np.ndarray,
    target_rotated: np.ndarray,
    wrong_reference: np.ndarray,
    wrong_rotated: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    plate = _correlation(target_reference, np.rot90(target_rotated), mask)
    scanner = _correlation(target_reference, target_rotated, mask)
    wrong = max(
        _correlation(target_reference, wrong_reference, mask),
        _correlation(target_reference, wrong_rotated, mask),
        _correlation(target_reference, np.rot90(wrong_rotated), mask),
    )
    return {
        "plate_following_correlation": plate,
        "scanner_fixed_correlation": scanner,
        "wrong_plate_correlation": wrong,
        "plate_following_margin": plate - max(scanner, wrong),
        "scanner_fixed_margin": scanner - max(plate, wrong),
    }


def evaluate(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    parent_spec = contract["parents"]["p6ax_evidence"]
    parent_payload = (root / parent_spec["path"]).read_bytes()
    parent = json.loads(parent_payload)
    if (
        _sha(parent_payload) != parent_spec["sha256"]
        or parent.get("status") != parent_spec["required_status"]
    ):
        raise ValueError("P6AY P6AX parent drift")
    center_row = next(
        row for row in parent["rows"] if row["id"] == "uchicago_1905_barnard_10b161"
    )
    center_margin = float(center_row["plate_following_margin"])
    if center_margin != float(parent_spec["required_center_barnard_margin"]):
        raise ValueError("P6AY center Barnard margin drift")
    source_lock = contract["parents"]["source_lock"]
    if _sha((root / source_lock["path"]).read_bytes()) != source_lock["sha256"]:
        raise ValueError("P6AY source lock drift")
    analysis = contract["analysis"]
    target_reference, target_reference_facts = _load_profile(
        root, contract["target"]["reference"], analysis
    )
    target_rotated, target_rotated_facts = _load_profile(
        root, contract["target"]["rotated"], analysis
    )
    wrong_reference, wrong_reference_facts = _load_profile(
        root, contract["wrong_plate_control"]["reference"], analysis
    )
    wrong_rotated, wrong_rotated_facts = _load_profile(
        root, contract["wrong_plate_control"]["rotated"], analysis
    )
    metrics = _score(
        target_reference["profile"],
        target_rotated["profile"],
        wrong_reference["profile"],
        wrong_rotated["profile"],
        target_reference["mask"],
    )
    gates = contract["gates"]
    facts = [
        target_reference_facts,
        target_rotated_facts,
        wrong_reference_facts,
        wrong_rotated_facts,
    ]
    support_pass = all(
        row["patches"] == int(gates["required_patches_per_scan"])
        and row["valid_frequency_bins"] >= int(gates["minimum_valid_frequency_bins"])
        and np.isfinite(row["residual_standard_deviation"])
        and row["residual_standard_deviation"]
        >= float(gates["minimum_residual_standard_deviation"])
        for row in facts
    ) and all(np.isfinite(value) for value in metrics.values())
    margin_ratio = metrics["plate_following_margin"] / center_margin
    plate_pass = bool(
        metrics["plate_following_margin"]
        >= float(gates["minimum_plate_following_margin"])
    )
    ratio_pass = bool(
        margin_ratio >= float(gates["minimum_margin_retainment_ratio_vs_center_scan"])
    )
    scanner_pass = bool(
        metrics["scanner_fixed_margin"]
        >= float(gates["minimum_plate_following_margin"])
    )
    automatic_pass = bool(
        support_pass
        and plate_pass
        and ratio_pass
        and scanner_pass == bool(gates["required_scanner_fixed_pass"])
    )
    scientific = {
        "schema": "neuro-film.u6-p6ay-barnard-fullscan-directional-confirmation-d0-report.v1",
        "experiment_id": contract["experiment_id"],
        "config_sha256": _sha(_canonical(contract)),
        "source": {
            "target": {
                "reference": target_reference_facts,
                "rotated": target_rotated_facts,
            },
            "wrong_plate_control": {
                "reference": wrong_reference_facts,
                "rotated": wrong_rotated_facts,
            },
        },
        "metrics": {**metrics, "margin_retainment_ratio_vs_center_scan": margin_ratio},
        "gates": {
            "support_pass": support_pass,
            "plate_following_pass": plate_pass,
            "margin_retainment_pass": ratio_pass,
            "scanner_fixed_pass": scanner_pass,
        },
        "status": (
            "PASS_BARNARD_WITHIN_PLATE_ACQUISITION_SCALE_CONFIRMATION_D0"
            if automatic_pass
            else "FAIL_BARNARD_FULLSCAN_DIRECTIONAL_CONFIRMATION_D0"
        ),
        "automatic_pass": automatic_pass,
        "decision": (
            contract["decision_if_pass"]
            if automatic_pass
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**scientific, "stable_evidence_id": _sha(_canonical(scientific))}
