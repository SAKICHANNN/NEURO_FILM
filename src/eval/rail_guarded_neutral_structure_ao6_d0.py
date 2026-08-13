"""P4IS source-inclusive analytical execution for P4IR neutral structure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.neutral_transmittance_structure_ao6_d0 import (
    apply_neutral_transmittance_structure,
)
from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate as evaluate_base
from src.eval.sigmoid_scanner_ao6_value_d1 import load_contract as load_base_contract
from src.roll2film.factorized_boundary_guard import apply_target_residual_boundary_guard

SCHEMA = "neuro-film.u6-p4is-rail-guarded-neutral-structure-ao6-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4is-rail-guarded-neutral-structure-ao6-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mechanism = payload.get("mechanism", {})
    if (
        payload.get("schema") != SCHEMA
        or mechanism.get("rail") != "source-inclusive-analytical-residual-scale"
        or mechanism.get("hard_clipping_allowed") is not False
        or mechanism.get("posthoc_limiting_allowed") is not False
        or mechanism.get("cohort_fit_allowed") is not False
    ):
        raise ValueError("unsupported P4IS contract")
    return payload


def apply_rail_guarded_neutral_structure(
    encoded_base: np.ndarray,
    source_linear: np.ndarray,
    index: int,
    amplitude: float,
    *,
    hard_boundary_epsilon: float,
    guard_boundary_epsilon: float,
) -> np.ndarray:
    linear_base = np.ascontiguousarray(
        encoded_srgb_to_linear(np.asarray(encoded_base, dtype=np.float64)),
        dtype=np.float32,
    )
    target = apply_neutral_transmittance_structure(
        linear_base, source_linear, index, amplitude
    )
    guarded = apply_target_residual_boundary_guard(
        linear_base,
        target,
        strength=1.0,
        hard_boundary_epsilon_encoded_srgb=hard_boundary_epsilon,
        guard_boundary_epsilon_encoded_srgb=guard_boundary_epsilon,
    )
    return np.ascontiguousarray(linear_srgb_to_encoded(guarded.output), dtype=np.float32)


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_lock = contract["parents"]["p4ir_evidence"]
    parent_path = root / parent_lock["path"]
    if _sha(parent_path) != parent_lock["sha256"]:
        raise ValueError("P4IS parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != parent_lock["required_decision"]:
        raise ValueError("P4IS parent decision drift")
    base_lock = contract["parents"]["base_contract"]
    base_path = root / base_lock["path"]
    if _sha(base_path) != base_lock["sha256"]:
        raise ValueError("P4IS base contract drift")
    mechanism = contract["mechanism"]

    def apply(
        encoded_base: np.ndarray,
        source_linear: np.ndarray,
        index: int,
        amplitude: float,
    ) -> np.ndarray:
        return apply_rail_guarded_neutral_structure(
            encoded_base,
            source_linear,
            index,
            amplitude,
            hard_boundary_epsilon=float(
                mechanism["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon=float(
                mechanism["guard_boundary_epsilon_encoded_srgb"]
            ),
        )

    result = evaluate_base(
        load_base_contract(base_path), root, post_base_structure_builder=apply
    )
    passed = bool(result["automatic_pass"])
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "parent_stable_evidence_id": parent["stable_evidence_id"],
        "ao6_bundle_sha256": result["ao6_bundle_sha256"],
        "mechanism": mechanism,
        "rows": result["rows"],
        "metrics": result["metrics"],
        "checks": result["checks"],
        "automatic_pass": passed,
        "blind_review_allowed": False,
        "fresh_confirmation_allowed": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
