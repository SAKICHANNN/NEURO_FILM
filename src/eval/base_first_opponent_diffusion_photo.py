"""P4IT fresh-photo evaluation of P4HX/P4IA after the fixed AO6 base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.opponent_diffusion_photographic_confirmation import (
    AnalyticalOpponentDiffusionRuntime,
    evaluate_with_runtime,
)
from src.eval.opponent_diffusion_photographic_confirmation import (
    load_contract as load_parent_contract,
)
from src.eval.p4hu_ao6_value import evaluate_base_first_source_arms

SCHEMA = "neuro-film.u6-p4it-base-first-opponent-diffusion-photo-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p4it-base-first-opponent-diffusion-photo-result.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mechanism = payload.get("mechanism", {})
    if (
        payload.get("schema") != SCHEMA
        or mechanism.get("source_context") != "original-encoded-source-only"
        or mechanism.get("P4HX_parameters_unchanged") is not True
        or mechanism.get("P4IA_envelope_unchanged") is not True
        or mechanism.get("hard_clipping_allowed") is not False
        or mechanism.get("posthoc_limiting_allowed") is not False
        or mechanism.get("cohort_fitting_allowed") is not False
    ):
        raise ValueError("unsupported P4IT contract")
    return payload


def evaluate(contract: dict[str, Any], *, root: Path, output_dir: Path) -> dict[str, Any]:
    evidence_lock = contract["p4is_evidence"]
    evidence_path = root / evidence_lock["path"]
    if _sha(evidence_path) != evidence_lock["sha256"]:
        raise ValueError("P4IT P4IS identity drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != evidence_lock["required_decision"]:
        raise ValueError("P4IT P4IS decision drift")
    parent_lock = contract["parent_contract"]
    parent_path = root / parent_lock["path"]
    if _sha(parent_path) != parent_lock["sha256"]:
        raise ValueError("P4IT parent contract drift")
    parent = load_parent_contract(parent_path)
    parent["schema"] = (
        "neuro-film.u6-p4hz-opponent-diffusion-photographic-confirmation-contract.v1"
    )
    parent["experiment_id"] = contract["experiment_id"]
    parent["comparison"]["arms"] = [
        "fixed_ao6_colour_only_t15_c35",
        "matched_scanner_only_ao6_t15_c35",
        "p4it_base_first_opponent_diffusion_physical_only_diagnostic",
        "p4it_base_first_opponent_diffusion_then_fixed_ao6_t15_c35",
    ]
    parent["decision_if_pass"] = contract["decision_if_pass"]
    parent["decision_if_fail"] = contract["decision_if_fail"]
    parent["claim_ceiling"] = contract["claim_ceiling"]
    return evaluate_with_runtime(
        parent,
        root=root,
        output_dir=output_dir,
        runtime_class=AnalyticalOpponentDiffusionRuntime,
        result_schema=RESULT_SCHEMA,
        source_arm_evaluator=evaluate_base_first_source_arms,
    )
