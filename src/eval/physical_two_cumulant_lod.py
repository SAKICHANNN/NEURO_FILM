"""U6.P4H analytic two-cumulant physical-LOD audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.physical_density_conditioned_lod import (
    evaluate_density_conditioned_lod_with_compiler,
)
from src.eval.physical_structure_boundary_semantics import (
    _maximum_flat_boundary_mean_error,
    _normalized_profiles,
)
from src.film_physics.density_conditioned_structure import (
    compile_two_cumulant_profiles,
)


SCHEMA = "neuro_film.u6_p4h_two_cumulant_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4h_two_cumulant_lod_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4H parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4H contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["compiler"]["per_image_fit_or_normalization_allowed"]
        or contract["compiler"]["empirical_parameter_fit_allowed"]
        or contract["compiler"]["output_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4H contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    _load_exact(
        root,
        parents["p4g_contract"],
        parents["p4g_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4g_decision"],
        parents["p4g_decision_sha256"],
    )
    if (
        decision["decision"]
        != "close_normalized_support_retain_p4d_base_pitch_only"
        or not decision["next_leaf"].startswith("U6.P4H")
    ):
        raise ValueError("U6.P4G did not open the two-cumulant audit")
    return contract, parent, decision


def evaluate_two_cumulant_lod(
    contract: dict[str, Any],
    parent: dict[str, Any],
    parent_decision: dict[str, Any],
) -> dict[str, Any]:
    report = evaluate_density_conditioned_lod_with_compiler(
        contract,
        parent,
        profile_compiler=compile_two_cumulant_profiles,
        report_schema=REPORT_SCHEMA,
        base_profile_transform=_normalized_profiles,
    )
    flat_errors = {
        "development": _maximum_flat_boundary_mean_error(
            parent,
            seed_offset=int(
                contract["synthetic_scene"]["development_seed_offset"]
            ),
        ),
        "confirmation": _maximum_flat_boundary_mean_error(
            parent,
            seed_offset=int(
                contract["synthetic_scene"]["confirmation_seed_offset"]
            ),
        ),
    }
    parent_variance = float(
        parent_decision["results"][
            "normalized_support_confirmation_factor_2_maximum_reference_variance_ratio"
        ]
    )
    gates = contract["automatic_gates"]
    extra_checks = {
        "single_cumulant_negative_control": parent_variance
        >= float(
            gates[
                "minimum_single_cumulant_confirmation_factor_2_variance_failure"
            ]
        ),
        "normalized_boundary_flat_mean": max(flat_errors.values())
        <= float(gates["maximum_normalized_boundary_flat_mean_error"]),
    }
    checks = {**report["checks"], **extra_checks}
    passed = all(checks.values())
    report.update(
        {
            "checks": checks,
            "automatic_pass": passed,
            "decision": (
                contract["branch_rule"]["pass"]
                if passed
                else contract["branch_rule"]["fail"]
            ),
            "single_cumulant_confirmation_factor_2_maximum_reference_variance_ratio": (
                parent_variance
            ),
            "normalized_boundary_flat_mean_errors": flat_errors,
        }
    )
    return report


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_two_cumulant_lod",
    "load_contract",
]
