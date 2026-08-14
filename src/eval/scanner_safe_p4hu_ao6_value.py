"""U6.P7I automatic value test for scanner-safe P4HU plus fixed AO6."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.eval import p4hu_ao6_value as p7h

CONTRACT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-result.v1"
WORKER_REPORT_SCHEMA = (
    "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-worker-report.v1"
)
FINAL_REPORT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-run-report.v1"
ARMS = (
    "fixed_ao6_colour_only_t15_c35",
    "matched_scanner_only_ao6_t15_c35",
    "p4hu_scanner_safe_physical_only_diagnostic",
    "p4hu_scanner_safe_then_fixed_ao6_t15_c35",
)
RUNTIME_ID = "neuro-film.scanner-safe-residual.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P7I evaluator contract")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P7I evaluator contract")
    if tuple(contract["comparison"]["arms"]) != ARMS:
        raise ValueError("U6.P7I arm identity or order drift")
    if contract["comparison"].get("automatic_gates") != "inherit_exact_u6_p7h":
        raise ValueError("U6.P7I must inherit the exact P7H automatic gates")
    candidate = contract["candidate"]
    if candidate != {
        "scanner_safe_residual_runtime_id": RUNTIME_ID,
        "application_domain": "post-scanner-linear-before-encode-and-ao6",
        "reference_arm_domain": "matched-scanner-source-linear",
        "direction_change_allowed": False,
        "p4hu_or_ao6_refit_allowed": False,
    }:
        raise ValueError("U6.P7I candidate policy drift")
    gates = contract["additional_gates"]
    expected = {
        "maximum_limited_pixel_fraction",
        "minimum_median_scale",
        "maximum_collinearity_error",
    }
    if set(gates) != expected:
        raise ValueError("U6.P7I additional gate inventory drift")
    for key in expected:
        value = gates[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"invalid U6.P7I gate: {key}")
    if float(gates["maximum_limited_pixel_fraction"]) > 1.0:
        raise ValueError("invalid U6.P7I limited-pixel gate")
    if float(gates["minimum_median_scale"]) > 1.0:
        raise ValueError("invalid U6.P7I median-scale gate")


def _load_context(
    root: Path, contract: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    str,
]:
    parents = contract["parents"]
    p7h_contract = p7h._load_bound_json(root, parents["p7h_contract"], "P7H contract")
    p7h._require_parent_decision(root, parents["p7h_evidence"], "P7H evidence")
    p7h._require_parent_decision(root, parents["p6am_evidence"], "P6AM evidence")
    p7h._require_parent_decision(root, parents["p6al_evidence"], "P6AL evidence")
    p7h._validate_contract(p7h_contract)
    p4hu_contract, profile, ao6_payload = p7h._load_parents(root, p7h_contract)
    selected, manifest_sha256 = p7h._load_manifest(root, p7h_contract["source"])
    return (
        p7h_contract,
        p4hu_contract,
        profile,
        ao6_payload,
        selected,
        manifest_sha256,
    )


def _scanner_safe_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    receipts = [row.get("scanner_safe_residual") for row in rows]
    if any(not isinstance(receipt, dict) for receipt in receipts):
        raise RuntimeError("scanner-safe residual receipt is missing")
    typed = [receipt for receipt in receipts if isinstance(receipt, dict)]
    return {
        "receipt_count": len(typed),
        "runtime_ids": sorted({str(item.get("runtime_id")) for item in typed}),
        "maximum_limited_pixel_fraction": max(
            float(item["limited_pixel_fraction"]) for item in typed
        ),
        "minimum_median_scale": min(float(item["median_scale"]) for item in typed),
        "minimum_scale": min(float(item["minimum_scale"]) for item in typed),
        "maximum_collinearity_error": max(
            float(item["maximum_collinearity_error"]) for item in typed
        ),
    }


def _scanner_safe_checks(
    aggregates: dict[str, Any], gates: dict[str, Any], *, expected_rows: int
) -> dict[str, bool]:
    finite = all(
        math.isfinite(float(aggregates[key]))
        for key in (
            "maximum_limited_pixel_fraction",
            "minimum_median_scale",
            "minimum_scale",
            "maximum_collinearity_error",
        )
    )
    return {
        "complete_scanner_safe_receipts": aggregates["receipt_count"] == expected_rows,
        "exact_scanner_safe_runtime": aggregates["runtime_ids"] == [RUNTIME_ID],
        "finite_scanner_safe_receipts": finite,
        "limited_pixel_fraction": finite
        and aggregates["maximum_limited_pixel_fraction"]
        <= float(gates["maximum_limited_pixel_fraction"]),
        "median_scale": finite
        and aggregates["minimum_median_scale"]
        >= float(gates["minimum_median_scale"]),
        "collinearity": finite
        and aggregates["maximum_collinearity_error"]
        <= float(gates["maximum_collinearity_error"]),
    }


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path,
    output_dir: Path,
    build_dir: Path,
) -> dict[str, Any]:
    """Render and automatically gate the exact scanner-safe P4HU combination."""

    _validate_contract(contract)
    (
        p7h_contract,
        p4hu_contract,
        profile_payload,
        ao6_payload,
        selected,
        manifest_sha256,
    ) = _load_context(root, contract)
    components = p7h.reconstruct_bounded_photographic_profile(profile_payload)
    runtime = p7h.build_native_density_stage_runtime(
        root=root,
        build_dir=build_dir,
        candidate=p4hu_contract["candidate"],
        profile_payload=profile_payload,
        components=components,
    )
    parent_candidate = p7h_contract["candidate"]
    scanner = p7h._scanner_profile(parent_candidate)
    base_seeds = tuple(int(value) for value in parent_candidate["layer_field_seeds"])
    stride = int(parent_candidate["field_seed_stride_per_source"])
    transforms = p7h_contract["source"].get("transforms", {})
    rows = []
    for index, source_row in enumerate(selected):
        seeds = tuple(seed + index * stride for seed in base_seeds)
        rows.append(
            p7h.evaluate_scanner_safe_source_arms(
                root=root,
                output_dir=output_dir,
                row=source_row,
                transform=transforms.get(source_row["id"]),
                seeds=seeds,
                runtime=runtime,
                ao6_payload=ao6_payload,
                scanner=scanner,
                gates=p7h_contract["automatic_gates"],
                arm_ids=ARMS,
            )
        )
    arm_aggregates = p7h.aggregate_arm_rows(rows)
    expected_rows = int(p7h_contract["source"]["expected_evaluation_rows"])
    p7h_checks = p7h.automatic_arm_checks(
        arm_aggregates,
        p7h_contract["automatic_gates"],
        expected_rows=expected_rows,
    )
    scanner_safe_aggregates = _scanner_safe_aggregates(rows)
    scanner_safe_checks = _scanner_safe_checks(
        scanner_safe_aggregates,
        contract["additional_gates"],
        expected_rows=expected_rows,
    )
    automatic_pass = all(p7h_checks.values()) and all(scanner_safe_checks.values())
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": p7h._canonical_sha256(contract),
        "parent_p7h_contract_sha256": contract["parents"]["p7h_contract"]["sha256"],
        "manifest_sha256": manifest_sha256,
        "profile_bundle_sha256": profile_payload["bundle_sha256"],
        "ao6_payload_sha256": p7h._canonical_sha256(ao6_payload),
        "native_toolchains": runtime.native_toolchains,
        "scanner_profile": {
            "scanner_mtf_sigma_pixels_rgb": [0.7, 0.7, 0.7],
            "gaussian_truncate": 3.0,
        },
        "arms": list(ARMS),
        "arm_roles": {
            ARMS[0]: "current_incumbent",
            ARMS[1]: "incremental_safety_reference",
            ARMS[2]: "diagnostic_only_never_promotion_eligible",
            ARMS[3]: "promotion_challenger",
        },
        "incremental_gate_reference_arm": ARMS[1],
        "rows": rows,
        "aggregates": arm_aggregates,
        "scanner_safe_aggregates": scanner_safe_aggregates,
        "p7h_gates": p7h_checks,
        "scanner_safe_gates": scanner_safe_checks,
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "decision": (
            contract["decision_if_pass"]
            if automatic_pass
            else contract["decision_if_fail"]
        ),
        "physical_only_promotion_eligible": False,
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": p7h._canonical_sha256(core)}


__all__ = [
    "ARMS",
    "CONTRACT_SCHEMA",
    "FINAL_REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "evaluate",
    "load_contract",
]
