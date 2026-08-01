"""Unchanged AY2 confirmation for the BN4 triangular logit transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_adaptive_lut_basis_development import (
    _canonical_bytes,
    _load_hashed_json,
    _sha256,
)
from src.eval.fivek_hard_case_medoid_development import (
    _load_ay0_population,
    _load_fresh_population,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_triangular_logit_transport_development import (
    METHODS,
    _build_predictions,
    _evaluate_population,
    _fit_population,
    validate_contract as validate_bn4_contract,
)


class FiveKTriangularConfirmationError(ValueError):
    """Raised when the BN5 confirmation contract or evidence drifts."""


def _identity_sets(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    pair_ids: set[str] = set()
    hashes: set[str] = set()
    for row in manifest.get("rows", []):
        pair_id = row.get("pair_id")
        if isinstance(pair_id, str):
            pair_ids.add(pair_id)
        for key, value in row.items():
            if key.lower().endswith("sha256") and isinstance(value, str):
                hashes.add(value.lower())
    return pair_ids, hashes


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKTriangularConfirmationError("contract is not frozen")
    if (
        config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or not config["leakage_controls"].get(
            "target_pixels_allowed_for_oracle_only"
        )
        or config["leakage_controls"].get(
            "target_pixels_allowed_for_adaptive_prediction"
        )
        or set(config["methods"]) != set(METHODS)
    ):
        raise FiveKTriangularConfirmationError("confirmation boundary drift")

    parent = _load_hashed_json(root, config["parent"], "decision")
    if (
        parent.get("status") != config["parent"]["required_status"]
        or parent.get("report_sha256")
        != config["parent"]["required_report_sha256"]
    ):
        raise FiveKTriangularConfirmationError("BN4 decision drift")
    parent_report = root / str(parent["report"])
    parent_config_path = root / str(parent["config"])
    if (
        _sha256(parent_report) != parent["report_sha256"]
        or _sha256(parent_config_path) != parent["config_sha256"]
    ):
        raise FiveKTriangularConfirmationError("BN4 evidence drift")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_validated = validate_bn4_contract(root, parent_config)

    source = config["confirmation_source"]
    source_config = root / str(source["config"])
    source_report_path = root / str(source["report"])
    manifest_path = root / str(source["manifest"])
    if (
        _sha256(source_config) != source["config_sha256"]
        or _sha256(source_report_path) != source["report_sha256"]
        or _sha256(manifest_path) != source["manifest_sha256"]
    ):
        raise FiveKTriangularConfirmationError("AY2 source drift")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        source_report.get("automatic_pass")
        is not source["required_automatic_pass"]
        or len(manifest.get("rows", [])) != int(source["rows"])
    ):
        raise FiveKTriangularConfirmationError("AY2 source is ineligible")

    confirmation_pairs, confirmation_hashes = _identity_sets(manifest)
    pair_overlap: set[str] = set()
    hash_overlap: set[str] = set()
    for item in parent_validated["populations"]:
        pairs, hashes = _identity_sets(item["manifest_payload"])
        pair_overlap.update(confirmation_pairs & pairs)
        hash_overlap.update(confirmation_hashes & hashes)
    if (
        len(pair_overlap)
        != int(config["leakage_controls"]["required_pair_id_overlap"])
        or len(hash_overlap)
        != int(config["leakage_controls"]["required_exact_sha_overlap"])
    ):
        raise FiveKTriangularConfirmationError("AY2 leakage gate failed")

    development_thresholds = parent_config["evaluation"]
    confirmation_thresholds = config["evaluation"]
    threshold_pairs = {
        "minimum_oracle_mean_improvement_over_identity_each_population": (
            "minimum_oracle_mean_improvement_over_identity"
        ),
        "minimum_adaptive_mean_improvement_over_global_each_population": (
            "minimum_adaptive_mean_improvement_over_global"
        ),
        "minimum_adaptive_win_fraction_over_global_each_population": (
            "minimum_adaptive_win_fraction_over_global"
        ),
        "maximum_adaptive_p95_ratio_to_global_each_population": (
            "maximum_adaptive_p95_ratio_to_global"
        ),
        "maximum_adaptive_worst_ratio_to_global_each_population": (
            "maximum_adaptive_worst_ratio_to_global"
        ),
        "minimum_adaptive_ao6_style_ratio_to_global_each_population": (
            "minimum_adaptive_ao6_style_ratio_to_global"
        ),
        "minimum_adaptive_median_safe_dose_each_population": (
            "minimum_adaptive_median_safe_dose"
        ),
        "minimum_adaptive_worst_safe_dose_each_population": (
            "minimum_adaptive_worst_safe_dose"
        ),
        "maximum_inverse_roundtrip_error": "maximum_inverse_roundtrip_error",
        "maximum_nonpositive_jacobian_count": (
            "maximum_nonpositive_jacobian_count"
        ),
        "maximum_new_boundary_fraction": "maximum_new_boundary_fraction",
        "maximum_operator_out_of_cube_fraction": (
            "maximum_operator_out_of_cube_fraction"
        ),
    }
    if any(
        development_thresholds[left] != confirmation_thresholds[right]
        for left, right in threshold_pairs.items()
    ):
        raise FiveKTriangularConfirmationError("BN4 threshold drift")
    return {
        "parent_config": parent_config,
        "parent_validated": parent_validated,
        "confirmation_manifest": manifest,
        "pair_overlap": sorted(pair_overlap),
        "hash_overlap": sorted(hash_overlap),
    }


def _confirmation_gates(
    population: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    adaptive = population["metrics"]["adaptive_triangular_transport"]
    oracle = population["metrics"]["oracle_triangular_transport"]
    return {
        "oracle_capacity": oracle["mean_improvement_over_identity"]
        >= thresholds["minimum_oracle_mean_improvement_over_identity"],
        "adaptive_mean": adaptive["mean_improvement_over_global"]
        >= thresholds["minimum_adaptive_mean_improvement_over_global"],
        "adaptive_wins": adaptive["win_fraction_over_global"]
        >= thresholds["minimum_adaptive_win_fraction_over_global"],
        "adaptive_p95": adaptive["p95_ratio_to_global"]
        <= thresholds["maximum_adaptive_p95_ratio_to_global"],
        "adaptive_worst": adaptive["worst_ratio_to_global"]
        <= thresholds["maximum_adaptive_worst_ratio_to_global"],
        "adaptive_style": adaptive["style_ratio_to_global"]
        >= thresholds["minimum_adaptive_ao6_style_ratio_to_global"],
        "median_safe_dose": adaptive["median_safe_dose"]
        >= thresholds["minimum_adaptive_median_safe_dose"],
        "worst_safe_dose": adaptive["minimum_safe_dose"]
        >= thresholds["minimum_adaptive_worst_safe_dose"],
        "inverse": max(
            method["maximum_inverse_roundtrip_error"]
            for method in population["metrics"].values()
        )
        <= thresholds["maximum_inverse_roundtrip_error"],
        "jacobian": max(
            method["maximum_nonpositive_jacobian_count"]
            for method in population["metrics"].values()
        )
        <= thresholds["maximum_nonpositive_jacobian_count"],
        "boundary": max(
            method["maximum_new_boundary_fraction"]
            for method in population["metrics"].values()
        )
        <= thresholds["maximum_new_boundary_fraction"],
        "cube": max(
            method["maximum_out_of_cube_fraction"]
            for method in population["metrics"].values()
        )
        <= thresholds["maximum_operator_out_of_cube_fraction"],
    }


def run_confirmation(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    parent_config = validated["parent_config"]
    parent_validated = validated["parent_validated"]
    curve = parent_validated["bj0"]["curve_validated"]
    ay0 = _load_ay0_population(root, curve["ay0_config"], curve["ay0_report"])
    ay0["name"] = parent_config["development_populations"][0]["name"]
    confirmation = _load_fresh_population(
        root,
        curve["ay0_config"],
        validated["confirmation_manifest"],
        group_field=None,
    )
    confirmation["name"] = "ay2_disjoint_content_confirmation"
    operator = parent_config["operator"]
    lower = parent_validated["lower_bounds"]
    upper = parent_validated["upper_bounds"]
    _fit_population(ay0, operator=operator, lower=lower, upper=upper)
    _fit_population(confirmation, operator=operator, lower=lower, upper=upper)
    predictions = _build_predictions(
        ay0, [confirmation], parent_config, lower=lower, upper=upper
    )
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )
    population = _evaluate_population(
        confirmation,
        predictions[confirmation["name"]],
        operator_config=operator,
        renderer=renderer,
    )
    gates = _confirmation_gates(population, config["evaluation"])
    stable = {
        "population": population,
        "gates": gates,
        "leakage": {
            "pair_id_overlap": len(validated["pair_overlap"]),
            "exact_sha_overlap": len(validated["hash_overlap"]),
        },
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
    }


__all__ = [
    "FiveKTriangularConfirmationError",
    "run_confirmation",
    "validate_contract",
]
