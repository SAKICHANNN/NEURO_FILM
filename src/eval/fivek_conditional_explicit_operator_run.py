"""Hash-bound runner for the FiveK bounded conditional explicit operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_conditional_explicit_operator import (
    evaluate_confirmation,
    evaluate_development,
)


class FiveKConditionalExplicitOperatorRunError(ValueError):
    """Raised when a frozen conditional-operator contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _load_hashed(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if not path.is_file() or _sha256(path) != str(spec["sha256"]):
        raise FiveKConditionalExplicitOperatorRunError("parent evidence drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveKConditionalExplicitOperatorRunError(
            "parent evidence must be an object"
        )
    return value


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_model_selection_allowed") is not False
        or config.get("learned_final_rgb_allowed") is not False
        or config.get("production_integration_allowed") is not False
        or config.get("film_or_stock_claim_allowed") is not False
        or model.get("query_target_used_at_inference") is not False
        or model.get("confirmation_used_for_fit_or_selection") is not False
        or model.get("direct_final_rgb_prediction") is not False
    ):
        raise FiveKConditionalExplicitOperatorRunError(
            "conditional operator boundary drift"
        )
    decision = _load_hashed(root, config["parent_pairwise_decision"])
    oracle = _load_hashed(root, config["parent_oracle"])
    nearest = _load_hashed(root, config["parent_nearest_report"])
    manifest = _load_hashed(root, config["parent_manifest"])
    if (
        decision.get("decision")
        != "close_pairwise_case_ranker_before_confirmation"
        or decision.get("ranker_capacity_rescue_allowed") is not False
        or oracle.get("automatic_pass") is not True
        or oracle.get("stable_evidence_id")
        != config["parent_oracle"]["stable_evidence_id"]
        or nearest.get("selected_family") != "tone_layout"
        or manifest.get("split_summary", {}).get(
            "selection_used_target_or_pixels"
        )
        is not False
    ):
        raise FiveKConditionalExplicitOperatorRunError(
            "ineligible parent branch"
        )
    return {
        "decision": decision,
        "oracle": oracle,
        "nearest": nearest,
        "manifest": manifest,
    }


def run_conditional_operator(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    development_populations: dict[str, list[dict[str, Any]]] = {}
    development_results: dict[str, Any] = {}
    for variant_name in config["required_pass_variants"]:
        target_variant = config["target_variants"][variant_name][
            "target_variant"
        ]
        development = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=target_variant,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        development_populations[variant_name] = development
        development_results[variant_name] = evaluate_development(
            rows=development,
            oracle_report=validated["oracle"]["variants"][variant_name],
            nearest_report=validated["nearest"]["development"][
                "tone_layout"
            ][variant_name],
            descriptor_spec=config["descriptor"],
            model_spec=config["model"],
            operator_config=config["operator"],
            evaluation=config["evaluation"],
            gates=config["development_gates"],
        )
    development_pass = all(
        development_results[name]["automatic_pass"]
        for name in config["required_pass_variants"]
    )
    confirmation_results: dict[str, Any] = {}
    if development_pass:
        for variant_name in config["required_pass_variants"]:
            target_variant = config["target_variants"][variant_name][
                "target_variant"
            ]
            confirmation = load_split_population(
                validated["manifest"],
                split="confirmation",
                target_variant=target_variant,
                maximum_side=int(config["decode"]["maximum_side"]),
            )
            confirmation_results[variant_name] = evaluate_confirmation(
                development_rows=development_populations[variant_name],
                confirmation_rows=confirmation,
                oracle_report=validated["oracle"]["variants"][variant_name],
                nearest_report=validated["nearest"]["confirmation"][
                    variant_name
                ],
                descriptor_spec=config["descriptor"],
                model_spec=config["model"],
                operator_config=config["operator"],
                evaluation=config["evaluation"],
                gates=config["confirmation_gates"],
            )
    stable = {
        "parent_oracle_sha256": config["parent_oracle"]["sha256"],
        "parent_nearest_report_sha256": config["parent_nearest_report"][
            "sha256"
        ],
        "development": development_results,
        "development_pass": development_pass,
        "confirmation_executed": development_pass,
        "confirmation": confirmation_results,
        "automatic_pass": development_pass
        and all(
            confirmation_results[name]["automatic_pass"]
            for name in config["required_pass_variants"]
        ),
    }
    report = {
        "schema": "neuro_film.u5_r2bq3_fivek_conditional_explicit_operator.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "confirmation_model_selection_used_targets": False,
        "learned_final_rgb_used": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKConditionalExplicitOperatorRunError",
    "run_conditional_operator",
    "validate_contract",
]
