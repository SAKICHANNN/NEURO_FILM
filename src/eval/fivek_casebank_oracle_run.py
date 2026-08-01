"""Hash-bound runner for the frozen FiveK off-diagonal case-bank Oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_casebank_oracle import (
    evaluate_offdiagonal_oracle,
    load_split_population,
)


class FiveKCasebankOracleRunError(ValueError):
    """Raised when parent evidence or the Oracle protocol drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_hashed_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise FiveKCasebankOracleRunError(f"parent evidence drift: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FiveKCasebankOracleRunError("parent evidence must be an object")
    return payload


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKCasebankOracleRunError("Oracle contract is not frozen")
    if (
        config.get("router_training_allowed")
        or config.get("confirmation_target_use_for_selection_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise FiveKCasebankOracleRunError("Oracle boundary drift")
    parent = config["parent_population"]
    manifest = _load_hashed_json(
        root, parent["manifest"], parent["manifest_sha256"]
    )
    normalization_report = _load_hashed_json(
        root, parent["report"], parent["report_sha256"]
    )
    summary = manifest.get("split_summary", {})
    if (
        summary.get("development_rows") != parent["development_rows"]
        or summary.get("confirmation_rows") != parent["confirmation_rows"]
        or summary.get("selection_used_target_or_pixels") is not False
    ):
        raise FiveKCasebankOracleRunError("parent population is not eligible")
    adjudication_spec = config.get("parent_source_adjudication")
    if adjudication_spec is None:
        if normalization_report.get("automatic_pass") is not True:
            raise FiveKCasebankOracleRunError("parent population is not eligible")
    else:
        if (
            adjudication_spec.get("required_automatic_pass") is not True
            or adjudication_spec.get("required_target_pixels_accessed")
            is not False
        ):
            raise FiveKCasebankOracleRunError(
                "source-only adjudication contract drift"
            )
        adjudication = _load_hashed_json(
            root,
            adjudication_spec["report"],
            adjudication_spec["report_sha256"],
        )
        if (
            normalization_report.get("automatic_pass") is not False
            or adjudication.get("automatic_pass") is not True
            or adjudication.get("target_pixels_accessed") is not False
            or adjudication.get("operator_fitting_allowed") is not False
            or adjudication.get("parent_manifest_sha256")
            != parent["manifest_sha256"]
        ):
            raise FiveKCasebankOracleRunError(
                "source-only adjudication is not eligible"
            )
    variants = config["target_variants"]
    if set(variants) != {"aligned_expert", "filtered"}:
        raise FiveKCasebankOracleRunError("target-control inventory drift")
    if config.get("required_pass_variants") != [
        "aligned_expert",
        "filtered",
    ]:
        raise FiveKCasebankOracleRunError("pass-variant policy drift")
    operator = config["operator"]
    lower = operator.get("lower_bounds", [])
    upper = operator.get("upper_bounds", [])
    if (
        operator.get("family") != "monotone_triangular_logit_transport"
        or operator.get("parameter_count") != 14
        or len(lower) != 14
        or len(upper) != 14
        or not all(math.isfinite(float(value)) for value in [*lower, *upper])
        or not all(float(lo) < float(hi) for lo, hi in zip(lower, upper))
        or operator.get("hard_output_clipping_allowed") is not False
        or operator.get("spatial_or_semantic_features_allowed") is not False
        or operator.get("learned_final_rgb_allowed") is not False
    ):
        raise FiveKCasebankOracleRunError("explicit operator contract drift")
    required_gate_names = {
        "minimum_mean_improvement_over_identity",
        "minimum_win_fraction_over_identity",
        "maximum_p95_ratio_to_identity",
        "maximum_worst_ratio_to_identity",
        "minimum_mean_improvement_over_global",
        "minimum_win_fraction_over_global",
        "maximum_p95_ratio_to_global",
        "maximum_worst_ratio_to_global",
        "minimum_mean_improvement_over_strength_oracle",
        "minimum_win_fraction_over_strength_oracle",
        "minimum_mean_improvement_over_random_case",
        "minimum_win_fraction_over_random_case",
        "minimum_bootstrap_lower_improvement",
        "minimum_distinct_selected_cases",
        "maximum_selected_case_share",
    }
    for variant in variants.values():
        evaluation = variant.get("evaluation", {})
        doses = evaluation.get("strength_doses", [])
        gates = evaluation.get("gates", {})
        if (
            variant.get("target_variant") not in {"aligned_expert", "filtered"}
            or not doses
            or doses != sorted(set(doses))
            or doses[0] != 0.0
            or doses[-1] != 1.0
            or int(evaluation.get("bootstrap_repetitions", 0)) < 1000
            or set(gates) != required_gate_names
        ):
            raise FiveKCasebankOracleRunError("evaluation contract drift")
    return {"manifest": manifest}


def run_oracle(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}
    for name in config["required_pass_variants"]:
        variant = config["target_variants"][name]
        maximum_side = int(config["decode"]["maximum_side"])
        development = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=variant["target_variant"],
            maximum_side=maximum_side,
        )
        confirmation = load_split_population(
            validated["manifest"],
            split="confirmation",
            target_variant=variant["target_variant"],
            maximum_side=maximum_side,
        )
        variants[name] = evaluate_offdiagonal_oracle(
            development,
            confirmation,
            {
                "operator": config["operator"],
                "evaluation": variant["evaluation"],
                "router_training_allowed": False,
                "final_rgb_learning_allowed": False,
            },
        )
        del development, confirmation
    stable = {
        "parent_manifest_sha256": config["parent_population"][
            "manifest_sha256"
        ],
        "variants": variants,
        "automatic_pass": all(
            variants[name]["automatic_pass"]
            for name in config["required_pass_variants"]
        ),
    }
    report = {
        "schema": "neuro_film.u5_r2bq1_casebank_oracle_report.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "router_training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKCasebankOracleRunError",
    "run_oracle",
    "validate_contract",
]
