"""Independent confirmation of the frozen AY6 monotone curve predictor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_hard_case_medoid_development import (
    _load_ay0_population,
    _load_fresh_population,
)
from src.eval.fivek_monotone_channel_curve_development import (
    _evaluate_population,
    _prepare_parameters,
    validate_contract as validate_development_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)


class FiveKMonotoneCurveConfirmationError(ValueError):
    """Raised when confirmation evidence or the frozen method drifts."""


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


def _load_hashed_json(
    root: Path,
    path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKMonotoneCurveConfirmationError(
            f"evidence drift: {path}"
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKMonotoneCurveConfirmationError(
            "confirmation contract is not frozen"
        )
    if (
        config.get("confirmation_target_use_for_training_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or config["development"].get("change_allowed")
    ):
        raise FiveKMonotoneCurveConfirmationError(
            "confirmation boundary drift"
        )
    development = config["development"]
    development_config = _load_hashed_json(
        root, development["config"], development["config_sha256"]
    )
    development_decision = _load_hashed_json(
        root, development["decision"], development["decision_sha256"]
    )
    development_report = _load_hashed_json(
        root, development["report"], development["report_sha256"]
    )
    validated_development = validate_development_contract(
        root, development_config
    )
    confirmation = config["confirmation"]
    normalization_config = _load_hashed_json(
        root,
        confirmation["normalization_config"],
        confirmation["normalization_config_sha256"],
    )
    manifest = _load_hashed_json(
        root, confirmation["manifest"], confirmation["manifest_sha256"]
    )
    report = _load_hashed_json(
        root, confirmation["report"], confirmation["report_sha256"]
    )
    decision = _load_hashed_json(
        root, confirmation["decision"], confirmation["decision_sha256"]
    )
    development_alpha = development_report["populations"]["fresh_63"][
        "selected_alphas"
    ][0]["full_training_alpha"]
    if (
        development_decision.get("status")
        != development["required_status"]
        or development_report.get("automatic_pass") is not True
        or float(development_alpha)
        != float(development["expected_full_training_alpha"])
        or report.get("automatic_pass") is not True
        or decision.get("status") != confirmation["required_status"]
        or len(manifest.get("rows", []))
        != confirmation["required_rows"]
        or normalization_config.get("candidate_rendering_allowed")
        is not False
    ):
        raise FiveKMonotoneCurveConfirmationError(
            "confirmation eligibility drift"
        )
    return {
        "development_config": development_config,
        "development_report": development_report,
        "validated_development": validated_development,
        "manifest": manifest,
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
    development_config = validated["development_config"]
    development_evidence = validated["validated_development"]
    ay0 = _load_ay0_population(
        root,
        development_evidence["ay0_config"],
        development_evidence["ay0_report"],
    )
    confirmation = _load_fresh_population(
        root,
        development_evidence["ay0_config"],
        validated["manifest"],
    )
    confirmation["name"] = "third_disjoint_64_confirmation"
    operator = development_config["operator"]
    knots = np.asarray(operator["knot_inputs"], dtype=np.float64)
    endpoint_weight = float(operator["endpoint_anchor_weight"])
    predictions = _prepare_parameters(
        ay0,
        confirmation,
        knots=knots,
        stride=int(operator["fit_sample_stride"]),
        endpoint_weight=endpoint_weight,
        alphas=[
            float(value)
            for value in development_config["prediction"]["ridge_alphas"]
        ],
    )
    confirmation_predictions = predictions[confirmation["name"]]
    selected_alpha = confirmation_predictions["alphas"][0][
        "full_training_alpha"
    ]
    expected_alpha = float(
        config["development"]["expected_full_training_alpha"]
    )
    if float(selected_alpha) != expected_alpha:
        raise FiveKMonotoneCurveConfirmationError(
            "training-only alpha selection drift"
        )
    renderer = build_fixed_ao6_renderer(
        development_evidence["safe_validated"]["fixed_config"],
        development_evidence["safe_validated"]["fixed_validated"],
    )
    population = _evaluate_population(
        confirmation,
        confirmation_predictions,
        knots=knots,
        epsilon=float(
            development_config["parents"]["ay3"]["boundary_epsilon"]
        ),
        endpoint_weight=endpoint_weight,
        renderer=renderer,
    )
    metrics = population["metrics"]
    ridge = metrics["ridge"]
    oracle = metrics["oracle"]
    thresholds = config["evaluation"]
    gates = {
        "oracle_capacity": oracle["mean_improvement_over_identity"]
        >= thresholds["minimum_oracle_mean_improvement_over_identity"],
        "ridge_mean": ridge["mean_improvement_over_global"]
        >= thresholds["minimum_ridge_mean_improvement_over_global"],
        "ridge_wins": ridge["win_fraction_over_global"]
        >= thresholds["minimum_ridge_win_fraction_over_global"],
        "ridge_p95": ridge["p95_ratio_to_global"]
        <= thresholds["maximum_ridge_p95_ratio_to_global"],
        "ridge_worst": ridge["worst_ratio_to_global"]
        <= thresholds["maximum_ridge_worst_ratio_to_global"],
        "ridge_style": ridge["style_ratio_to_global"]
        >= thresholds["minimum_ridge_ao6_style_ratio_to_global"],
        "median_limited": ridge["median_limited_fraction"]
        <= thresholds["maximum_ridge_median_limited_fraction"],
        "p95_limited": ridge["p95_limited_fraction"]
        <= thresholds["maximum_ridge_p95_limited_fraction"],
        "boundary": max(
            method["maximum_new_boundary_fraction"]
            for method in metrics.values()
        )
        <= thresholds["maximum_new_boundary_fraction"],
    }
    stable = {
        "selected_training_alpha": selected_alpha,
        "population": population,
        "gates": gates,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
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
    "FiveKMonotoneCurveConfirmationError",
    "run_confirmation",
    "validate_contract",
]
