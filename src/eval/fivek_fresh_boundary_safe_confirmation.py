"""Fresh confirmation of the bounded neutral predictor and safe executor."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_neutral_base,
    validate_contract as validate_safe_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_neutral_base_parameter_pilot import (
    fit_neutral_base_parameters,
    source_descriptor,
)
from src.eval.fivek_unseen_content_confirmation import (
    _bootstrap_lower,
    _median_delta_e76,
    _new_boundary_fraction,
    _rmse,
    _summary,
    load_srgb16,
)
from scripts.build_fivek_freeze_pack import load_raw_default


class FiveKFreshBoundarySafeError(ValueError):
    """Raised when fresh confirmation inputs or boundaries drift."""


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
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKFreshBoundarySafeError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKFreshBoundarySafeError("contract is not frozen")
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
    ):
        raise FiveKFreshBoundarySafeError("confirmatory boundary drift")
    development = config["development"]
    development_config = _load_hashed_json(
        root, development["config"], development["config_sha256"]
    )
    development_report = _load_hashed_json(
        root, development["report"], development["report_sha256"]
    )
    safe = config["safe_executor"]
    safe_config = _load_hashed_json(
        root, safe["config"], safe["config_sha256"]
    )
    safe_decision = _load_hashed_json(
        root, safe["decision"], safe["decision_sha256"]
    )
    safe_validated = validate_safe_contract(root, safe_config)
    confirmation = config["confirmation"]
    manifest = _load_hashed_json(
        root, confirmation["manifest"], confirmation["manifest_sha256"]
    )
    report = _load_hashed_json(
        root, confirmation["report"], confirmation["report_sha256"]
    )
    decision = _load_hashed_json(
        root, confirmation["decision"], confirmation["decision_sha256"]
    )
    if (
        len(manifest.get("rows", [])) != confirmation["required_rows"]
        or report.get("automatic_pass") is not True
        or decision.get("status") != "pass_fresh_confirmation_ready"
        or safe_decision.get("status")
        != "development_pass_fresh_confirmation_required"
        or safe_decision.get(
            "maximum_neutral_or_final_ao6_new_boundary_fraction"
        )
        != 0.0
        or development.get(
            "feature_parameter_alpha_or_capacity_change_allowed"
        )
        is not False
        or safe.get("change_allowed") is not False
    ):
        raise FiveKFreshBoundarySafeError("eligibility boundary drift")
    return {
        "development_config": development_config,
        "development_report": development_report,
        "manifest": manifest,
        "safe_config": safe_config,
        "safe_validated": safe_validated,
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
    development_rows = validated["development_report"]["rows"]
    source_manifest = json.loads(
        (
            root
            / development_config["source_evidence"]["manifest"]
        ).read_text(encoding="utf-8")
    )
    source_by_id = {
        row["pair_id"]: row for row in source_manifest["rows"]
    }
    maximum_side = int(config["confirmation"]["maximum_side"])
    descriptors: list[np.ndarray] = []
    parameters: list[list[float]] = []
    for row in development_rows:
        source_row = source_by_id[row["pair_id"]]
        source, _ = load_raw_default(
            root / source_row["raw_path"], maximum_side
        )
        descriptors.append(
            source_descriptor(
                np.clip(source, 0.0, 1.0), development_config
            )
        )
        parameters.append(row["fitted_parameters"])
    descriptor_array = np.stack(descriptors)
    parameter_array = np.asarray(parameters, dtype=np.float64)
    scaler = StandardScaler().fit(descriptor_array)
    model = Ridge(alpha=float(config["development"]["ridge_alpha"])).fit(
        scaler.transform(descriptor_array), parameter_array
    )
    global_parameters = np.median(parameter_array, axis=0)
    lower = np.asarray(
        development_config["operator"]["lower_bounds"], dtype=np.float64
    )
    upper = np.asarray(
        development_config["operator"]["upper_bounds"], dtype=np.float64
    )
    renderer = build_fixed_ao6_renderer(
        validated["safe_validated"]["fixed_config"],
        validated["safe_validated"]["fixed_validated"],
    )
    methods = [str(value) for value in config["methods"]]
    neutral_errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    limited = {method: [] for method in methods}
    fit_success: list[bool] = []
    rows: list[dict[str, Any]] = []
    epsilon = float(config["safe_executor"]["boundary_epsilon"])
    for evidence in validated["manifest"]["rows"]:
        source_path = Path(evidence["source_path"])
        target_path = Path(evidence["target_path"])
        if (
            _sha256(source_path) != evidence["source_sha256"]
            or _sha256(target_path) != evidence["target_sha256"]
        ):
            raise FiveKFreshBoundarySafeError(
                f"fresh asset drift: {evidence['pair_id']}"
            )
        source = load_srgb16(source_path, maximum_side)
        target = load_srgb16(target_path, maximum_side)
        fitted, success = fit_neutral_base_parameters(
            source, target, development_config
        )
        fit_success.append(success)
        ridge_parameters = np.clip(
            model.predict(
                scaler.transform(
                    source_descriptor(
                        source, development_config
                    )[None, :]
                )
            )[0],
            lower,
            upper,
        )
        parameter_map = {
            "identity": np.asarray(
                development_config["operator"]["identity"],
                dtype=np.float64,
            ),
            "global": global_parameters,
            "ridge": ridge_parameters,
            "oracle": fitted,
        }
        target_look, _ = renderer(target)
        record: dict[str, Any] = {
            "pair_id": evidence["pair_id"],
            "camera_model": evidence["camera_model"],
            "fit_success": success,
            "target_style_delta_e76": _median_delta_e76(
                target, target_look
            ),
        }
        for method in methods:
            neutral, scale = apply_boundary_safe_neutral_base(
                source,
                parameter_map[method],
                boundary_epsilon=epsilon,
            )
            look, diagnostics = renderer(neutral)
            neutral_error = _rmse(neutral, target)
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(neutral, look)
            neutral_boundary = _new_boundary_fraction(
                source, neutral, epsilon
            )
            look_boundary = _new_boundary_fraction(
                neutral, look, epsilon
            )
            boundary = max(neutral_boundary, look_boundary)
            limited_fraction = float(np.mean(scale < 1.0 - 1e-12))
            neutral_errors[method].append(neutral_error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            boundaries[method].append(boundary)
            limited[method].append(limited_fraction)
            record[method] = {
                "parameters": parameter_map[method].tolist(),
                "neutral_rmse": neutral_error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "limited_fraction": limited_fraction,
                "minimum_scale": float(np.min(scale)),
                "neutral_new_boundary_fraction": neutral_boundary,
                "look_new_boundary_fraction": look_boundary,
                "new_boundary_fraction": boundary,
                **diagnostics,
            }
        rows.append(record)
    metrics = {
        method: {
            "neutral": _summary(neutral_errors[method]),
            "look": _summary(look_errors[method]),
            "median_style_delta_e76": float(
                np.median(styles[method])
            ),
            "median_limited_fraction": float(
                np.median(limited[method])
            ),
            "p95_limited_fraction": float(
                np.quantile(limited[method], 0.95)
            ),
            "maximum_new_boundary_fraction": float(
                max(boundaries[method])
            ),
        }
        for method in methods
    }
    neutral_global = np.asarray(neutral_errors["global"])
    neutral_ridge = np.asarray(neutral_errors["ridge"])
    look_global = np.asarray(look_errors["global"])
    look_ridge = np.asarray(look_errors["ridge"])
    identity = np.asarray(neutral_errors["identity"])
    oracle = np.asarray(neutral_errors["oracle"])
    evaluation = config["evaluation"]
    target_style = float(
        np.median([row["target_style_delta_e76"] for row in rows])
    )
    observed = {
        "fit_success_fraction": float(np.mean(fit_success)),
        "oracle_median_improvement_over_identity": float(
            np.median(
                (identity - oracle) / np.maximum(identity, 1e-12)
            )
        ),
        "neutral_ridge_mean_improvement_over_global": float(
            (neutral_global.mean() - neutral_ridge.mean())
            / max(neutral_global.mean(), 1e-12)
        ),
        "neutral_ridge_win_fraction_over_global": float(
            np.mean(neutral_ridge < neutral_global)
        ),
        "neutral_ridge_p95_ratio_to_global": float(
            np.quantile(neutral_ridge, 0.95)
            / max(np.quantile(neutral_global, 0.95), 1e-12)
        ),
        "neutral_ridge_bootstrap_improvement_lower": _bootstrap_lower(
            neutral_global,
            neutral_ridge,
            seed=int(evaluation["bootstrap_seed"]),
            resamples=int(evaluation["bootstrap_resamples"]),
        ),
        "look_ridge_mean_improvement_over_global": float(
            (look_global.mean() - look_ridge.mean())
            / max(look_global.mean(), 1e-12)
        ),
        "look_ridge_win_fraction_over_global": float(
            np.mean(look_ridge < look_global)
        ),
        "look_ridge_p95_ratio_to_global": float(
            np.quantile(look_ridge, 0.95)
            / max(np.quantile(look_global, 0.95), 1e-12)
        ),
        "look_ridge_bootstrap_improvement_lower": _bootstrap_lower(
            look_global,
            look_ridge,
            seed=int(evaluation["bootstrap_seed"]) + 1,
            resamples=int(evaluation["bootstrap_resamples"]),
        ),
        "target_median_style_delta_e76": target_style,
        "ridge_to_target_median_style_ratio": float(
            metrics["ridge"]["median_style_delta_e76"]
            / max(target_style, 1e-12)
        ),
        "ridge_median_limited_fraction": metrics["ridge"][
            "median_limited_fraction"
        ],
        "ridge_p95_limited_fraction": metrics["ridge"][
            "p95_limited_fraction"
        ],
        "maximum_new_boundary_fraction": float(
            max(
                metrics[method]["maximum_new_boundary_fraction"]
                for method in methods
            )
        ),
    }
    camera_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        camera_rows[row["camera_model"]].append(index)
    camera_strata = {
        camera: {
            "rows": len(indices),
            "neutral_ridge_mean_improvement_over_global": float(
                (
                    neutral_global[indices].mean()
                    - neutral_ridge[indices].mean()
                )
                / max(neutral_global[indices].mean(), 1e-12)
            ),
            "neutral_ridge_win_fraction_over_global": float(
                np.mean(
                    neutral_ridge[indices] < neutral_global[indices]
                )
            ),
        }
        for camera, indices in sorted(camera_rows.items())
    }
    gates = {
        "fit_success": observed["fit_success_fraction"]
        == evaluation["require_fit_success_fraction"],
        "oracle_capacity": observed[
            "oracle_median_improvement_over_identity"
        ]
        >= evaluation["minimum_oracle_median_improvement_over_identity"],
        "neutral_mean": observed[
            "neutral_ridge_mean_improvement_over_global"
        ]
        >= evaluation["minimum_neutral_ridge_mean_improvement_over_global"],
        "neutral_wins": observed[
            "neutral_ridge_win_fraction_over_global"
        ]
        >= evaluation["minimum_neutral_ridge_win_fraction_over_global"],
        "neutral_tail": observed[
            "neutral_ridge_p95_ratio_to_global"
        ]
        <= evaluation["maximum_neutral_ridge_p95_ratio_to_global"],
        "neutral_bootstrap": observed[
            "neutral_ridge_bootstrap_improvement_lower"
        ]
        > evaluation[
            "minimum_neutral_ridge_bootstrap_improvement_lower"
        ],
        "look_mean": observed[
            "look_ridge_mean_improvement_over_global"
        ]
        >= evaluation["minimum_look_ridge_mean_improvement_over_global"],
        "look_wins": observed["look_ridge_win_fraction_over_global"]
        >= evaluation["minimum_look_ridge_win_fraction_over_global"],
        "look_tail": observed["look_ridge_p95_ratio_to_global"]
        <= evaluation["maximum_look_ridge_p95_ratio_to_global"],
        "look_bootstrap": observed[
            "look_ridge_bootstrap_improvement_lower"
        ]
        > evaluation["minimum_look_ridge_bootstrap_improvement_lower"],
        "target_style": observed["target_median_style_delta_e76"]
        >= evaluation["minimum_target_median_style_delta_e76"],
        "ridge_style": observed["ridge_to_target_median_style_ratio"]
        >= evaluation["minimum_ridge_to_target_median_style_ratio"],
        "median_limited": observed["ridge_median_limited_fraction"]
        <= evaluation["maximum_ridge_median_limited_fraction"],
        "p95_limited": observed["ridge_p95_limited_fraction"]
        <= evaluation["maximum_ridge_p95_limited_fraction"],
        "new_boundaries": observed["maximum_new_boundary_fraction"]
        <= evaluation["maximum_new_boundary_fraction"],
    }
    stable = {
        "metrics": metrics,
        "observed": observed,
        "camera_strata": camera_strata,
        "gates": gates,
        "rows": rows,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        "confirmation_manifest_sha256": config["confirmation"][
            "manifest_sha256"
        ],
        "row_count": len(rows),
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
    "FiveKFreshBoundarySafeError",
    "run_confirmation",
    "validate_contract",
]
