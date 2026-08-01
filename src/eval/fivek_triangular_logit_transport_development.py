"""Evaluate the frozen BN4 triangular logit colour transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.eval.fivek_adaptive_lut_basis_development import (
    _array_sha256,
    _canonical_bytes,
    _load_hashed_json,
    _select_alpha,
    _sha256,
    validate_contract as validate_bj0_contract,
)
from src.eval.fivek_hard_case_medoid_development import (
    _load_ay0_population,
    _load_fresh_population,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_unseen_content_confirmation import (
    _median_delta_e76,
    _new_boundary_fraction,
    _rmse,
    _summary,
)
from src.roll2film.triangular_logit_transport import (
    PARAMETER_COUNT,
    TriangularLogitTransport,
    fit_triangular_logit_transport,
    select_safe_transport,
)


class FiveKTriangularTransportError(ValueError):
    """Raised when the BN4 contract, inputs, or evidence drift."""


METHODS = (
    "identity",
    "global_triangular_transport",
    "adaptive_triangular_transport",
    "oracle_triangular_transport",
)


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKTriangularTransportError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    lower = np.asarray(operator.get("lower_bounds"), dtype=np.float64)
    upper = np.asarray(operator.get("upper_bounds"), dtype=np.float64)
    identity = np.asarray(operator.get("identity"), dtype=np.float64)
    if (
        operator.get("family") != "monotone_triangular_logit_transport"
        or operator.get("parameter_count") != PARAMETER_COUNT
        or lower.shape != (PARAMETER_COUNT,)
        or upper.shape != (PARAMETER_COUNT,)
        or identity.shape != (PARAMETER_COUNT,)
        or not np.array_equal(identity, np.zeros(PARAMETER_COUNT))
        or np.any(lower >= identity)
        or np.any(upper <= identity)
        or operator.get("safe_dose_grid_size") != 9
        or operator.get("safe_dose_bisection_iterations") != 24
        or operator.get("hard_output_clipping_allowed")
        or operator.get("post_operator_pixel_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or operator.get("learned_final_rgb_allowed")
        or prediction.get("basis_rank") != 8
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or set(config["methods"]) != set(METHODS)
    ):
        raise FiveKTriangularTransportError("triangular transport boundary drift")

    parent = _load_hashed_json(root, config["parent"], "decision")
    if parent.get("status") != config["parent"]["required_status"]:
        raise FiveKTriangularTransportError("parent status drift")
    data_path = root / str(config["data_protocol"]["config"])
    if (
        not data_path.is_file()
        or _sha256(data_path)
        != str(config["data_protocol"]["config_sha256"]).lower()
    ):
        raise FiveKTriangularTransportError("data protocol drift")
    bj0_config = json.loads(data_path.read_text(encoding="utf-8"))
    bj0 = validate_bj0_contract(root, bj0_config)
    populations = []
    for item in config["development_populations"]:
        manifest = _load_hashed_json(root, item, "manifest")
        if len(manifest.get("rows", [])) != int(item["rows"]):
            raise FiveKTriangularTransportError(
                f"population drift: {item['name']}"
            )
        populations.append({**item, "manifest_payload": manifest})
    return {
        "bj0": bj0,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "populations": populations,
    }


def _fit_population(
    population: Mapping[str, Any],
    *,
    operator: Mapping[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    fitted = []
    for row in population["rows"]:
        parameters, success = fit_triangular_logit_transport(
            row["source"],
            row["target"],
            lower_bounds=lower,
            upper_bounds=upper,
            sample_stride=int(operator["sample_stride"]),
            identity_shrinkage=float(operator["identity_shrinkage"]),
            maximum_evaluations=int(operator["maximum_fit_evaluations"]),
        )
        if not success:
            raise FiveKTriangularTransportError(
                f"per-image transport fit failed: {row['pair_id']}"
            )
        row["fitted_triangular_transport"] = parameters
        fitted.append(parameters)
    return np.stack(fitted)


def _fit_predict_basis(
    train_x: np.ndarray,
    train_parameters: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    *,
    rank: int,
    alphas: list[float],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pca = PCA(n_components=rank, svd_solver="full").fit(train_parameters)
    scores = pca.transform(train_parameters)
    alpha = _select_alpha(train_x, scores, train_groups, alphas)
    scaler = StandardScaler().fit(train_x)
    predicted_scores = Ridge(alpha=alpha).fit(
        scaler.transform(train_x), scores
    ).predict(scaler.transform(test_x))
    predicted = pca.inverse_transform(predicted_scores)
    predicted = np.clip(predicted, lower, upper)
    global_parameters = np.median(train_parameters, axis=0)
    return (
        predicted,
        np.repeat(global_parameters[None, :], len(test_x), axis=0),
        {
            "alpha": alpha,
            "explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "basis_sha256": _array_sha256(pca.components_),
            "mean_sha256": _array_sha256(pca.mean_),
        },
    )


def _build_predictions(
    ay0: Mapping[str, Any],
    others: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, dict[str, Any]]:
    prediction = config["basis_prediction"]
    rank = int(prediction["basis_rank"])
    alphas = [float(value) for value in prediction["ridge_alphas"]]
    train_x = np.stack([row["descriptor"] for row in ay0["rows"]])
    train_y = np.stack(
        [row["fitted_triangular_transport"] for row in ay0["rows"]]
    )
    groups = np.asarray([row["group"] for row in ay0["rows"]], dtype=object)
    adaptive = np.empty_like(train_y)
    global_parameters = np.empty_like(train_y)
    fold_records = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(train_x, groups=groups), start=1
    ):
        predicted, global_fold, evidence = _fit_predict_basis(
            train_x[train],
            train_y[train],
            groups[train],
            train_x[test],
            rank=rank,
            alphas=alphas,
            lower=lower,
            upper=upper,
        )
        adaptive[test] = predicted
        global_parameters[test] = global_fold
        fold_records.append(
            {
                "fold": fold,
                "held_groups": sorted(set(groups[test].tolist())),
                **evidence,
            }
        )
    result = {
        ay0["name"]: {
            "adaptive": adaptive,
            "global": global_parameters,
            "fit_evidence": fold_records,
        }
    }
    for population in others:
        test_x = np.stack([row["descriptor"] for row in population["rows"]])
        predicted, global_fold, evidence = _fit_predict_basis(
            train_x,
            train_y,
            groups,
            test_x,
            rank=rank,
            alphas=alphas,
            lower=lower,
            upper=upper,
        )
        result[population["name"]] = {
            "adaptive": predicted,
            "global": global_fold,
            "fit_evidence": [evidence],
        }
    return result


def _safe_operator(
    parameters: np.ndarray, operator: Mapping[str, Any]
) -> tuple[TriangularLogitTransport, dict[str, float | int]]:
    return select_safe_transport(
        parameters=parameters,
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(
            operator["minimum_jacobian_determinant"]
        ),
        maximum_jacobian_condition=float(operator["maximum_jacobian_condition"]),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
    )


def _evaluate_population(
    population: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    operator_config: Mapping[str, Any],
    renderer: Any,
) -> dict[str, Any]:
    errors = {method: [] for method in METHODS}
    look_errors = {method: [] for method in METHODS}
    styles = {method: [] for method in METHODS}
    doses = {method: [] for method in METHODS}
    boundaries = {method: [] for method in METHODS}
    out_of_cube = {method: [] for method in METHODS}
    determinants = {method: [] for method in METHODS}
    nonpositive = {method: [] for method in METHODS}
    conditions = {method: [] for method in METHODS}
    inverse_errors = {method: [] for method in METHODS}
    rows = []
    zero = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    cache: dict[
        str, tuple[TriangularLogitTransport, dict[str, float | int]]
    ] = {}
    for index, row in enumerate(population["rows"]):
        parameters_by_method = {
            "identity": zero,
            "global_triangular_transport": prediction["global"][index],
            "adaptive_triangular_transport": prediction["adaptive"][index],
            "oracle_triangular_transport": row["fitted_triangular_transport"],
        }
        target_look, _ = renderer(row["target"])
        record = {"pair_id": row["pair_id"], "group": row["group"]}
        for method, parameters in parameters_by_method.items():
            parameter_sha = _array_sha256(parameters)
            if parameter_sha not in cache:
                cache[parameter_sha] = _safe_operator(
                    parameters, operator_config
                )
            safe_operator, diagnostics = cache[parameter_sha]
            candidate = safe_operator.apply(row["source"])
            look, _ = renderer(candidate)
            error = _rmse(candidate, row["target"])
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(candidate, look)
            boundary = max(
                _new_boundary_fraction(row["source"], candidate, 0.0),
                _new_boundary_fraction(candidate, look, 0.0),
            )
            escaped = float(np.mean((candidate < 0.0) | (candidate > 1.0)))
            errors[method].append(error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            doses[method].append(float(safe_operator.dose))
            boundaries[method].append(boundary)
            out_of_cube[method].append(escaped)
            determinants[method].append(float(diagnostics["minimum_determinant"]))
            nonpositive[method].append(
                int(diagnostics["nonpositive_determinant_count"])
            )
            conditions[method].append(float(diagnostics["maximum_condition"]))
            inverse_errors[method].append(
                float(diagnostics["maximum_inverse_roundtrip_error"])
            )
            record[method] = {
                "neutral_rmse": error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "safe_dose": float(safe_operator.dose),
                "minimum_jacobian_determinant": determinants[method][-1],
                "nonpositive_jacobian_count": nonpositive[method][-1],
                "maximum_jacobian_condition": conditions[method][-1],
                "maximum_inverse_roundtrip_error": inverse_errors[method][-1],
                "new_boundary_fraction": boundary,
                "out_of_cube_fraction": escaped,
                "parameter_sha256": parameter_sha,
            }
        rows.append(record)

    global_error = np.asarray(errors["global_triangular_transport"])
    identity_error = np.asarray(errors["identity"])
    global_style = np.asarray(styles["global_triangular_transport"])
    metrics = {}
    for method in METHODS:
        method_error = np.asarray(errors[method])
        metrics[method] = {
            "neutral": _summary(errors[method]),
            "look": _summary(look_errors[method]),
            "mean_improvement_over_global": float(
                (global_error.mean() - method_error.mean())
                / max(global_error.mean(), 1.0e-12)
            ),
            "win_fraction_over_global": float(np.mean(method_error < global_error)),
            "p95_ratio_to_global": float(
                np.quantile(method_error, 0.95)
                / max(np.quantile(global_error, 0.95), 1.0e-12)
            ),
            "worst_ratio_to_global": float(
                np.max(method_error) / max(np.max(global_error), 1.0e-12)
            ),
            "mean_improvement_over_identity": float(
                (identity_error.mean() - method_error.mean())
                / max(identity_error.mean(), 1.0e-12)
            ),
            "style_ratio_to_global": float(
                np.median(styles[method])
                / max(np.median(global_style), 1.0e-12)
            ),
            "median_safe_dose": float(np.median(doses[method])),
            "minimum_safe_dose": float(np.min(doses[method])),
            "minimum_jacobian_determinant": float(np.min(determinants[method])),
            "maximum_nonpositive_jacobian_count": int(np.max(nonpositive[method])),
            "maximum_jacobian_condition": float(np.max(conditions[method])),
            "maximum_inverse_roundtrip_error": float(np.max(inverse_errors[method])),
            "maximum_new_boundary_fraction": float(np.max(boundaries[method])),
            "maximum_out_of_cube_fraction": float(np.max(out_of_cube[method])),
        }
    return {
        "name": population["name"],
        "row_count": len(rows),
        "group_count": len(set(row["group"] for row in population["rows"])),
        "fit_evidence": prediction["fit_evidence"],
        "metrics": metrics,
        "rows": rows,
    }


def _population_gates(
    population: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    adaptive = population["metrics"]["adaptive_triangular_transport"]
    oracle = population["metrics"]["oracle_triangular_transport"]
    all_metrics = population["metrics"].values()
    return {
        "oracle_capacity": oracle["mean_improvement_over_identity"]
        >= thresholds[
            "minimum_oracle_mean_improvement_over_identity_each_population"
        ],
        "adaptive_mean": adaptive["mean_improvement_over_global"]
        >= thresholds[
            "minimum_adaptive_mean_improvement_over_global_each_population"
        ],
        "adaptive_wins": adaptive["win_fraction_over_global"]
        >= thresholds[
            "minimum_adaptive_win_fraction_over_global_each_population"
        ],
        "adaptive_p95": adaptive["p95_ratio_to_global"]
        <= thresholds["maximum_adaptive_p95_ratio_to_global_each_population"],
        "adaptive_worst": adaptive["worst_ratio_to_global"]
        <= thresholds["maximum_adaptive_worst_ratio_to_global_each_population"],
        "adaptive_style": adaptive["style_ratio_to_global"]
        >= thresholds[
            "minimum_adaptive_ao6_style_ratio_to_global_each_population"
        ],
        "median_safe_dose": adaptive["median_safe_dose"]
        >= thresholds["minimum_adaptive_median_safe_dose_each_population"],
        "worst_safe_dose": adaptive["minimum_safe_dose"]
        >= thresholds["minimum_adaptive_worst_safe_dose_each_population"],
        "inverse": max(metric["maximum_inverse_roundtrip_error"] for metric in all_metrics)
        <= thresholds["maximum_inverse_roundtrip_error"],
        "jacobian": max(
            metric["maximum_nonpositive_jacobian_count"]
            for metric in population["metrics"].values()
        )
        <= thresholds["maximum_nonpositive_jacobian_count"],
        "boundary": max(
            metric["maximum_new_boundary_fraction"]
            for metric in population["metrics"].values()
        )
        <= thresholds["maximum_new_boundary_fraction"],
        "cube": max(
            metric["maximum_out_of_cube_fraction"]
            for metric in population["metrics"].values()
        )
        <= thresholds["maximum_operator_out_of_cube_fraction"],
    }


def run_development(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    curve = validated["bj0"]["curve_validated"]
    ay0 = _load_ay0_population(root, curve["ay0_config"], curve["ay0_report"])
    ay0["name"] = config["development_populations"][0]["name"]
    others = []
    for item in validated["populations"][1:]:
        population = _load_fresh_population(
            root, curve["ay0_config"], item["manifest_payload"]
        )
        population["name"] = item["name"]
        others.append(population)
    for population in [ay0, *others]:
        _fit_population(
            population,
            operator=config["operator"],
            lower=validated["lower_bounds"],
            upper=validated["upper_bounds"],
        )
    predictions = _build_predictions(
        ay0,
        others,
        config,
        lower=validated["lower_bounds"],
        upper=validated["upper_bounds"],
    )
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )
    populations = {
        population["name"]: _evaluate_population(
            population,
            predictions[population["name"]],
            operator_config=config["operator"],
            renderer=renderer,
        )
        for population in [ay0, *others]
    }
    gates = {
        name: _population_gates(population, config["evaluation"])
        for name, population in populations.items()
    }
    stable = {"populations": populations, "gates": gates}
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(all(values.values()) for values in gates.values()),
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
    "FiveKTriangularTransportError",
    "run_development",
    "validate_contract",
]
