"""Evaluate a clean-room strict-interior parallel projection-curve basis."""

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
from src.roll2film.parallel_projection_curves import (
    ParallelProjectionCurveOperator,
    ProjectionCurveError,
    fit_projection_curve_coefficients,
    select_safe_dose,
    validate_projection_directions,
)


class FiveKProjectionCurveError(ValueError):
    """Raised when the BN1 contract or evidence drifts."""


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKProjectionCurveError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    try:
        directions = validate_projection_directions(
            np.asarray(operator["projection_directions"], dtype=np.float64)
        )
    except ProjectionCurveError as exc:
        raise FiveKProjectionCurveError(str(exc)) from exc
    if (
        len(directions) != 16
        or operator.get("control_point_count") != 9
        or operator.get("boundary_epsilon") != 1.0 / 510.0
        or operator.get("strict_interior_safety_factor") != 1.0 - 1.0e-10
        or operator.get("safe_dose_grid_size") != 7
        or operator.get("safe_dose_bisection_iterations") != 24
        or operator.get("hard_output_clipping_allowed")
        or operator.get("post_operator_pixel_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or prediction.get("basis_rank") != 8
        or prediction.get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or set(config["methods"])
        != {
            "identity",
            "global_projection_curves",
            "adaptive_projection_curves",
            "oracle_projection_curves",
        }
    ):
        raise FiveKProjectionCurveError("projection-curve boundary drift")

    source = _load_hashed_json(root, config["source_method"], "decision")
    if source.get("status") != config["source_method"]["required_status"]:
        raise FiveKProjectionCurveError("source-method decision drift")
    for parent in config["parents"]:
        decision = _load_hashed_json(root, parent, "decision")
        required_status = parent.get("required_status")
        required_decision = parent.get("required_decision")
        if required_status is not None and decision.get("status") != required_status:
            raise FiveKProjectionCurveError("parent status drift")
        if (
            required_decision is not None
            and decision.get("decision") != required_decision
        ):
            raise FiveKProjectionCurveError("parent decision drift")

    data_path = root / str(config["data_protocol"]["config"])
    if (
        not data_path.is_file()
        or _sha256(data_path)
        != str(config["data_protocol"]["config_sha256"]).lower()
    ):
        raise FiveKProjectionCurveError("data protocol drift")
    bj0_config = json.loads(data_path.read_text(encoding="utf-8"))
    bj0 = validate_bj0_contract(root, bj0_config)
    populations = []
    for item in config["development_populations"]:
        manifest = _load_hashed_json(root, item, "manifest")
        if len(manifest.get("rows", [])) != int(item["rows"]):
            raise FiveKProjectionCurveError(f"population drift: {item['name']}")
        populations.append({**item, "manifest_payload": manifest})
    return {"directions": directions, "bj0": bj0, "populations": populations}


def _fit_population(
    population: Mapping[str, Any],
    *,
    directions: np.ndarray,
    operator: Mapping[str, Any],
) -> np.ndarray:
    fitted = []
    for row in population["rows"]:
        fitted.append(
            fit_projection_curve_coefficients(
                row["source"],
                row["target"],
                directions=directions,
                control_point_count=int(operator["control_point_count"]),
                boundary_epsilon=float(operator["boundary_epsilon"]),
                sample_stride=int(operator["sample_stride"]),
                target_activation_limit=float(
                    operator["target_activation_limit"]
                ),
                identity_shrinkage=float(operator["identity_shrinkage"]),
                first_difference_smoothness=float(
                    operator["first_difference_smoothness"]
                ),
                coefficient_absolute_limit=float(
                    operator["coefficient_absolute_limit"]
                ),
                strict_interior_safety_factor=float(
                    operator["strict_interior_safety_factor"]
                ),
            )
        )
    result = np.stack(fitted)
    for row, coefficients in zip(population["rows"], result, strict=True):
        row["fitted_projection_curves"] = coefficients
    return result


def _fit_predict_basis(
    train_x: np.ndarray,
    train_coefficients: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    *,
    rank: int,
    alphas: list[float],
    coefficient_limit: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flat = train_coefficients.reshape(len(train_coefficients), -1)
    pca = PCA(n_components=rank, svd_solver="full").fit(flat)
    scores = pca.transform(flat)
    alpha = _select_alpha(train_x, scores, train_groups, alphas)
    scaler = StandardScaler().fit(train_x)
    predicted_scores = Ridge(alpha=alpha).fit(
        scaler.transform(train_x), scores
    ).predict(scaler.transform(test_x))
    predicted = pca.inverse_transform(predicted_scores).reshape(
        (len(test_x),) + train_coefficients.shape[1:]
    )
    predicted = np.clip(predicted, -coefficient_limit, coefficient_limit)
    global_coefficients = np.median(train_coefficients, axis=0)
    return (
        predicted,
        np.repeat(global_coefficients[None, ...], len(test_x), axis=0),
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
) -> dict[str, dict[str, Any]]:
    rank = int(config["basis_prediction"]["basis_rank"])
    alphas = [
        float(value) for value in config["basis_prediction"]["ridge_alphas"]
    ]
    coefficient_limit = float(config["operator"]["coefficient_absolute_limit"])
    train_x = np.stack([row["descriptor"] for row in ay0["rows"]])
    train_y = np.stack(
        [row["fitted_projection_curves"] for row in ay0["rows"]]
    )
    groups = np.asarray([row["group"] for row in ay0["rows"]], dtype=object)
    adaptive = np.empty_like(train_y)
    global_coefficients = np.empty_like(train_y)
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
            coefficient_limit=coefficient_limit,
        )
        adaptive[test] = predicted
        global_coefficients[test] = global_fold
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
            "global": global_coefficients,
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
            coefficient_limit=coefficient_limit,
        )
        result[population["name"]] = {
            "adaptive": predicted,
            "global": global_fold,
            "fit_evidence": [evidence],
        }
    return result


def _safe_operator(
    directions: np.ndarray,
    coefficients: np.ndarray,
    operator: Mapping[str, Any],
) -> tuple[ParallelProjectionCurveOperator, dict[str, float | int]]:
    return select_safe_dose(
        directions=directions,
        coefficients=coefficients,
        boundary_epsilon=float(operator["boundary_epsilon"]),
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(
            operator["minimum_jacobian_determinant"]
        ),
        maximum_jacobian_condition=float(operator["maximum_jacobian_condition"]),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
        strict_interior_safety_factor=float(
            operator["strict_interior_safety_factor"]
        ),
    )


def _evaluate_population(
    population: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    directions: np.ndarray,
    operator_config: Mapping[str, Any],
    renderer: Any,
) -> dict[str, Any]:
    methods = (
        "identity",
        "global_projection_curves",
        "adaptive_projection_curves",
        "oracle_projection_curves",
    )
    errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    doses = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    out_of_cube = {method: [] for method in methods}
    determinants = {method: [] for method in methods}
    nonpositive = {method: [] for method in methods}
    conditions = {method: [] for method in methods}
    rows = []
    zero = np.zeros_like(population["rows"][0]["fitted_projection_curves"])
    cache: dict[str, tuple[ParallelProjectionCurveOperator, dict[str, float | int]]] = {}
    epsilon = float(operator_config["boundary_epsilon"])
    for index, row in enumerate(population["rows"]):
        coefficients_by_method = {
            "identity": zero,
            "global_projection_curves": prediction["global"][index],
            "adaptive_projection_curves": prediction["adaptive"][index],
            "oracle_projection_curves": row["fitted_projection_curves"],
        }
        target_look, _ = renderer(row["target"])
        record = {"pair_id": row["pair_id"], "group": row["group"]}
        for method, coefficients in coefficients_by_method.items():
            coefficient_sha = _array_sha256(coefficients)
            if coefficient_sha not in cache:
                cache[coefficient_sha] = _safe_operator(
                    directions, coefficients, operator_config
                )
            safe_operator, diagnostics = cache[coefficient_sha]
            candidate = safe_operator.apply(row["source"])
            look, _ = renderer(candidate)
            error = _rmse(candidate, row["target"])
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(candidate, look)
            boundary = max(
                _new_boundary_fraction(row["source"], candidate, epsilon),
                _new_boundary_fraction(candidate, look, epsilon),
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
            record[method] = {
                "neutral_rmse": error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "safe_dose": float(safe_operator.dose),
                "minimum_jacobian_determinant": determinants[method][-1],
                "nonpositive_jacobian_count": nonpositive[method][-1],
                "maximum_jacobian_condition": conditions[method][-1],
                "new_boundary_fraction": boundary,
                "out_of_cube_fraction": escaped,
                "coefficient_sha256": coefficient_sha,
            }
        rows.append(record)

    global_error = np.asarray(errors["global_projection_curves"])
    identity_error = np.asarray(errors["identity"])
    global_style = np.asarray(styles["global_projection_curves"])
    metrics = {}
    for method in methods:
        method_error = np.asarray(errors[method])
        metrics[method] = {
            "neutral": _summary(errors[method]),
            "look": _summary(look_errors[method]),
            "mean_improvement_over_global": float(
                (global_error.mean() - method_error.mean())
                / max(global_error.mean(), 1e-12)
            ),
            "win_fraction_over_global": float(np.mean(method_error < global_error)),
            "p95_ratio_to_global": float(
                np.quantile(method_error, 0.95)
                / max(np.quantile(global_error, 0.95), 1e-12)
            ),
            "worst_ratio_to_global": float(
                np.max(method_error) / max(np.max(global_error), 1e-12)
            ),
            "mean_improvement_over_identity": float(
                (identity_error.mean() - method_error.mean())
                / max(identity_error.mean(), 1e-12)
            ),
            "style_ratio_to_global": float(
                np.median(styles[method]) / max(np.median(global_style), 1e-12)
            ),
            "median_safe_dose": float(np.median(doses[method])),
            "minimum_safe_dose": float(np.min(doses[method])),
            "minimum_jacobian_determinant": float(
                np.min(determinants[method])
            ),
            "maximum_nonpositive_jacobian_count": int(
                np.max(nonpositive[method])
            ),
            "maximum_jacobian_condition": float(np.max(conditions[method])),
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
    operator = config["operator"]
    for population in [ay0, *others]:
        _fit_population(
            population,
            directions=validated["directions"],
            operator=operator,
        )
    predictions = _build_predictions(ay0, others, config)
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )
    populations = {
        population["name"]: _evaluate_population(
            population,
            predictions[population["name"]],
            directions=validated["directions"],
            operator_config=operator,
            renderer=renderer,
        )
        for population in [ay0, *others]
    }
    thresholds = config["evaluation"]
    gates = {}
    for name, population in populations.items():
        adaptive = population["metrics"]["adaptive_projection_curves"]
        oracle = population["metrics"]["oracle_projection_curves"]
        gates[name] = {
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
            "jacobian": adaptive["maximum_nonpositive_jacobian_count"]
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
    stable = {"populations": populations, "gates": gates}
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(
            all(population.values()) for population in gates.values()
        ),
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
    "FiveKProjectionCurveError",
    "run_development",
    "validate_contract",
]
