"""Intrinsically cube-preserving adaptive explicit LUT-basis experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_adaptive_lut_basis_development import (
    FiveKAdaptiveLUTError,
    _array_sha256,
    _build_predictions,
    _canonical_bytes,
    _load_hashed_json,
    _sha256,
    _smoothness_matrix,
    _trilinear_features,
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


class FiveKCubeLUTError(ValueError):
    """Raised when the frozen cube-preserving LUT contract drifts."""


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKCubeLUTError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    if (
        operator.get("grid_size") != 4
        or operator.get("channel_envelope") != "4*x_c*(1-x_c)"
        or operator.get("node_coefficient_minimum") != -0.25
        or operator.get("node_coefficient_maximum") != 0.25
        or operator.get("hard_output_clipping_allowed")
        or operator.get("post_operator_gamut_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or prediction.get("basis_rank") != 8
        or prediction.get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise FiveKCubeLUTError("cube-preserving boundary drift")
    try:
        parent = _load_hashed_json(root, config["parent"], "decision")
    except FiveKAdaptiveLUTError as exc:
        raise FiveKCubeLUTError(str(exc)) from exc
    if parent.get("status") != config["parent"]["required_status"]:
        raise FiveKCubeLUTError("parent decision drift")
    populations = []
    for item in config["development_populations"]:
        try:
            manifest = _load_hashed_json(root, item, "manifest")
        except FiveKAdaptiveLUTError as exc:
            raise FiveKCubeLUTError(str(exc)) from exc
        if len(manifest.get("rows", [])) != int(item["rows"]):
            raise FiveKCubeLUTError(f"row drift: {item['name']}")
        populations.append({**item, "manifest_payload": manifest})
    return {"populations": populations}


def apply_cube_preserving_lut(
    source: np.ndarray, coefficient_lut: np.ndarray
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    lut = np.asarray(coefficient_lut, dtype=np.float64)
    if (
        source_array.ndim != 3
        or source_array.shape[2] != 3
        or not np.all(np.isfinite(source_array))
        or np.any(source_array < 0.0)
        or np.any(source_array > 1.0)
        or lut.shape != (4, 4, 4, 3)
        or not np.all(np.isfinite(lut))
        or np.any(lut < -0.25)
        or np.any(lut > 0.25)
    ):
        raise FiveKCubeLUTError("invalid cube-preserving LUT peers")
    coefficients = _trilinear_features(source_array, 4) @ lut.reshape(-1, 3)
    coefficients = coefficients.reshape(source_array.shape)
    output = source_array + (
        4.0 * source_array * (1.0 - source_array) * coefficients
    )
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise FiveKCubeLUTError("cube-preserving proof failed")
    return output


def fit_cube_preserving_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    grid_size: int,
    sample_stride: int,
    identity_shrinkage: float,
    smoothness: float,
    coefficient_minimum: float,
    coefficient_maximum: float,
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 3
        or source_array.shape[2] != 3
        or grid_size != 4
        or sample_stride < 1
        or coefficient_minimum != -0.25
        or coefficient_maximum != 0.25
    ):
        raise FiveKCubeLUTError("invalid cube-preserving fit peers")
    sampled_source = source_array[::sample_stride, ::sample_stride]
    sampled_target = target_array[::sample_stride, ::sample_stride]
    features = _trilinear_features(sampled_source, grid_size)
    residual = (sampled_target - sampled_source).reshape(-1, 3)
    envelope = (
        4.0 * sampled_source * (1.0 - sampled_source)
    ).reshape(-1, 3)
    differences = _smoothness_matrix(grid_size)
    regularizer = (
        identity_shrinkage * np.eye(grid_size**3)
        + smoothness * (differences.T @ differences)
    )
    nodes = np.empty((grid_size**3, 3), dtype=np.float64)
    for channel in range(3):
        design = features * envelope[:, channel, None]
        nodes[:, channel] = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ residual[:, channel],
        )
    return np.clip(
        nodes.reshape(grid_size, grid_size, grid_size, 3),
        coefficient_minimum,
        coefficient_maximum,
    )


def _prepare_luts(
    population: Mapping[str, Any],
    operator: Mapping[str, Any],
    *,
    fit_function: Any = fit_cube_preserving_lut,
) -> None:
    for row in population["rows"]:
        row["fitted_lut"] = fit_function(
            row["source"],
            row["target"],
            grid_size=int(operator["grid_size"]),
            sample_stride=int(operator["fit_sample_stride"]),
            identity_shrinkage=float(operator["identity_shrinkage"]),
            smoothness=float(operator["first_difference_smoothness"]),
            coefficient_minimum=float(operator["node_coefficient_minimum"]),
            coefficient_maximum=float(operator["node_coefficient_maximum"]),
        )


def _evaluate(
    population: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    renderer: Any,
    epsilon: float,
    apply_function: Any = apply_cube_preserving_lut,
) -> dict[str, Any]:
    methods = (
        "identity",
        "global_lut",
        "adaptive_basis_lut",
        "oracle_fitted_lut",
    )
    errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    out_of_cube = {method: [] for method in methods}
    rows = []
    zero = np.zeros((4, 4, 4, 3), dtype=np.float64)
    for index, row in enumerate(population["rows"]):
        lut_map = {
            "identity": zero,
            "global_lut": prediction["global"][index],
            "adaptive_basis_lut": prediction["adaptive"][index],
            "oracle_fitted_lut": row["fitted_lut"],
        }
        target_look, _ = renderer(row["target"])
        record = {"pair_id": row["pair_id"], "group": row["group"]}
        for method, lut in lut_map.items():
            candidate = apply_function(row["source"], lut)
            look, _ = renderer(candidate)
            error = _rmse(candidate, row["target"])
            errors[method].append(error)
            look_errors[method].append(_rmse(look, target_look))
            styles[method].append(_median_delta_e76(candidate, look))
            boundaries[method].append(
                max(
                    _new_boundary_fraction(
                        row["source"], candidate, epsilon
                    ),
                    _new_boundary_fraction(candidate, look, epsilon),
                )
            )
            out_of_cube[method].append(
                float(np.mean((candidate < 0.0) | (candidate > 1.0)))
            )
            record[method] = {
                "neutral_rmse": error,
                "look_rmse": look_errors[method][-1],
                "style_delta_e76": styles[method][-1],
                "new_boundary_fraction": boundaries[method][-1],
                "out_of_cube_fraction": out_of_cube[method][-1],
                "lut_sha256": _array_sha256(lut),
            }
        rows.append(record)
    global_error = np.asarray(errors["global_lut"])
    identity_error = np.asarray(errors["identity"])
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
            "win_fraction_over_global": float(
                np.mean(method_error < global_error)
            ),
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
                np.median(styles[method])
                / max(np.median(styles["global_lut"]), 1e-12)
            ),
            "maximum_new_boundary_fraction": float(
                np.max(boundaries[method])
            ),
            "maximum_out_of_cube_fraction": float(
                np.max(out_of_cube[method])
            ),
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
    parent_decision = _load_hashed_json(root, config["parent"], "decision")
    parent_report = json.loads(
        (root / parent_decision["report"]).read_text(encoding="utf-8")
    )
    parent_config_path = root / parent_decision["config"]
    if _sha256(parent_config_path) != parent_report["config_sha256"]:
        raise FiveKCubeLUTError("BJ0 parent config drift")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    bj0 = validate_bj0_contract(root, parent_config)
    return run_validated_development(
        root=root,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        software_commit=software_commit,
        validated=validated,
        bj0=bj0,
        fit_function=fit_cube_preserving_lut,
        apply_function=apply_cube_preserving_lut,
    )


def run_validated_development(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
    validated: Mapping[str, Any],
    bj0: Mapping[str, Any],
    fit_function: Any,
    apply_function: Any,
) -> dict[str, Any]:
    """Run a validated bounded-LUT representation through the fixed protocol."""

    curve = bj0["curve_validated"]
    ay0 = _load_ay0_population(
        root, curve["ay0_config"], curve["ay0_report"]
    )
    ay0["name"] = config["development_populations"][0]["name"]
    others = []
    for item in validated["populations"][1:]:
        population = _load_fresh_population(
            root, curve["ay0_config"], item["manifest_payload"]
        )
        population["name"] = item["name"]
        others.append(population)
    for population in [ay0, *others]:
        _prepare_luts(
            population,
            config["operator"],
            fit_function=fit_function,
        )
    predictions = _build_predictions(ay0, others, config)
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )
    epsilon = float(
        bj0["curve_config"]["parents"]["ay3"]["boundary_epsilon"]
    )
    populations = {
        population["name"]: _evaluate(
            population,
            predictions[population["name"]],
            renderer=renderer,
            epsilon=epsilon,
            apply_function=apply_function,
        )
        for population in [ay0, *others]
    }
    thresholds = config["evaluation"]
    gates = {}
    for name, population in populations.items():
        adaptive = population["metrics"]["adaptive_basis_lut"]
        oracle = population["metrics"]["oracle_fitted_lut"]
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
            <= thresholds[
                "maximum_adaptive_p95_ratio_to_global_each_population"
            ],
            "adaptive_worst": adaptive["worst_ratio_to_global"]
            <= thresholds[
                "maximum_adaptive_worst_ratio_to_global_each_population"
            ],
            "adaptive_style": adaptive["style_ratio_to_global"]
            >= thresholds[
                "minimum_adaptive_ao6_style_ratio_to_global_each_population"
            ],
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
    "FiveKCubeLUTError",
    "apply_cube_preserving_lut",
    "fit_cube_preserving_lut",
    "run_development",
    "run_validated_development",
    "validate_contract",
]
