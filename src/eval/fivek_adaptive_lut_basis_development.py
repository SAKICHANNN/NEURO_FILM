"""Bounded image-adaptive explicit LUT-basis development experiment."""

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

from src.eval.fivek_hard_case_medoid_development import (
    _load_ay0_population,
    _load_fresh_population,
)
from src.eval.fivek_monotone_channel_curve_development import (
    validate_contract as validate_curve_contract,
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


class FiveKAdaptiveLUTError(ValueError):
    """Raised when the frozen adaptive-LUT contract or evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(
    root: Path, item: Mapping[str, Any], key: str
) -> dict[str, Any]:
    path = root / str(item[key])
    if not path.is_file() or _sha256(path) != str(
        item[f"{key}_sha256"]
    ).lower():
        raise FiveKAdaptiveLUTError(f"evidence drift: {item[key]}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKAdaptiveLUTError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    if (
        operator.get("grid_size") != 4
        or operator.get("hard_output_clipping_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or prediction.get("basis_rank") != 8
        or prediction.get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or set(config["methods"])
        != {
            "identity",
            "global_lut",
            "adaptive_basis_lut",
            "oracle_fitted_lut",
        }
    ):
        raise FiveKAdaptiveLUTError("adaptive-LUT boundary drift")
    curve = _load_hashed_json(
        root, config["parents"]["confirmed_curve"], "decision"
    )
    product = _load_hashed_json(
        root, config["parents"]["closed_curve_product_value"], "decision"
    )
    if (
        curve.get("status")
        != config["parents"]["confirmed_curve"]["required_status"]
        or product.get("status")
        != config["parents"]["closed_curve_product_value"]["required_status"]
    ):
        raise FiveKAdaptiveLUTError("parent decision drift")
    curve_report = _load_hashed_json(root, curve, "report")
    curve_config_path = root / str(curve["config"])
    if (
        not curve_config_path.is_file()
        or _sha256(curve_config_path)
        != str(curve_report["config_sha256"]).lower()
    ):
        raise FiveKAdaptiveLUTError("confirmed curve config drift")
    curve_config = json.loads(curve_config_path.read_text(encoding="utf-8"))
    development_config = _load_hashed_json(
        root, curve_config["development"], "config"
    )
    curve_validated = validate_curve_contract(root, development_config)
    populations = []
    for item in config["development_populations"]:
        manifest = _load_hashed_json(root, item, "manifest")
        rows = manifest.get("rows", [])
        if len(rows) != int(item["rows"]):
            raise FiveKAdaptiveLUTError(
                f"population row drift: {item['name']}"
            )
        populations.append({**item, "manifest_payload": manifest})
    return {
        "populations": populations,
        "curve_config": development_config,
        "curve_validated": curve_validated,
    }


def _trilinear_features(rgb: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if (
        grid_size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise FiveKAdaptiveLUTError("invalid LUT input")
    scaled = values * (grid_size - 1)
    lower = np.minimum(
        np.floor(scaled).astype(np.int64), grid_size - 2
    )
    fraction = scaled - lower
    features = np.zeros(
        (len(values), grid_size**3), dtype=np.float64
    )
    rows = np.arange(len(values))
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                weight = (
                    (fraction[:, 0] if dr else 1.0 - fraction[:, 0])
                    * (fraction[:, 1] if dg else 1.0 - fraction[:, 1])
                    * (fraction[:, 2] if db else 1.0 - fraction[:, 2])
                )
                node = (
                    (lower[:, 0] + dr) * grid_size * grid_size
                    + (lower[:, 1] + dg) * grid_size
                    + lower[:, 2]
                    + db
                )
                features[rows, node] += weight
    return features


def _smoothness_matrix(grid_size: int) -> np.ndarray:
    rows = []
    for r in range(grid_size):
        for g in range(grid_size):
            for b in range(grid_size):
                current = (r * grid_size + g) * grid_size + b
                for axis in range(3):
                    peer = [r, g, b]
                    if peer[axis] + 1 >= grid_size:
                        continue
                    peer[axis] += 1
                    adjacent = (
                        (peer[0] * grid_size + peer[1]) * grid_size
                        + peer[2]
                    )
                    row = np.zeros(grid_size**3, dtype=np.float64)
                    row[current] = 1.0
                    row[adjacent] = -1.0
                    rows.append(row)
    return np.stack(rows)


def fit_residual_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    grid_size: int,
    sample_stride: int,
    identity_shrinkage: float,
    smoothness: float,
    maximum_absolute_residual: float,
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 3
        or source_array.shape[2] != 3
        or sample_stride < 1
        or identity_shrinkage <= 0.0
        or smoothness < 0.0
        or maximum_absolute_residual <= 0.0
    ):
        raise FiveKAdaptiveLUTError("invalid LUT fit peers")
    sampled_source = source_array[::sample_stride, ::sample_stride]
    sampled_target = target_array[::sample_stride, ::sample_stride]
    features = _trilinear_features(sampled_source, grid_size)
    residual = (sampled_target - sampled_source).reshape(-1, 3)
    differences = _smoothness_matrix(grid_size)
    system = (
        features.T @ features
        + identity_shrinkage * np.eye(grid_size**3)
        + smoothness * (differences.T @ differences)
    )
    nodes = np.linalg.solve(system, features.T @ residual)
    return np.clip(
        nodes.reshape(grid_size, grid_size, grid_size, 3),
        -maximum_absolute_residual,
        maximum_absolute_residual,
    )


def apply_residual_lut(source: np.ndarray, residual_lut: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    lut = np.asarray(residual_lut, dtype=np.float64)
    if (
        source_array.ndim != 3
        or source_array.shape[2] != 3
        or lut.ndim != 4
        or lut.shape[:3] != (lut.shape[0],) * 3
        or lut.shape[3] != 3
    ):
        raise FiveKAdaptiveLUTError("invalid residual LUT")
    residual = _trilinear_features(source_array, lut.shape[0]) @ lut.reshape(
        -1, 3
    )
    return source_array + residual.reshape(source_array.shape)


def apply_unbounded_residual_safely(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared analytical scale to a finite, possibly OOG candidate."""

    source = np.asarray(source_rgb, dtype=np.float64)
    candidate = np.asarray(candidate_rgb, dtype=np.float64)
    if (
        source.shape != candidate.shape
        or source.ndim != 3
        or source.shape[2] != 3
        or not 0.0 < boundary_epsilon < 0.5
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(candidate))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise FiveKAdaptiveLUTError("invalid analytical residual peers")
    residual = candidate - source
    source_interior = (source > boundary_epsilon) & (
        source < 1.0 - boundary_epsilon
    )
    lower = np.where(source_interior, boundary_epsilon, 0.0)
    upper = np.where(source_interior, 1.0 - boundary_epsilon, 1.0)
    channel_scale = np.ones_like(source)
    positive = residual > 0.0
    negative = residual < 0.0
    channel_scale[positive] = (
        upper[positive] - source[positive]
    ) / residual[positive]
    channel_scale[negative] = (
        lower[negative] - source[negative]
    ) / residual[negative]
    scale = np.clip(np.min(channel_scale, axis=2), 0.0, 1.0)
    scale = np.where(scale < 1.0, np.nextafter(scale, 0.0), scale)
    source_boundary = (source <= boundary_epsilon) | (
        source >= 1.0 - boundary_epsilon
    )
    for _ in range(16):
        output = source + scale[..., None] * residual
        output_boundary = (output <= boundary_epsilon) | (
            output >= 1.0 - boundary_epsilon
        )
        escaped = (output < 0.0) | (output > 1.0)
        bad = np.any(
            escaped | (output_boundary & ~source_boundary), axis=2
        )
        if not np.any(bad):
            break
        scale[bad] = np.nextafter(scale[bad], 0.0)
    else:
        raise FiveKAdaptiveLUTError(
            "analytical scale did not converge to the safe cube"
        )
    if not np.all(np.isfinite(output)):
        raise FiveKAdaptiveLUTError("analytical output is non-finite")
    return output, scale


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> float:
    splits = min(4, len(np.unique(groups)))
    if splits < 2:
        raise FiveKAdaptiveLUTError("insufficient inner groups")
    scores = []
    for alpha in alphas:
        errors = []
        for train, valid in GroupKFold(n_splits=splits).split(
            x, groups=groups
        ):
            scaler = StandardScaler().fit(x[train])
            predicted = Ridge(alpha=alpha).fit(
                scaler.transform(x[train]), y[train]
            ).predict(scaler.transform(x[valid]))
            errors.extend(
                np.mean((predicted - y[valid]) ** 2, axis=1).tolist()
            )
        scores.append((float(np.mean(errors)), alpha))
    return min(scores)[1]


def _fit_predict_basis(
    train_x: np.ndarray,
    train_luts: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    *,
    rank: int,
    alphas: list[float],
    maximum_absolute_residual: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flat = train_luts.reshape(len(train_luts), -1)
    pca = PCA(n_components=rank, svd_solver="full").fit(flat)
    scores = pca.transform(flat)
    alpha = _select_alpha(train_x, scores, train_groups, alphas)
    scaler = StandardScaler().fit(train_x)
    predicted_scores = Ridge(alpha=alpha).fit(
        scaler.transform(train_x), scores
    ).predict(scaler.transform(test_x))
    predicted = pca.inverse_transform(predicted_scores).reshape(
        (len(test_x),) + train_luts.shape[1:]
    )
    predicted = np.clip(
        predicted,
        -maximum_absolute_residual,
        maximum_absolute_residual,
    )
    global_lut = np.median(train_luts, axis=0)
    return (
        predicted,
        np.repeat(global_lut[None, ...], len(test_x), axis=0),
        {
            "alpha": alpha,
            "explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "basis_sha256": _array_sha256(pca.components_),
            "mean_sha256": _array_sha256(pca.mean_),
        },
    )


def _prepare_population_luts(
    population: Mapping[str, Any], operator: Mapping[str, Any]
) -> np.ndarray:
    fitted = []
    for row in population["rows"]:
        fitted.append(
            fit_residual_lut(
                row["source"],
                row["target"],
                grid_size=int(operator["grid_size"]),
                sample_stride=int(operator["fit_sample_stride"]),
                identity_shrinkage=float(operator["identity_shrinkage"]),
                smoothness=float(operator["first_difference_smoothness"]),
                maximum_absolute_residual=float(
                    operator["maximum_absolute_node_residual"]
                ),
            )
        )
    result = np.stack(fitted)
    for row, lut in zip(population["rows"], result, strict=True):
        row["fitted_lut"] = lut
    return result


def _build_predictions(
    ay0: Mapping[str, Any],
    others: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    rank = int(config["basis_prediction"]["basis_rank"])
    alphas = [
        float(value) for value in config["basis_prediction"]["ridge_alphas"]
    ]
    maximum = float(config["operator"]["maximum_absolute_node_residual"])
    train_x = np.stack([row["descriptor"] for row in ay0["rows"]])
    train_y = np.stack([row["fitted_lut"] for row in ay0["rows"]])
    groups = np.asarray([row["group"] for row in ay0["rows"]], dtype=object)
    adaptive = np.empty_like(train_y)
    global_luts = np.empty_like(train_y)
    fold_records = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(train_x, groups=groups), start=1
    ):
        predicted, global_lut, evidence = _fit_predict_basis(
            train_x[train],
            train_y[train],
            groups[train],
            train_x[test],
            rank=rank,
            alphas=alphas,
            maximum_absolute_residual=maximum,
        )
        adaptive[test] = predicted
        global_luts[test] = global_lut
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
            "global": global_luts,
            "fit_evidence": fold_records,
        }
    }
    for population in others:
        test_x = np.stack([row["descriptor"] for row in population["rows"]])
        predicted, global_lut, evidence = _fit_predict_basis(
            train_x,
            train_y,
            groups,
            test_x,
            rank=rank,
            alphas=alphas,
            maximum_absolute_residual=maximum,
        )
        result[population["name"]] = {
            "adaptive": predicted,
            "global": global_lut,
            "fit_evidence": [evidence],
        }
    return result


def _evaluate_population(
    population: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    renderer: Any,
    epsilon: float,
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
    limited = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    rows = []
    zero = np.zeros_like(population["rows"][0]["fitted_lut"])
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
            raw_candidate = apply_residual_lut(row["source"], lut)
            candidate, scale = apply_unbounded_residual_safely(
                row["source"],
                raw_candidate,
                boundary_epsilon=epsilon,
            )
            look, _ = renderer(candidate)
            error = _rmse(candidate, row["target"])
            errors[method].append(error)
            look_errors[method].append(_rmse(look, target_look))
            styles[method].append(_median_delta_e76(candidate, look))
            limited[method].append(float(np.mean(scale < 1.0 - 1e-12)))
            boundaries[method].append(
                max(
                    _new_boundary_fraction(
                        row["source"], candidate, epsilon
                    ),
                    _new_boundary_fraction(candidate, look, epsilon),
                )
            )
            record[method] = {
                "neutral_rmse": error,
                "look_rmse": look_errors[method][-1],
                "style_delta_e76": styles[method][-1],
                "limited_fraction": limited[method][-1],
                "new_boundary_fraction": boundaries[method][-1],
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
            "median_limited_fraction": float(np.median(limited[method])),
            "p95_limited_fraction": float(
                np.quantile(limited[method], 0.95)
            ),
            "maximum_new_boundary_fraction": float(
                np.max(boundaries[method])
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
    curve = validated["curve_validated"]
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
    operator = config["operator"]
    for population in [ay0, *others]:
        _prepare_population_luts(population, operator)
    predictions = _build_predictions(ay0, others, config)
    renderer = build_fixed_ao6_renderer(
        curve["safe_validated"]["fixed_config"],
        curve["safe_validated"]["fixed_validated"],
    )
    epsilon = float(
        validated["curve_config"]["parents"]["ay3"]["boundary_epsilon"]
    )
    populations = {
        population["name"]: _evaluate_population(
            population,
            predictions[population["name"]],
            renderer=renderer,
            epsilon=epsilon,
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
            "median_limited": adaptive["median_limited_fraction"]
            <= thresholds[
                "maximum_adaptive_median_limited_fraction_each_population"
            ],
            "p95_limited": adaptive["p95_limited_fraction"]
            <= thresholds[
                "maximum_adaptive_p95_limited_fraction_each_population"
            ],
            "boundary": max(
                method["maximum_new_boundary_fraction"]
                for method in population["metrics"].values()
            )
            <= thresholds["maximum_new_boundary_fraction"],
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
    "FiveKAdaptiveLUTError",
    "apply_residual_lut",
    "apply_unbounded_residual_safely",
    "fit_residual_lut",
    "run_development",
    "validate_contract",
]
