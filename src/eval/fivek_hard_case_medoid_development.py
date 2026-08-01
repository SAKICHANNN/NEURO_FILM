"""Development audit for hard retrieval of bounded FiveK case operators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.build_fivek_freeze_pack import (
    filtered_target,
    load_expert_icc_srgb,
    load_raw_default,
    resize_to_shape,
)
from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_neutral_base,
    validate_contract as validate_safe_contract,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_neutral_base_parameter_pilot import source_descriptor
from src.eval.fivek_unseen_content_confirmation import (
    _median_delta_e76,
    _new_boundary_fraction,
    _rmse,
    _summary,
    load_srgb16,
)


class FiveKHardCaseMedoidError(ValueError):
    """Raised when the frozen hard-case experiment boundary drifts."""


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
    root: Path, item: Mapping[str, Any], key: str
) -> dict[str, Any]:
    path = root / str(item[key])
    expected = str(item[f"{key}_sha256"]).lower()
    if not path.is_file() or _sha256(path) != expected:
        raise FiveKHardCaseMedoidError(f"evidence drift: {item[key]}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKHardCaseMedoidError("contract is not frozen")
    if (
        config.get("new_training_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or config["fixed_model"].get("case_blending_allowed")
        or config["fixed_model"].get("case_refitting_allowed")
    ):
        raise FiveKHardCaseMedoidError("hard-case boundary drift")
    expected_methods = {
        "global",
        "ridge",
        "content_nearest",
        "ridge_projected_medoid",
        "content_top3_ridge_medoid",
        "content_top5_ridge_medoid",
    }
    if set(config["methods"]) != expected_methods:
        raise FiveKHardCaseMedoidError("method inventory drift")
    parents = config["parents"]
    ay0_config = _load_hashed_json(root, parents["ay0"], "config")
    ay0_report = _load_hashed_json(root, parents["ay0"], "report")
    safe_config = _load_hashed_json(root, parents["ay3"], "config")
    safe_decision = _load_hashed_json(root, parents["ay3"], "decision")
    fresh_manifest = _load_hashed_json(
        root, parents["fresh_population"], "manifest"
    )
    fresh_report = _load_hashed_json(
        root, parents["fresh_population"], "report"
    )
    fresh_decision = _load_hashed_json(
        root, parents["fresh_population"], "decision"
    )
    failed = _load_hashed_json(
        root, parents["failed_uncertainty"], "decision"
    )
    safe_validated = validate_safe_contract(root, safe_config)
    if (
        ay0_report.get("automatic_pass") is not True
        or safe_decision.get("status")
        != "development_pass_fresh_confirmation_required"
        or fresh_report.get("automatic_pass") is not True
        or fresh_decision.get("status") != "pass_fresh_confirmation_ready"
        or len(fresh_manifest.get("rows", [])) != 63
        or failed.get("status")
        != parents["failed_uncertainty"]["required_status"]
    ):
        raise FiveKHardCaseMedoidError("parent eligibility drift")
    return {
        "ay0_config": ay0_config,
        "ay0_report": ay0_report,
        "safe_config": safe_config,
        "safe_validated": safe_validated,
        "fresh_manifest": fresh_manifest,
    }


def _parameter_choices(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_ids: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[str | None]]]:
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)
    ridge = np.clip(
        Ridge(alpha=alpha).fit(train_scaled, train_y).predict(test_scaled),
        lower,
        upper,
    )
    global_parameters = np.repeat(
        np.median(train_y, axis=0)[None, :], len(test_x), axis=0
    )
    content_distance = np.linalg.norm(
        test_scaled[:, None, :] - train_scaled[None, :, :], axis=2
    ) / np.sqrt(train_scaled.shape[1])
    span = np.maximum(upper - lower, 1e-12)
    parameter_distance = np.linalg.norm(
        (ridge[:, None, :] - train_y[None, :, :]) / span[None, None, :],
        axis=2,
    )
    nearest_index = np.argmin(content_distance, axis=1)
    projected_index = np.argmin(parameter_distance, axis=1)
    parameters: dict[str, np.ndarray] = {
        "global": global_parameters,
        "ridge": ridge,
        "content_nearest": train_y[nearest_index],
        "ridge_projected_medoid": train_y[projected_index],
    }
    identities: dict[str, list[str | None]] = {
        "global": [None] * len(test_x),
        "ridge": [None] * len(test_x),
        "content_nearest": train_ids[nearest_index].tolist(),
        "ridge_projected_medoid": train_ids[projected_index].tolist(),
    }
    for count in (3, 5):
        nearest = np.argsort(
            content_distance, axis=1, kind="stable"
        )[:, :count]
        selected = np.asarray(
            [
                indices[
                    np.argmin(parameter_distance[row_index, indices])
                ]
                for row_index, indices in enumerate(nearest)
            ],
            dtype=np.int64,
        )
        name = f"content_top{count}_ridge_medoid"
        parameters[name] = train_y[selected]
        identities[name] = train_ids[selected].tolist()
    return parameters, identities


def _load_ay0_population(
    root: Path, ay0_config: Mapping[str, Any], ay0_report: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = json.loads(
        (root / ay0_config["source_evidence"]["manifest"]).read_text(
            encoding="utf-8"
        )
    )
    source_by_id = {row["pair_id"]: row for row in manifest["rows"]}
    fitted_by_id = {
        row["pair_id"]: row["fitted_parameters"]
        for row in ay0_report["rows"]
    }
    maximum_side = int(ay0_config["decode"]["maximum_side"])
    policy = type(
        "TargetPolicy",
        (),
        {
            "luma_strength": ay0_config["neutral_target"]["luma_strength"],
            "chroma_strength": ay0_config["neutral_target"][
                "chroma_strength"
            ],
            "chroma_headroom": ay0_config["neutral_target"][
                "chroma_headroom"
            ],
            "wb_anchor_strength": ay0_config["neutral_target"][
                "white_balance_anchor_strength"
            ],
        },
    )()
    rows = []
    for pair_id in sorted(fitted_by_id):
        evidence = source_by_id[pair_id]
        source, _ = load_raw_default(
            root / evidence["raw_path"], maximum_side
        )
        source = np.clip(source, 0.0, 1.0)
        expert = load_expert_icc_srgb(
            root / evidence["expert_path"], maximum_side
        )
        if expert.shape != source.shape:
            expert = resize_to_shape(expert, source.shape[:2])
        target = filtered_target(
            source, np.clip(expert, 0.0, 1.0), policy
        )
        rows.append(
            {
                "pair_id": pair_id,
                "group": evidence["camera_group_id"],
                "source": source,
                "target": target,
                "descriptor": source_descriptor(source, ay0_config),
                "fitted": np.asarray(
                    fitted_by_id[pair_id], dtype=np.float64
                ),
            }
        )
    return {"name": "ay0_complete_camera_group_crossfit", "rows": rows}


def _load_fresh_population(
    root: Path,
    ay0_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    group_field: str | None = "camera_model",
) -> dict[str, Any]:
    rows = []
    maximum_side = int(ay0_config["decode"]["maximum_side"])
    for evidence in manifest["rows"]:
        source_path = Path(evidence["source_path"])
        target_path = Path(evidence["target_path"])
        if (
            _sha256(source_path) != evidence["source_sha256"]
            or _sha256(target_path) != evidence["target_sha256"]
        ):
            raise FiveKHardCaseMedoidError(
                f"fresh asset drift: {evidence['pair_id']}"
            )
        source = load_srgb16(source_path, maximum_side)
        target = load_srgb16(target_path, maximum_side)
        rows.append(
            {
                "pair_id": evidence["pair_id"],
                "group": (
                    evidence[group_field]
                    if group_field is not None
                    else "unknown"
                ),
                "source": source,
                "target": target,
                "descriptor": source_descriptor(source, ay0_config),
            }
        )
    return {"name": "fresh_63", "rows": rows}


def _evaluate_population(
    population: Mapping[str, Any],
    *,
    training_population: Mapping[str, Any],
    methods: list[str],
    alpha: float,
    lower: np.ndarray,
    upper: np.ndarray,
    epsilon: float,
    renderer: Any,
    crossfit: bool,
) -> dict[str, Any]:
    rows = population["rows"]
    descriptors = np.stack([row["descriptor"] for row in rows])
    groups = np.asarray([row["group"] for row in rows], dtype=object)
    parameter_predictions = {
        method: np.empty((len(rows), len(lower)), dtype=np.float64)
        for method in methods
    }
    case_ids: dict[str, list[str | None]] = {
        method: [None] * len(rows) for method in methods
    }
    if crossfit:
        training_rows = rows
        splits = GroupKFold(n_splits=5).split(
            descriptors, groups=groups
        )
    else:
        training_rows = training_population["rows"]
        splits = [(np.arange(len(training_rows)), np.arange(len(rows)))]
    training_x_all = np.stack(
        [row["descriptor"] for row in training_rows]
    )
    training_y_all = np.stack(
        [row["fitted"] for row in training_rows]
    )
    training_ids_all = np.asarray(
        [row["pair_id"] for row in training_rows], dtype=object
    )
    for train_index, test_index in splits:
        if crossfit:
            train_x = descriptors[train_index]
            train_y = np.stack(
                [training_rows[index]["fitted"] for index in train_index]
            )
            train_ids = np.asarray(
                [training_rows[index]["pair_id"] for index in train_index],
                dtype=object,
            )
        else:
            train_x = training_x_all
            train_y = training_y_all
            train_ids = training_ids_all
        choices, identities = _parameter_choices(
            train_x,
            train_y,
            train_ids,
            descriptors[test_index],
            alpha=alpha,
            lower=lower,
            upper=upper,
        )
        for method in methods:
            parameter_predictions[method][test_index] = choices[method]
            for local_index, row_index in enumerate(test_index):
                case_ids[method][int(row_index)] = identities[method][
                    local_index
                ]
    neutral_errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    limited = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    output_rows = []
    for index, row in enumerate(rows):
        target_look, _ = renderer(row["target"])
        target_style = _median_delta_e76(row["target"], target_look)
        record: dict[str, Any] = {
            "pair_id": row["pair_id"],
            "group": row["group"],
            "target_style_delta_e76": target_style,
        }
        for method in methods:
            parameters = parameter_predictions[method][index]
            neutral, scale = apply_boundary_safe_neutral_base(
                row["source"],
                parameters,
                boundary_epsilon=epsilon,
            )
            look, _ = renderer(neutral)
            neutral_error = _rmse(neutral, row["target"])
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(neutral, look)
            boundary = max(
                _new_boundary_fraction(row["source"], neutral, epsilon),
                _new_boundary_fraction(neutral, look, epsilon),
            )
            limited_fraction = float(np.mean(scale < 1.0 - 1e-12))
            neutral_errors[method].append(neutral_error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            limited[method].append(limited_fraction)
            boundaries[method].append(boundary)
            record[method] = {
                "parameters": parameters.tolist(),
                "selected_case_id": case_ids[method][index],
                "neutral_rmse": neutral_error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "limited_fraction": limited_fraction,
                "new_boundary_fraction": boundary,
            }
        output_rows.append(record)
    global_neutral = np.asarray(neutral_errors["global"])
    metrics = {}
    for method in methods:
        method_neutral = np.asarray(neutral_errors[method])
        selected = {
            value for value in case_ids[method] if value is not None
        }
        metrics[method] = {
            "neutral": _summary(neutral_errors[method]),
            "look": _summary(look_errors[method]),
            "mean_improvement_over_global": float(
                (global_neutral.mean() - method_neutral.mean())
                / max(global_neutral.mean(), 1e-12)
            ),
            "win_fraction_over_global": float(
                np.mean(method_neutral < global_neutral)
            ),
            "p95_ratio_to_global": float(
                np.quantile(method_neutral, 0.95)
                / max(np.quantile(global_neutral, 0.95), 1e-12)
            ),
            "worst_ratio_to_global": float(
                np.max(method_neutral)
                / max(np.max(global_neutral), 1e-12)
            ),
            "median_style_delta_e76": float(np.median(styles[method])),
            "style_ratio_to_global": float(
                np.median(styles[method])
                / max(np.median(styles["global"]), 1e-12)
            ),
            "median_limited_fraction": float(np.median(limited[method])),
            "p95_limited_fraction": float(
                np.quantile(limited[method], 0.95)
            ),
            "maximum_new_boundary_fraction": float(
                np.max(boundaries[method])
            ),
            "distinct_selected_cases": len(selected),
        }
    return {
        "name": population["name"],
        "row_count": len(rows),
        "group_count": len(set(groups.tolist())),
        "metrics": metrics,
        "rows": output_rows,
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
    ay0 = _load_ay0_population(
        root, validated["ay0_config"], validated["ay0_report"]
    )
    fresh = _load_fresh_population(
        root, validated["ay0_config"], validated["fresh_manifest"]
    )
    lower = np.asarray(
        validated["ay0_config"]["operator"]["lower_bounds"],
        dtype=np.float64,
    )
    upper = np.asarray(
        validated["ay0_config"]["operator"]["upper_bounds"],
        dtype=np.float64,
    )
    renderer = build_fixed_ao6_renderer(
        validated["safe_validated"]["fixed_config"],
        validated["safe_validated"]["fixed_validated"],
    )
    methods = [str(method) for method in config["methods"]]
    common = {
        "training_population": ay0,
        "methods": methods,
        "alpha": float(config["fixed_model"]["ridge_alpha"]),
        "lower": lower,
        "upper": upper,
        "epsilon": float(config["parents"]["ay3"]["boundary_epsilon"]),
        "renderer": renderer,
    }
    populations = {
        ay0["name"]: _evaluate_population(ay0, crossfit=True, **common),
        fresh["name"]: _evaluate_population(
            fresh, crossfit=False, **common
        ),
    }
    thresholds = config["evaluation"]
    gates = {}
    eligible_methods = []
    hard_methods = [
        method
        for method in methods
        if method not in {"global", "ridge"}
    ]
    for method in hard_methods:
        method_gates = {}
        for name, population in populations.items():
            metric = population["metrics"][method]
            method_gates[name] = {
                "mean": metric["mean_improvement_over_global"]
                >= thresholds[
                    "minimum_mean_improvement_over_global_each_population"
                ],
                "wins": metric["win_fraction_over_global"]
                >= thresholds[
                    "minimum_win_fraction_over_global_each_population"
                ],
                "p95": metric["p95_ratio_to_global"]
                <= thresholds[
                    "maximum_p95_ratio_to_global_each_population"
                ],
                "worst": metric["worst_ratio_to_global"]
                <= thresholds[
                    "maximum_worst_ratio_to_global_each_population"
                ],
                "style": metric["style_ratio_to_global"]
                >= thresholds[
                    "minimum_ao6_style_ratio_each_population"
                ],
                "median_limited": metric["median_limited_fraction"]
                <= thresholds[
                    "maximum_median_limited_fraction_each_population"
                ],
                "p95_limited": metric["p95_limited_fraction"]
                <= thresholds[
                    "maximum_p95_limited_fraction_each_population"
                ],
                "boundary": metric["maximum_new_boundary_fraction"]
                <= thresholds["maximum_new_boundary_fraction"],
                "case_diversity": metric["distinct_selected_cases"]
                >= thresholds[
                    "minimum_distinct_selected_cases_each_population"
                ],
            }
        gates[method] = method_gates
        if all(
            all(values.values()) for values in method_gates.values()
        ):
            eligible_methods.append(method)
    selected_method = None
    if eligible_methods:
        selected_method = min(
            eligible_methods,
            key=lambda method: (
                -min(
                    populations[name]["metrics"][method][
                        "mean_improvement_over_global"
                    ]
                    for name in populations
                ),
                max(
                    populations[name]["metrics"][method][
                        "p95_ratio_to_global"
                    ]
                    for name in populations
                ),
                method,
            ),
        )
    stable = {
        "populations": populations,
        "gates": gates,
        "eligible_methods": eligible_methods,
        "selected_method": selected_method,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": selected_method is not None,
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
    "FiveKHardCaseMedoidError",
    "run_development",
    "validate_contract",
]
