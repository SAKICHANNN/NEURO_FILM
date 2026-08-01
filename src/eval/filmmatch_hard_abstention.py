"""Development-only hard abstention for the frozen BL0 operator family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.eval.filmmatch_source_conditioned_operator import (
    _cube,
    _fit_group_operator,
    decode_operator_parameters,
    encode_operator_parameters,
    fit_ridge,
    predict_ridge,
    select_nested_alpha,
    source_descriptor,
)


def source_median_luma(source_rgb: np.ndarray) -> float:
    """Return one source-only support score without a physical EV claim."""

    source = np.asarray(source_rgb, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or not len(source)
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("source_rgb must be finite [0,1] Nx3")
    return float(np.median(source @ np.asarray([0.2126, 0.7152, 0.0722])))


def _aggregate(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    return prediction_metrics(np.concatenate(predictions), np.concatenate(targets))


def select_hard_policy(
    *,
    scores: np.ndarray,
    adaptive_predictions: list[np.ndarray],
    global_predictions: list[np.ndarray],
    targets: list[np.ndarray],
    routing: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select one threshold using only development OOF predictions."""

    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != len(targets)
        or len(adaptive_predictions) != len(targets)
        or len(global_predictions) != len(targets)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("policy rows must be finite and aligned")
    global_aggregate = _aggregate(global_predictions, targets)
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for quantile in map(float, routing["threshold_quantiles"]):
        threshold = float(np.quantile(values, quantile))
        active = values >= threshold
        routed = [
            adaptive_predictions[index] if active[index] else global_predictions[index]
            for index in range(len(values))
        ]
        routed_aggregate = _aggregate(routed, targets)
        group_ratios = np.asarray(
            [
                prediction_metrics(routed[index], targets[index])["rgb_rmse"]
                / prediction_metrics(global_predictions[index], targets[index])["rgb_rmse"]
                for index in range(len(values))
            ]
        )
        active_wins = float(np.mean(group_ratios[active] < 1.0)) if np.any(active) else 0.0
        improvement = float(
            1.0 - routed_aggregate["rgb_rmse"] / global_aggregate["rgb_rmse"]
        )
        row = {
            "quantile": quantile,
            "threshold": threshold,
            "active_groups": int(np.count_nonzero(active)),
            "active_fraction": float(np.mean(active)),
            "active_group_win_fraction": active_wins,
            "rgb_rmse_improvement_over_global": improvement,
            "p95_ratio_to_global": float(
                routed_aggregate["p95_rgb_euclidean"]
                / global_aggregate["p95_rgb_euclidean"]
            ),
            "worst_group_rmse_ratio_to_global": float(np.max(group_ratios)),
            "eligible": False,
        }
        row["eligible"] = bool(
            improvement >= float(routing["minimum_oof_improvement_over_global"])
            and row["active_fraction"] >= float(routing["minimum_oof_active_fraction"])
            and row["active_fraction"] <= float(routing["maximum_oof_active_fraction"])
            and active_wins >= float(routing["minimum_oof_active_group_win_fraction"])
            and row["p95_ratio_to_global"] <= float(routing["maximum_oof_p95_ratio_to_global"])
            and row["worst_group_rmse_ratio_to_global"]
            <= float(routing["maximum_oof_worst_group_rmse_ratio_to_global"])
        )
        rows.append(row)
        if row["eligible"]:
            eligible.append(row)
    if not eligible:
        return None, rows
    selected = min(
        eligible,
        key=lambda row: (
            -row["rgb_rmse_improvement_over_global"],
            -row["active_fraction"],
            -row["threshold"],
        ),
    )
    return dict(selected), rows


def evaluate_hard_abstention(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-fit a hard adaptive/global policy without outer-target tuning."""

    source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = datasets["reflective_records"]
    group_ids = np.asarray(
        [f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}" for row in records]
    )
    row_illuminants = np.asarray([str(row["illuminant"]) for row in records])
    groups = sorted(set(group_ids.tolist()))
    if len(groups) != int(config["data"]["groups"]):
        raise ValueError("reflective group count drift")
    illuminant_by_group = {
        group: str(row_illuminants[np.flatnonzero(group_ids == group)[0]])
        for group in groups
    }
    expected_patches = int(config["data"]["patches_per_group"])
    group_source: dict[str, np.ndarray] = {}
    group_target: dict[str, np.ndarray] = {}
    descriptors: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    fitted_operators = {}
    fitted_parameters: dict[str, np.ndarray] = {}
    operator = parent_config["operator"]
    for group in groups:
        selected = group_ids == group
        if int(np.count_nonzero(selected)) != expected_patches:
            raise ValueError("reflective patches-per-group drift")
        group_source[group] = source[selected]
        group_target[group] = target[selected]
        descriptors[group] = source_descriptor(source[selected], parent_config)
        scores[group] = source_median_luma(source[selected])
        fitted_operators[group] = _fit_group_operator(source[selected], target[selected], parent_config)
        fitted_parameters[group] = encode_operator_parameters(
            fitted_operators[group],
            curve_identity_mixture=float(operator["curve_identity_mixture"]),
            matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
        )

    alpha_candidates = [float(value) for value in parent_config["predictor"]["alpha_candidates"]]
    seed = int(config["negative_control"]["seed"])
    cube = _cube(int(config["structural_audit"]["cube_size"]))
    fold_rows = []
    all_predictions = {name: [] for name in ("global", "dense_adaptive", "hard", "shuffled_hard")}
    all_targets: list[np.ndarray] = []
    group_rows = []
    minimum_jacobian = np.inf
    maximum_oog = 0.0
    unavailable_policies = 0
    outer_fold_wins = 0

    for outer_index, held_illuminant in enumerate(sorted(set(illuminant_by_group.values()))):
        development = [g for g in groups if illuminant_by_group[g] != held_illuminant]
        held = [g for g in groups if illuminant_by_group[g] == held_illuminant]
        x_train = np.stack([descriptors[g] for g in development])
        y_train = np.stack([fitted_parameters[g] for g in development])
        labels = np.asarray([illuminant_by_group[g] for g in development])
        alpha, alpha_rows = select_nested_alpha(x_train, y_train, labels, alpha_candidates)
        rng = np.random.default_rng(seed + outer_index)
        shuffled_y = y_train[rng.permutation(len(y_train))]
        shuffled_alpha, shuffled_alpha_rows = select_nested_alpha(
            x_train, shuffled_y, labels, alpha_candidates
        )

        oof_adaptive: list[np.ndarray] = []
        oof_shuffled: list[np.ndarray] = []
        oof_global: list[np.ndarray] = []
        oof_targets: list[np.ndarray] = []
        oof_scores = []
        for nested_held in sorted(set(labels.tolist())):
            train_mask = labels != nested_held
            adaptive_model = fit_ridge(x_train[train_mask], y_train[train_mask], alpha=alpha)
            shuffled_model = fit_ridge(
                x_train[train_mask], shuffled_y[train_mask], alpha=shuffled_alpha
            )
            nested_groups = [g for g in development if illuminant_by_group[g] == nested_held]
            train_groups = [g for g in development if illuminant_by_group[g] != nested_held]
            nested_global = _fit_group_operator(
                np.concatenate([group_source[g] for g in train_groups]),
                np.concatenate([group_target[g] for g in train_groups]),
                parent_config,
            )
            for group in nested_groups:
                adaptive, _ = decode_operator_parameters(
                    predict_ridge(adaptive_model, descriptors[group])[0],
                    curve_identity_mixture=float(operator["curve_identity_mixture"]),
                    matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
                    free_logit_bounds=tuple(map(float, operator["free_logit_bounds"])),
                )
                shuffled, _ = decode_operator_parameters(
                    predict_ridge(shuffled_model, descriptors[group])[0],
                    curve_identity_mixture=float(operator["curve_identity_mixture"]),
                    matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
                    free_logit_bounds=tuple(map(float, operator["free_logit_bounds"])),
                )
                oof_adaptive.append(adaptive.apply(group_source[group]))
                oof_shuffled.append(shuffled.apply(group_source[group]))
                oof_global.append(nested_global.apply(group_source[group]))
                oof_targets.append(group_target[group])
                oof_scores.append(scores[group])
        policy, policy_rows = select_hard_policy(
            scores=np.asarray(oof_scores),
            adaptive_predictions=oof_adaptive,
            global_predictions=oof_global,
            targets=oof_targets,
            routing=config["routing"],
        )
        shuffled_policy, shuffled_policy_rows = select_hard_policy(
            scores=np.asarray(oof_scores),
            adaptive_predictions=oof_shuffled,
            global_predictions=oof_global,
            targets=oof_targets,
            routing=config["routing"],
        )
        if policy is None:
            unavailable_policies += 1

        model = fit_ridge(x_train, y_train, alpha=alpha)
        shuffled_model = fit_ridge(x_train, shuffled_y, alpha=shuffled_alpha)
        global_operator = _fit_group_operator(
            np.concatenate([group_source[g] for g in development]),
            np.concatenate([group_target[g] for g in development]),
            parent_config,
        )
        fold_global = []
        fold_hard = []
        for group in held:
            adaptive, _ = decode_operator_parameters(
                predict_ridge(model, descriptors[group])[0],
                curve_identity_mixture=float(operator["curve_identity_mixture"]),
                matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
                free_logit_bounds=tuple(map(float, operator["free_logit_bounds"])),
            )
            shuffled, _ = decode_operator_parameters(
                predict_ridge(shuffled_model, descriptors[group])[0],
                curve_identity_mixture=float(operator["curve_identity_mixture"]),
                matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
                free_logit_bounds=tuple(map(float, operator["free_logit_bounds"])),
            )
            adaptive_active = policy is not None and scores[group] >= float(policy["threshold"])
            shuffled_active = shuffled_policy is not None and scores[group] >= float(shuffled_policy["threshold"])
            predictions = {
                "global": global_operator.apply(group_source[group]),
                "dense_adaptive": adaptive.apply(group_source[group]),
                "hard": (adaptive if adaptive_active else global_operator).apply(group_source[group]),
                "shuffled_hard": (shuffled if shuffled_active else global_operator).apply(group_source[group]),
            }
            determinants = adaptive.jacobian_determinants(cube)
            minimum_jacobian = min(minimum_jacobian, float(np.min(determinants)))
            maximum_oog = max(
                maximum_oog,
                float(np.mean(np.any((predictions["hard"] < 0.0) | (predictions["hard"] > 1.0), axis=1))),
            )
            metrics = {name: prediction_metrics(value, group_target[group]) for name, value in predictions.items()}
            row = {
                "group": group,
                "source_median_luma": scores[group],
                "adaptive_active": bool(adaptive_active),
                "shuffled_active": bool(shuffled_active),
                "metrics": metrics,
                "hard_group_rmse_ratio_to_global": float(metrics["hard"]["rgb_rmse"] / metrics["global"]["rgb_rmse"]),
            }
            group_rows.append(row)
            fold_global.append(predictions["global"])
            fold_hard.append(predictions["hard"])
            for name, value in predictions.items():
                all_predictions[name].append(value)
            all_targets.append(group_target[group])
        fold_global_metrics = _aggregate(fold_global, [group_target[g] for g in held])
        fold_hard_metrics = _aggregate(fold_hard, [group_target[g] for g in held])
        fold_improvement = float(1.0 - fold_hard_metrics["rgb_rmse"] / fold_global_metrics["rgb_rmse"])
        outer_fold_wins += int(fold_improvement > 0.0)
        fold_rows.append({
            "held_illuminant": held_illuminant,
            "selected_alpha": alpha,
            "nested_alpha_scores": alpha_rows,
            "policy": policy,
            "policy_candidates": policy_rows,
            "shuffled_selected_alpha": shuffled_alpha,
            "shuffled_nested_alpha_scores": shuffled_alpha_rows,
            "shuffled_policy": shuffled_policy,
            "shuffled_policy_candidates": shuffled_policy_rows,
            "held_rgb_rmse_improvement_over_global": fold_improvement,
        })

    aggregate_metrics = {name: _aggregate(values, all_targets) for name, values in all_predictions.items()}
    global_rmse = aggregate_metrics["global"]["rgb_rmse"]
    hard_rmse = aggregate_metrics["hard"]["rgb_rmse"]
    dense_rmse = aggregate_metrics["dense_adaptive"]["rgb_rmse"]
    shuffled_rmse = aggregate_metrics["shuffled_hard"]["rgb_rmse"]
    active = np.asarray([row["adaptive_active"] for row in group_rows])
    group_ratios = np.asarray([row["hard_group_rmse_ratio_to_global"] for row in group_rows])
    observed = {
        "hard_rgb_rmse_improvement_over_global": float(1.0 - hard_rmse / global_rmse),
        "hard_rgb_rmse_improvement_over_dense_adaptive": float(1.0 - hard_rmse / dense_rmse),
        "hard_rgb_rmse_improvement_over_shuffled_hard": float(1.0 - hard_rmse / shuffled_rmse),
        "hard_active_groups": int(np.count_nonzero(active)),
        "hard_active_fraction": float(np.mean(active)),
        "hard_active_group_win_fraction": float(np.mean(group_ratios[active] < 1.0)) if np.any(active) else 0.0,
        "outer_fold_win_fraction": float(outer_fold_wins / len(fold_rows)),
        "hard_aggregate_p95_ratio_to_global": float(aggregate_metrics["hard"]["p95_rgb_euclidean"] / aggregate_metrics["global"]["p95_rgb_euclidean"]),
        "hard_worst_group_rmse_ratio_to_global": float(np.max(group_ratios)),
        "unavailable_policy_folds": unavailable_policies,
        "minimum_cube_jacobian": float(minimum_jacobian),
        "maximum_out_of_cube_fraction": float(maximum_oog),
    }
    gate = config["automatic_gate"]
    structural = config["structural_audit"]
    gates = {
        "value_over_global": observed["hard_rgb_rmse_improvement_over_global"] >= float(gate["minimum_hard_rgb_rmse_improvement_over_global"]),
        "value_over_dense": observed["hard_rgb_rmse_improvement_over_dense_adaptive"] >= float(gate["minimum_hard_rgb_rmse_improvement_over_dense_adaptive"]),
        "shuffled_control": observed["hard_rgb_rmse_improvement_over_shuffled_hard"] >= float(gate["minimum_hard_rgb_rmse_improvement_over_shuffled_hard"]),
        "minimum_coverage": observed["hard_active_fraction"] >= float(gate["minimum_hard_active_fraction"]),
        "maximum_coverage": observed["hard_active_fraction"] <= float(gate["maximum_hard_active_fraction"]),
        "active_group_wins": observed["hard_active_group_win_fraction"] >= float(gate["minimum_hard_active_group_win_fraction"]),
        "outer_fold_wins": observed["outer_fold_win_fraction"] >= float(gate["minimum_outer_fold_win_fraction"]),
        "p95": observed["hard_aggregate_p95_ratio_to_global"] <= float(gate["maximum_hard_aggregate_p95_ratio_to_global"]),
        "worst_group": observed["hard_worst_group_rmse_ratio_to_global"] <= float(gate["maximum_hard_worst_group_rmse_ratio_to_global"]),
        "all_policies_available": unavailable_policies == 0,
        "out_of_cube": observed["maximum_out_of_cube_fraction"] <= float(structural["maximum_out_of_cube_fraction"]),
        "jacobian": observed["minimum_cube_jacobian"] >= float(structural["minimum_jacobian_determinant"]),
    }
    passed = bool(all(gates.values()))
    if not gates["out_of_cube"] or not gates["jacobian"]:
        branch = "structural_failure"
    elif unavailable_policies:
        branch = "no_policy"
    elif passed:
        branch = "pass"
    else:
        branch = "unstable"
    stable = {
        "folds": fold_rows,
        "groups": group_rows,
        "aggregate_metrics": aggregate_metrics,
        "observed": observed,
        "gates": gates,
        "automatic_gate_passed": passed,
        "branch": branch,
        "validation_render_opened": False,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": canonical_sha256(stable)}


__all__ = ["evaluate_hard_abstention", "select_hard_policy", "source_median_luma"]
