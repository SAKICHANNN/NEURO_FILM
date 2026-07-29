"""U5.R2AR1 equal-group robust Velvia display-proxy operator audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.combined_velvia_operator import load_combined_velvia_pairs
from src.real_film.velvia_chart_explainability import _fit_record, _metrics
from src.roll2film.cube_diffeomorphic_flow import finite_difference_jacobians
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


SCHEMA = "neuro_film.u5_r2ar1_velvia_group_balanced_operator_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2ar1_velvia_group_balanced_operator_report.v1"
)


class VelviaGroupBalancedOperatorError(ValueError):
    """Raised when frozen AR1 lineage or semantics drift."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _read_exact_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    raw = path.read_bytes()
    if _sha256(raw) != binding["sha256"]:
        raise VelviaGroupBalancedOperatorError(
            f"input hash drift: {binding['path']}"
        )
    return json.loads(raw)


def validate_contract(config: dict[str, Any], root: Path) -> None:
    if config.get("schema") != SCHEMA:
        raise VelviaGroupBalancedOperatorError("unsupported AR1 contract")
    for name in ("combined_operator_decision", "cube_flow_decision"):
        parent = config["parents"][name]
        payload = _read_exact_json(root, parent)
        if payload.get("decision") != parent["required_decision"]:
            raise VelviaGroupBalancedOperatorError(f"{name} decision drift")
    for name in ("chart_pairs", "palette_pairs"):
        _read_exact_json(root, config["inputs"][name])
    if config["candidate"] != {
        "family": "equal_proxy_group_weighted_bounded_one_matrix",
        "weighting": "exact_equal_total_weight_by_cross_repetition",
        "fold_count": 5,
        "fold_seed": 34191,
    }:
        raise VelviaGroupBalancedOperatorError("AR1 candidate drift")
    if not all(bool(value) for value in config["development_disclosure"].values()):
        raise VelviaGroupBalancedOperatorError("AR1 disclosure drift")
    if (
        config["fit"].get("model") != "one_matrix"
        or config["evaluation"].get("domains")
        != ["velvia_chart", "velvia_palette"]
    ):
        raise VelviaGroupBalancedOperatorError("AR1 evaluation drift")


def _load_pairs(
    config: dict[str, Any], root: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    inputs = config["inputs"]
    loader_config = {
        "inputs": {
            "chart_paired_u8_sha256": inputs["chart_pairs"][
                "paired_u8_sha256"
            ],
            "palette_asset_sha256": inputs["palette_pairs"]["asset_sha256"],
            "velvia_palette_paired_u8_sha256": inputs["palette_pairs"][
                "paired_u8_sha256"
            ],
            "chart_rows": inputs["chart_pairs"]["rows"],
            "palette_rows": inputs["palette_pairs"]["rows"],
        }
    }
    return load_combined_velvia_pairs(
        root / inputs["chart_pairs"]["path"],
        root / inputs["palette_pairs"]["path"],
        loader_config,
    )


def _fit(source: np.ndarray, target: np.ndarray, fit: dict[str, Any], seed: int):
    return fit_positive_film_response_operator(
        source,
        target,
        model="one_matrix",
        identity_mixture=float(fit["identity_mixture"]),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(
            fit["maximum_function_evaluations"]
        ),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=seed,
    )


def _balanced_rows(
    chart_source: np.ndarray,
    chart_target: np.ndarray,
    palette_source: np.ndarray,
    palette_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Give each domain exactly equal total least-squares weight."""

    return (
        np.concatenate(
            (
                np.repeat(chart_source, len(palette_source), axis=0),
                np.repeat(palette_source, len(chart_source), axis=0),
            )
        ),
        np.concatenate(
            (
                np.repeat(chart_target, len(palette_target), axis=0),
                np.repeat(palette_target, len(chart_target), axis=0),
            )
        ),
    )


def _domain_metrics(
    operator: Any,
    chart_source: np.ndarray,
    chart_target: np.ndarray,
    palette_source: np.ndarray,
    palette_target: np.ndarray,
) -> dict[str, Any]:
    chart = _metrics(operator.apply(chart_source), chart_target)
    palette = _metrics(operator.apply(palette_source), palette_target)
    combined_rmse = float(
        np.sqrt(
            (
                chart["rgb_rmse"] ** 2 * len(chart_source)
                + palette["rgb_rmse"] ** 2 * len(palette_source)
            )
            / (len(chart_source) + len(palette_source))
        )
    )
    return {
        "velvia_chart": chart,
        "velvia_palette": palette,
        "combined_rgb_rmse": combined_rmse,
        "worst_domain_rgb_rmse": float(
            max(chart["rgb_rmse"], palette["rgb_rmse"])
        ),
    }


def _structure(operator: Any, config: dict[str, Any]) -> dict[str, float]:
    evaluation = config["evaluation"]
    cube_axis = np.linspace(
        0.0, 1.0, int(evaluation["cube_axis_count"])
    )
    cube = np.stack(
        np.meshgrid(cube_axis, cube_axis, cube_axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    mapped = operator.apply(cube)
    jac_axis = np.linspace(
        0.05, 0.95, int(evaluation["jacobian_axis_count"])
    )
    points = np.stack(
        np.meshgrid(jac_axis, jac_axis, jac_axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    jacobians = finite_difference_jacobians(
        operator, points, step=float(evaluation["jacobian_step"])
    )
    return {
        "minimum_jacobian_determinant": float(
            np.min(np.linalg.det(jacobians))
        ),
        "maximum_jacobian_spectral_norm": float(
            np.max(np.linalg.svd(jacobians, compute_uv=False)[:, 0])
        ),
        "raw_out_of_cube_fraction": float(
            np.mean(np.any((mapped < 0.0) | (mapped > 1.0), axis=1))
        ),
    }


def _relative_improvement(candidate: float, control: float) -> float:
    return float(1.0 - candidate / control)


def evaluate_velvia_group_balanced_operator(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    validate_contract(config, root)
    datasets = _load_pairs(config, root)
    chart_source, chart_target = datasets["velvia_chart"]
    palette_source, palette_target = datasets["velvia_palette"]
    combined_source, combined_target = datasets["combined"]
    fit = config["fit"]
    base_seed = int(fit["seed"])

    pooled = _fit(combined_source, combined_target, fit, base_seed)
    balanced_source, balanced_target = _balanced_rows(
        chart_source, chart_target, palette_source, palette_target
    )
    balanced = _fit(balanced_source, balanced_target, fit, base_seed)
    full_metrics = {
        "pooled_control": _domain_metrics(
            pooled.operator,
            chart_source,
            chart_target,
            palette_source,
            palette_target,
        ),
        "group_balanced_candidate": _domain_metrics(
            balanced.operator,
            chart_source,
            chart_target,
            palette_source,
            palette_target,
        ),
    }

    rng = np.random.default_rng(int(config["candidate"]["fold_seed"]))
    fold_count = int(config["candidate"]["fold_count"])
    chart_folds = np.array_split(rng.permutation(len(chart_source)), fold_count)
    palette_folds = np.array_split(
        rng.permutation(len(palette_source)), fold_count
    )
    fold_rows: list[dict[str, Any]] = []
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"chart": [], "palette": []}
        for name in ("pooled_control", "group_balanced_candidate")
    }
    held_targets = {"chart": [], "palette": []}
    for fold_index, (chart_hold, palette_hold) in enumerate(
        zip(chart_folds, palette_folds, strict=True)
    ):
        chart_dev = np.setdiff1d(
            np.arange(len(chart_source)), chart_hold, assume_unique=True
        )
        palette_dev = np.setdiff1d(
            np.arange(len(palette_source)), palette_hold, assume_unique=True
        )
        fold_seed = base_seed + fold_index + 1
        fold_pooled = _fit(
            np.concatenate(
                (chart_source[chart_dev], palette_source[palette_dev])
            ),
            np.concatenate(
                (chart_target[chart_dev], palette_target[palette_dev])
            ),
            fit,
            fold_seed,
        )
        fold_balanced_source, fold_balanced_target = _balanced_rows(
            chart_source[chart_dev],
            chart_target[chart_dev],
            palette_source[palette_dev],
            palette_target[palette_dev],
        )
        fold_balanced = _fit(
            fold_balanced_source, fold_balanced_target, fit, fold_seed
        )
        metrics = {
            "pooled_control": _domain_metrics(
                fold_pooled.operator,
                chart_source[chart_hold],
                chart_target[chart_hold],
                palette_source[palette_hold],
                palette_target[palette_hold],
            ),
            "group_balanced_candidate": _domain_metrics(
                fold_balanced.operator,
                chart_source[chart_hold],
                chart_target[chart_hold],
                palette_source[palette_hold],
                palette_target[palette_hold],
            ),
        }
        fold_rows.append(
            {
                "fold_index": fold_index,
                "chart_holdout_indices": chart_hold.tolist(),
                "palette_holdout_indices": palette_hold.tolist(),
                "metrics": metrics,
                "worst_domain_improvement": _relative_improvement(
                    metrics["group_balanced_candidate"][
                        "worst_domain_rgb_rmse"
                    ],
                    metrics["pooled_control"]["worst_domain_rgb_rmse"],
                ),
            }
        )
        for name, operator in (
            ("pooled_control", fold_pooled.operator),
            ("group_balanced_candidate", fold_balanced.operator),
        ):
            collected[name]["chart"].append(
                operator.apply(chart_source[chart_hold])
            )
            collected[name]["palette"].append(
                operator.apply(palette_source[palette_hold])
            )
        held_targets["chart"].append(chart_target[chart_hold])
        held_targets["palette"].append(palette_target[palette_hold])

    crossfit_metrics = {}
    target_chart = np.concatenate(held_targets["chart"])
    target_palette = np.concatenate(held_targets["palette"])
    for name in collected:
        chart_prediction = np.concatenate(collected[name]["chart"])
        palette_prediction = np.concatenate(collected[name]["palette"])
        chart = _metrics(chart_prediction, target_chart)
        palette = _metrics(palette_prediction, target_palette)
        crossfit_metrics[name] = {
            "velvia_chart": chart,
            "velvia_palette": palette,
            "combined_rgb_rmse": float(
                np.sqrt(
                    (
                        chart["rgb_rmse"] ** 2 * len(target_chart)
                        + palette["rgb_rmse"] ** 2 * len(target_palette)
                    )
                    / (len(target_chart) + len(target_palette))
                )
            ),
            "worst_domain_rgb_rmse": float(
                max(chart["rgb_rmse"], palette["rgb_rmse"])
            ),
        }

    full_worst_improvement = _relative_improvement(
        full_metrics["group_balanced_candidate"]["worst_domain_rgb_rmse"],
        full_metrics["pooled_control"]["worst_domain_rgb_rmse"],
    )
    crossfit_worst_improvement = _relative_improvement(
        crossfit_metrics["group_balanced_candidate"][
            "worst_domain_rgb_rmse"
        ],
        crossfit_metrics["pooled_control"]["worst_domain_rgb_rmse"],
    )
    full_combined_regression = -_relative_improvement(
        full_metrics["group_balanced_candidate"]["combined_rgb_rmse"],
        full_metrics["pooled_control"]["combined_rgb_rmse"],
    )
    crossfit_combined_regression = -_relative_improvement(
        crossfit_metrics["group_balanced_candidate"]["combined_rgb_rmse"],
        crossfit_metrics["pooled_control"]["combined_rgb_rmse"],
    )
    fold_win_fraction = float(
        np.mean(
            [
                row["worst_domain_improvement"] > 0.0
                for row in fold_rows
            ]
        )
    )
    structural = _structure(balanced.operator, config)
    gates = config["automatic_gates"]
    checks = {
        "full_worst_domain_improvement": full_worst_improvement
        >= float(gates["minimum_full_worst_domain_rmse_improvement"]),
        "crossfit_worst_domain_improvement": crossfit_worst_improvement
        >= float(gates["minimum_crossfit_worst_domain_rmse_improvement"]),
        "full_combined_regression": full_combined_regression
        <= float(gates["maximum_full_combined_rmse_regression"]),
        "crossfit_combined_regression": crossfit_combined_regression
        <= float(gates["maximum_crossfit_combined_rmse_regression"]),
        "fold_worst_domain_win_fraction": fold_win_fraction
        >= float(gates["minimum_fold_worst_domain_win_fraction"]),
        "positive_jacobian": structural["minimum_jacobian_determinant"]
        >= float(gates["minimum_jacobian_determinant"]),
        "bounded_jacobian_norm": structural[
            "maximum_jacobian_spectral_norm"
        ]
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "in_cube": structural["raw_out_of_cube_fraction"]
        <= float(gates["maximum_raw_out_of_cube_fraction"]),
    }
    automatic_pass = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "development_disclosure": config["development_disclosure"],
        "full_fit": {
            "pooled_control": _fit_record(pooled),
            "group_balanced_candidate": _fit_record(balanced),
            "metrics": full_metrics,
        },
        "crossfit": {
            "folds": fold_rows,
            "metrics": crossfit_metrics,
        },
        "summary": {
            "full_worst_domain_improvement": full_worst_improvement,
            "crossfit_worst_domain_improvement": crossfit_worst_improvement,
            "full_combined_regression": full_combined_regression,
            "crossfit_combined_regression": crossfit_combined_regression,
            "fold_worst_domain_win_fraction": fold_win_fraction,
        },
        "structural": structural,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_group_balanced_development_challenger"
            if automatic_pass
            else "close_group_balancing_on_display_proxy_family"
        ),
        "photographic_render_allowed": bool(automatic_pass),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "VelviaGroupBalancedOperatorError",
    "evaluate_velvia_group_balanced_operator",
    "validate_contract",
]
