"""Analytical shared-scale safety executor for the bounded neutral base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.color import rgb2lab

from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
    validate_contract as validate_ay1_contract,
)
from src.eval.fivek_neutral_base_parameter_pilot import apply_neutral_base
from src.eval.fivek_unseen_content_confirmation import load_srgb16


class BoundarySafeNeutralBaseError(ValueError):
    """Raised when the analytical safety contract or evidence drifts."""


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
        raise BoundarySafeNeutralBaseError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise BoundarySafeNeutralBaseError("contract is not frozen")
    neutral = config["neutral_operator"]
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_integration_allowed")
        or neutral.get("hard_output_clipping_allowed")
        or neutral.get("channel_independent_residual_scaling_allowed")
        or neutral.get("parameter_feature_alpha_or_capacity_change_allowed")
        or config["fixed_look"].get("change_allowed")
    ):
        raise BoundarySafeNeutralBaseError("analytical boundary drift")
    evidence = config["development_evidence"]
    ay2_config = _load_hashed_json(
        root, evidence["ay2_config"], evidence["ay2_config_sha256"]
    )
    ay2_report = _load_hashed_json(
        root, evidence["ay2_report"], evidence["ay2_report_sha256"]
    )
    source_manifest = _load_hashed_json(
        root,
        evidence["ay2_source_manifest"],
        evidence["ay2_source_manifest_sha256"],
    )
    neutral_config = _load_hashed_json(
        root, neutral["config"], neutral["config_sha256"]
    )
    if (
        ay2_report.get("automatic_pass") is not False
        or ay2_report.get("gates", {}).get("new_boundaries") is not False
        or len(ay2_report.get("rows", [])) != evidence["rows"]
        or len(source_manifest.get("rows", [])) != evidence["rows"]
    ):
        raise BoundarySafeNeutralBaseError("AY2 evidence is not the vetoed set")
    fixed = config["fixed_look"]
    ay1_config = {
        "status": "contract_frozen_implementation_ready",
        "neutral_base": {
            "config": neutral["config"],
            "config_sha256": neutral["config_sha256"],
            "decision": "configs/u5_r2ay0_fivek_neutral_base_parameter_decision_v1.json",
            "decision_sha256": "b24953e064de479d761c8c57659e6c3000046b429b6566ff55acdfd207b81b74",
            "report": "outputs/u5_r2ay0_fivek_neutral_base_parameter_pilot_v1/run_a/report.json",
            "report_sha256": "5fb8e8e74312fd2b71f5ce50f43893225fcccf61f48f5b59ff9a50c6a169af5d",
            "required_stable_evidence_id": "26353480f55cdd88c8b21f3d4200269693d0158013f7a679bd92bf8c80f4e3c4",
            "required_automatic_pass": True,
        },
        "fixed_look": {
            **fixed,
            "per_image_style_adjustment_allowed": False,
            "operator_refit_allowed": False,
            "strength_change_allowed": False,
        },
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "final_rgb_learning_allowed": False,
        "production_integration_allowed": False,
    }
    fixed_validated = validate_ay1_contract(root, ay1_config)
    return {
        "ay2_config": ay2_config,
        "ay2_report": ay2_report,
        "source_manifest": source_manifest,
        "neutral_config": neutral_config,
        "fixed_config": ay1_config,
        "fixed_validated": fixed_validated,
    }


def apply_boundary_safe_neutral_base(
    encoded_rgb: np.ndarray,
    parameters: np.ndarray,
    *,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared per-pixel scale along the explicit RGB residual."""

    source = np.asarray(encoded_rgb, dtype=np.float64)
    if not 0.0 < boundary_epsilon < 0.5:
        raise BoundarySafeNeutralBaseError("invalid boundary epsilon")
    candidate = apply_neutral_base(source, parameters)
    residual = candidate - source
    lower = np.nextafter(float(boundary_epsilon), 1.0)
    upper = np.nextafter(1.0 - float(boundary_epsilon), 0.0)
    interior = (source > boundary_epsilon) & (
        source < 1.0 - boundary_epsilon
    )
    channel_scale = np.ones_like(source, dtype=np.float64)
    positive = interior & (residual > 0.0)
    negative = interior & (residual < 0.0)
    channel_scale[positive] = (
        upper - source[positive]
    ) / residual[positive]
    channel_scale[negative] = (
        lower - source[negative]
    ) / residual[negative]
    scale = np.minimum(
        1.0, np.min(channel_scale, axis=2)
    )
    scale = np.maximum(scale, 0.0)
    # The exact quotient can reconstruct to the excluded boundary after the
    # multiply-add rounds.  Move every genuinely limited shared scale by one
    # representable float toward the source; this preserves the analytical
    # direction and common RGB scale without clipping an output channel.
    scale = np.where(
        scale < 1.0, np.nextafter(scale, 0.0), scale
    )
    output = source + scale[..., None] * residual
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise BoundarySafeNeutralBaseError(
            "analytical executor escaped the RGB cube"
        )
    source_boundary = (source <= boundary_epsilon) | (
        source >= 1.0 - boundary_epsilon
    )
    output_boundary = (output <= boundary_epsilon) | (
        output >= 1.0 - boundary_epsilon
    )
    if np.any(output_boundary & ~source_boundary):
        raise BoundarySafeNeutralBaseError(
            "analytical proof failed to prevent a new boundary"
        )
    return output, scale


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(np.asarray(first).reshape(-1, 1, 3))
    second_lab = rgb2lab(np.asarray(second).reshape(-1, 1, 3))
    return float(
        np.median(
            np.linalg.norm(
                second_lab.reshape(-1, 3) - first_lab.reshape(-1, 3),
                axis=1,
            )
        )
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_rmse": float(np.mean(array)),
        "median_rmse": float(np.median(array)),
        "p95_rmse": float(np.quantile(array, 0.95)),
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
    renderer = build_fixed_ao6_renderer(
        validated["fixed_config"], validated["fixed_validated"]
    )
    source_rows = {
        row["pair_id"]: row for row in validated["source_manifest"]["rows"]
    }
    methods = [str(value) for value in config["methods"]]
    neutral_errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    limited = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    epsilon = float(config["neutral_operator"]["boundary_epsilon"])
    maximum_side = int(
        validated["ay2_config"]["confirmation"]["maximum_side"]
    )
    rows = []
    for original in validated["ay2_report"]["rows"]:
        pair_id = str(original["pair_id"])
        evidence = source_rows[pair_id]
        source_path = root / evidence["source_path"]
        target_path = root / evidence["target_path"]
        if (
            _sha256(source_path) != evidence["source_sha256"]
            or _sha256(target_path) != evidence["target_sha256"]
        ):
            raise BoundarySafeNeutralBaseError(
                f"source drift: {pair_id}"
            )
        source = load_srgb16(source_path, maximum_side)
        target = load_srgb16(target_path, maximum_side)
        target_look, _ = renderer(target)
        record: dict[str, Any] = {"pair_id": pair_id}
        for method in methods:
            parameters = np.asarray(
                original[method]["parameters"], dtype=np.float64
            )
            neutral, scale = apply_boundary_safe_neutral_base(
                source,
                parameters,
                boundary_epsilon=epsilon,
            )
            look, diagnostics = renderer(neutral)
            neutral_error = _rmse(neutral, target)
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(neutral, look)
            limited_fraction = float(np.mean(scale < 1.0 - 1e-12))
            source_boundary = (source <= epsilon) | (
                source >= 1.0 - epsilon
            )
            output_boundary = (neutral <= epsilon) | (
                neutral >= 1.0 - epsilon
            )
            boundary = float(
                np.mean(output_boundary & ~source_boundary)
            )
            neutral_errors[method].append(neutral_error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            limited[method].append(limited_fraction)
            boundaries[method].append(boundary)
            record[method] = {
                "neutral_rmse": neutral_error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "limited_fraction": limited_fraction,
                "minimum_scale": float(np.min(scale)),
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
    ridge = np.asarray(neutral_errors["ridge"])
    global_errors = np.asarray(neutral_errors["global"])
    original_metrics = validated["ay2_report"]["metrics"]["ridge"]
    evaluation = config["evaluation"]
    observed = {
        "safe_ridge_mean_improvement_over_safe_global": float(
            (np.mean(global_errors) - np.mean(ridge))
            / max(np.mean(global_errors), 1e-12)
        ),
        "safe_ridge_win_fraction_over_safe_global": float(
            np.mean(ridge < global_errors)
        ),
        "safe_ridge_p95_ratio_to_safe_global": float(
            np.quantile(ridge, 0.95)
            / max(np.quantile(global_errors, 0.95), 1e-12)
        ),
        "safe_ridge_neutral_rmse_ratio_to_original_ridge": float(
            metrics["ridge"]["neutral"]["mean_rmse"]
            / original_metrics["neutral"]["mean_rmse"]
        ),
        "safe_ridge_look_rmse_ratio_to_original_ridge": float(
            metrics["ridge"]["look"]["mean_rmse"]
            / original_metrics["look"]["mean_rmse"]
        ),
        "safe_ridge_style_ratio_to_original_ridge": float(
            metrics["ridge"]["median_style_delta_e76"]
            / original_metrics["median_style_delta_e76"]
        ),
        "safe_ridge_median_limited_fraction": metrics["ridge"][
            "median_limited_fraction"
        ],
        "safe_ridge_p95_limited_fraction": metrics["ridge"][
            "p95_limited_fraction"
        ],
        "maximum_new_boundary_fraction": float(
            max(
                metrics[method]["maximum_new_boundary_fraction"]
                for method in methods
            )
        ),
    }
    gates = {
        "ridge_mean": observed[
            "safe_ridge_mean_improvement_over_safe_global"
        ]
        >= evaluation[
            "minimum_safe_ridge_mean_improvement_over_safe_global"
        ],
        "ridge_wins": observed[
            "safe_ridge_win_fraction_over_safe_global"
        ]
        >= evaluation[
            "minimum_safe_ridge_win_fraction_over_safe_global"
        ],
        "ridge_tail": observed[
            "safe_ridge_p95_ratio_to_safe_global"
        ]
        <= evaluation["maximum_safe_ridge_p95_ratio_to_safe_global"],
        "neutral_quality": observed[
            "safe_ridge_neutral_rmse_ratio_to_original_ridge"
        ]
        <= evaluation[
            "maximum_safe_ridge_neutral_rmse_ratio_to_original_ridge"
        ],
        "look_quality": observed[
            "safe_ridge_look_rmse_ratio_to_original_ridge"
        ]
        <= evaluation[
            "maximum_safe_ridge_look_rmse_ratio_to_original_ridge"
        ],
        "style": observed["safe_ridge_style_ratio_to_original_ridge"]
        >= evaluation["minimum_safe_ridge_style_ratio_to_original_ridge"],
        "median_limited": observed[
            "safe_ridge_median_limited_fraction"
        ]
        <= evaluation["maximum_safe_ridge_median_limited_fraction"],
        "p95_limited": observed["safe_ridge_p95_limited_fraction"]
        <= evaluation["maximum_safe_ridge_p95_limited_fraction"],
        "new_boundaries": observed["maximum_new_boundary_fraction"]
        <= evaluation["maximum_new_boundary_fraction"],
    }
    stable_payload = {
        "metrics": metrics,
        "observed": observed,
        "gates": gates,
        "rows": rows,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        "ay2_report_sha256": config["development_evidence"][
            "ay2_report_sha256"
        ],
        **stable_payload,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_payload)
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
    "BoundarySafeNeutralBaseError",
    "apply_boundary_safe_neutral_base",
    "run_development",
    "validate_contract",
]
