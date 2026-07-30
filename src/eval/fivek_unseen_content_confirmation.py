"""Untouched-content confirmation for the bounded FiveK neutral base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tifffile
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2lab

from scripts.build_fivek_freeze_pack import load_raw_default, resize_float
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
    validate_contract as validate_ay1_contract,
)
from src.eval.fivek_neutral_base_parameter_pilot import (
    apply_neutral_base,
    fit_neutral_base_parameters,
    source_descriptor,
)


class FiveKUnseenContentConfirmationError(ValueError):
    """Raised when the confirmation contract or lineage drifts."""


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
        raise FiveKUnseenContentConfirmationError(
            f"frozen evidence drift: {path}"
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKUnseenContentConfirmationError(
            "confirmation contract is not frozen"
        )
    if (
        config.get("operator_fitting_allowed_on_confirmation")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("confirmation_target_access_during_training_allowed")
        or config["development"].get("feature_or_parameter_change_allowed")
        or config["fixed_look"].get("operator_refit_allowed")
        or config["fixed_look"].get("strength_change_allowed")
    ):
        raise FiveKUnseenContentConfirmationError(
            "confirmation no-change boundary drift"
        )
    development = config["development"]
    development_config = _load_hashed_json(
        root, development["config"], development["config_sha256"]
    )
    development_report = _load_hashed_json(
        root, development["report"], development["report_sha256"]
    )
    if (
        development_report.get("automatic_pass") is not True
        or len(development_report.get("rows", [])) != development["rows"]
    ):
        raise FiveKUnseenContentConfirmationError(
            "development evidence is not eligible"
        )
    confirmation = config["confirmation"]
    confirmation_manifest = _load_hashed_json(
        root, confirmation["manifest"], confirmation["manifest_sha256"]
    )
    confirmation_report = _load_hashed_json(
        root, confirmation["report"], confirmation["report_sha256"]
    )
    if (
        confirmation_report.get("automatic_pass")
        is not confirmation["required_automatic_pass"]
        or len(confirmation_manifest.get("rows", []))
        != confirmation["rows"]
    ):
        raise FiveKUnseenContentConfirmationError(
            "confirmation source is not eligible"
        )
    ay1_config = {
        "status": "contract_frozen_implementation_ready",
        "neutral_base": {
            "config": development["config"],
            "config_sha256": development["config_sha256"],
            "decision": "configs/u5_r2ay0_fivek_neutral_base_parameter_decision_v1.json",
            "decision_sha256": "b24953e064de479d761c8c57659e6c3000046b429b6566ff55acdfd207b81b74",
            "report": development["report"],
            "report_sha256": development["report_sha256"],
            "required_stable_evidence_id": development_report[
                "stable_evidence_id"
            ],
            "required_automatic_pass": True,
        },
        "fixed_look": {
            **config["fixed_look"],
            "per_image_style_adjustment_allowed": False,
        },
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "final_rgb_learning_allowed": False,
        "production_integration_allowed": False,
    }
    fixed_validated = validate_ay1_contract(root, ay1_config)
    return {
        "development_config": development_config,
        "development_report": development_report,
        "confirmation_manifest": confirmation_manifest,
        "fixed_config": ay1_config,
        "fixed_validated": fixed_validated,
    }


def _load_srgb16(path: Path, maximum_side: int) -> np.ndarray:
    data = tifffile.imread(path)
    if data.ndim != 3 or data.shape[-1] != 3 or data.dtype != np.uint16:
        raise FiveKUnseenContentConfirmationError(
            f"expected exact RGB uint16 TIFF: {path}"
        )
    encoded = data.astype(np.float32) / np.float32(65535.0)
    return np.clip(resize_float(encoded, maximum_side), 0.0, 1.0)


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


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, epsilon: float
) -> float:
    source_boundary = (source <= epsilon) | (source >= 1.0 - epsilon)
    output_boundary = (output <= epsilon) | (output >= 1.0 - epsilon)
    return float(np.mean(output_boundary & ~source_boundary))


def _bootstrap_lower(
    global_errors: np.ndarray,
    ridge_errors: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(global_errors), size=(resamples, len(global_errors))
    )
    values = np.mean(
        global_errors[indices] - ridge_errors[indices], axis=1
    )
    return float(np.quantile(values, 0.025))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_rmse": float(np.mean(array)),
        "median_rmse": float(np.median(array)),
        "p95_rmse": float(np.quantile(array, 0.95)),
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
            / str(development_config["source_evidence"]["manifest"])
        ).read_text(encoding="utf-8")
    )
    source_by_id = {
        row["pair_id"]: row for row in source_manifest["rows"]
    }
    descriptors = []
    parameters = []
    maximum_side = int(config["confirmation"]["maximum_side"])
    # The confirmation targets are not opened until this model is fitted.
    for row in development_rows:
        source_row = source_by_id[row["pair_id"]]
        source, _ = load_raw_default(
            root / source_row["raw_path"], maximum_side
        )
        source = np.clip(source, 0.0, 1.0)
        descriptors.append(source_descriptor(source, development_config))
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
        validated["fixed_config"], validated["fixed_validated"]
    )
    methods = [str(value) for value in config["methods"]]
    neutral_errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    fit_success: list[bool] = []
    rows = []
    epsilon = float(
        validated["fixed_validated"]["ao6_config"]["factorization"][
            "hard_boundary_epsilon_encoded_srgb"
        ]
    )
    for evidence in validated["confirmation_manifest"]["rows"]:
        source_path = root / evidence["source_path"]
        target_path = root / evidence["target_path"]
        if (
            _sha256(source_path) != evidence["source_sha256"]
            or _sha256(target_path) != evidence["target_sha256"]
        ):
            raise FiveKUnseenContentConfirmationError(
                f"confirmation asset drift: {evidence['pair_id']}"
            )
        source = _load_srgb16(source_path, maximum_side)
        target = _load_srgb16(target_path, maximum_side)
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
        target_style = _median_delta_e76(target, target_look)
        record: dict[str, Any] = {
            "pair_id": evidence["pair_id"],
            "fit_success": success,
            "target_style_delta_e76": target_style,
        }
        for method in methods:
            neutral = apply_neutral_base(source, parameter_map[method])
            look, diagnostics = renderer(neutral)
            neutral_error = _rmse(neutral, target)
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(neutral, look)
            boundary = max(
                _new_boundary_fraction(source, neutral, epsilon),
                _new_boundary_fraction(neutral, look, epsilon),
            )
            neutral_errors[method].append(neutral_error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            boundaries[method].append(boundary)
            record[method] = {
                "parameters": parameter_map[method].tolist(),
                "neutral_rmse": neutral_error,
                "look_rmse": look_error,
                "style_delta_e76": style,
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
    seed = int(evaluation["bootstrap_seed"])
    resamples = int(evaluation["bootstrap_resamples"])
    target_style_median = float(
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
            (np.mean(neutral_global) - np.mean(neutral_ridge))
            / max(np.mean(neutral_global), 1e-12)
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
            seed=seed,
            resamples=resamples,
        ),
        "look_ridge_mean_improvement_over_global": float(
            (np.mean(look_global) - np.mean(look_ridge))
            / max(np.mean(look_global), 1e-12)
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
            seed=seed + 1,
            resamples=resamples,
        ),
        "target_median_style_delta_e76": target_style_median,
        "ridge_to_target_median_style_ratio": float(
            metrics["ridge"]["median_style_delta_e76"]
            / max(target_style_median, 1e-12)
        ),
        "maximum_new_boundary_fraction": float(
            max(
                metrics[method]["maximum_new_boundary_fraction"]
                for method in methods
            )
        ),
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
        "look_wins": observed[
            "look_ridge_win_fraction_over_global"
        ]
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
        "development_report_sha256": config["development"][
            "report_sha256"
        ],
        "confirmation_manifest_sha256": config["confirmation"][
            "manifest_sha256"
        ],
        "row_count": len(rows),
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
    "FiveKUnseenContentConfirmationError",
    "run_confirmation",
    "validate_contract",
]
