"""Grouped explicit-operator explainability for the AO0 Velvia chart proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2lab

from src.roll2film.adaptive_density_strength import srgb8_to_linear
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _linear_to_encoded(linear: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(
        bounded <= 0.0031308,
        12.92 * bounded,
        1.055 * np.power(bounded, 1.0 / 2.4) - 0.055,
    )


def load_exact_pairs(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    source_config = config["source"]
    if _sha256(raw) != source_config["ao0_paired_patches_json_sha256"]:
        raise ValueError("AO1 paired-patches JSON hash mismatch")
    payload = json.loads(raw)
    if (
        payload.get("schema") != "neuro-film.u5.r2ao0.paired-patches.v1"
        or payload.get("source_asset_sha256") != source_config["asset_sha256"]
        or payload.get("paired_patch_u8_sha256")
        != source_config["paired_patch_u8_sha256"]
        or payload.get("paired_patch_channel_order")
        != source_config["paired_patch_channel_order"]
    ):
        raise ValueError("AO1 paired-patches lineage mismatch")
    reference_u8 = np.asarray(payload["reference_patch_rgb_u8"])
    film_u8 = np.asarray(payload["film_patch_rgb_u8"])
    if (
        reference_u8.dtype.kind not in "iu"
        or film_u8.dtype.kind not in "iu"
        or reference_u8.shape != (24, 3)
        or film_u8.shape != (24, 3)
        or np.any(reference_u8 < 0)
        or np.any(reference_u8 > 255)
        or np.any(film_u8 < 0)
        or np.any(film_u8 > 255)
    ):
        raise ValueError("AO1 paired patches must be 24 finite RGB8 rows")
    paired = np.concatenate(
        (film_u8.astype(np.uint8), reference_u8.astype(np.uint8)), axis=1
    )
    if _sha256(paired.tobytes()) != source_config["paired_patch_u8_sha256"]:
        raise ValueError("AO1 paired patch byte identity mismatch")
    reference = srgb8_to_linear(reference_u8.astype(np.uint8).reshape(24, 1, 3))
    film = srgb8_to_linear(film_u8.astype(np.uint8).reshape(24, 1, 3))
    return reference.reshape(24, 3), film.reshape(24, 3)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    raw = np.asarray(prediction, dtype=np.float64)
    error = raw - target
    encoded = _linear_to_encoded(raw)
    target_encoded = _linear_to_encoded(target)
    delta = np.linalg.norm(
        rgb2lab(encoded.reshape(-1, 1, 3)).reshape(-1, 3)
        - rgb2lab(target_encoded.reshape(-1, 1, 3)).reshape(-1, 3),
        axis=1,
    )
    return {
        "rgb_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "rgb_maximum_absolute_error": float(np.max(np.abs(error))),
        "mean_delta_e76": float(np.mean(delta)),
        "maximum_delta_e76": float(np.max(delta)),
        "raw_out_of_cube_fraction": float(np.mean((raw < 0.0) | (raw > 1.0))),
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
    }


def _fit_affine(
    source: np.ndarray, target: np.ndarray, *, per_channel: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    if per_channel:
        parameters = np.stack(
            [
                np.linalg.lstsq(
                    np.column_stack((source[:, channel], np.ones(len(source)))),
                    target[:, channel],
                    rcond=None,
                )[0]
                for channel in range(3)
            ]
        )
        matrix = np.diag(parameters[:, 0])
        bias = parameters[:, 1]
    else:
        design = np.column_stack((source, np.ones(len(source))))
        parameters = np.linalg.lstsq(design, target, rcond=None)[0]
        matrix = parameters[:3].T
        bias = parameters[3]
    return (
        source @ matrix.T + bias,
        {
            "matrix": matrix.tolist(),
            "bias": bias.tolist(),
            "determinant": float(np.linalg.det(matrix)),
        },
    )


def _fit_record(result: PositiveFilmFitResult) -> dict[str, Any]:
    operator = result.operator.to_dict()
    return {
        "model": result.model,
        "converged": result.converged,
        "development_rgb_rmse": result.development_rgb_rmse,
        "development_maximum_absolute_error": (
            result.development_maximum_absolute_error
        ),
        "function_evaluations": result.function_evaluations,
        "restart_index": result.restart_index,
        "operator_sha256": _sha256(_canonical_json(operator)),
        "operator": operator,
    }


def evaluate_chart_proxy(
    source: np.ndarray, target: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.shape != (24, 3)
        or target.shape != source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(target < 0.0)
        or np.any(target > 1.0)
    ):
        raise ValueError("AO1 requires finite [0,1] 24x3 paired arrays")
    source_before = source.copy()
    models = config["models"]
    fit_options = {
        "identity_mixture": float(models["matrix_identity_mixture"]),
        "restart_count": int(models["restart_count"]),
        "maximum_function_evaluations": int(
            models["maximum_function_evaluations"]
        ),
        "function_tolerance": float(models["function_tolerance"]),
        "parameter_tolerance": float(models["parameter_tolerance"]),
        "gradient_tolerance": float(models["gradient_tolerance"]),
        "loss": str(models["loss"]),
        "seed": int(models["seed"]),
    }
    folds: list[dict[str, Any]] = []
    model_names = ("identity", "per_channel_affine", "full_affine", "one_matrix", "two_matrix")
    confirmation_rmse: dict[str, list[float]] = {name: [] for name in model_names}
    confirmation_delta_e76: dict[str, list[float]] = {
        name: [] for name in model_names
    }
    development_rmse: dict[str, list[float]] = {
        name: [] for name in model_names if name != "identity"
    }
    all_converged = True
    all_positive_in_cube = True

    for held_row in range(6):
        confirmation_indices = np.arange(held_row * 4, held_row * 4 + 4)
        development_indices = np.setdiff1d(np.arange(24), confirmation_indices)
        development_source = source[development_indices]
        development_target = target[development_indices]

        _, channel_parameters = _fit_affine(
            development_source, development_target, per_channel=True
        )
        _, affine_parameters = _fit_affine(
            development_source, development_target, per_channel=False
        )
        channel_prediction = (
            source @ np.asarray(channel_parameters["matrix"]).T
            + np.asarray(channel_parameters["bias"])
        )
        affine_prediction = (
            source @ np.asarray(affine_parameters["matrix"]).T
            + np.asarray(affine_parameters["bias"])
        )
        one = fit_positive_film_response_operator(
            development_source, development_target, model="one_matrix", **fit_options
        )
        two = fit_positive_film_response_operator(
            development_source, development_target, model="two_matrix", **fit_options
        )
        predictions = {
            "identity": source,
            "per_channel_affine": channel_prediction,
            "full_affine": affine_prediction,
            "one_matrix": one.operator.apply(source),
            "two_matrix": two.operator.apply(source),
        }
        fold_models: dict[str, Any] = {}
        for name, prediction in predictions.items():
            confirmation = _metrics(
                prediction[confirmation_indices], target[confirmation_indices]
            )
            confirmation_rmse[name].append(confirmation["rgb_rmse"])
            confirmation_delta_e76[name].append(confirmation["mean_delta_e76"])
            record: dict[str, Any] = {"confirmation": confirmation}
            if name != "identity":
                development = _metrics(
                    prediction[development_indices], target[development_indices]
                )
                development_rmse[name].append(development["rgb_rmse"])
                record["development"] = development
            fold_models[name] = record
        fold_models["per_channel_affine"]["parameters"] = channel_parameters
        fold_models["full_affine"]["parameters"] = affine_parameters
        fold_models["one_matrix"]["fit"] = _fit_record(one)
        fold_models["two_matrix"]["fit"] = _fit_record(two)
        all_converged = all_converged and one.converged and two.converged
        for name in ("one_matrix", "two_matrix"):
            metrics = fold_models[name]["confirmation"]
            all_positive_in_cube = (
                all_positive_in_cube
                and metrics["raw_minimum"] >= -1e-12
                and metrics["raw_maximum"] <= 1.0 + 1e-12
            )
        folds.append(
            {
                "held_chart_row": held_row,
                "development_indices": development_indices.tolist(),
                "confirmation_indices": confirmation_indices.tolist(),
                "models": fold_models,
            }
        )

    aggregate = {
        name: {
            "mean_confirmation_rgb_rmse": float(np.mean(values)),
            "maximum_fold_confirmation_rgb_rmse": float(np.max(values)),
            "mean_fold_confirmation_delta_e76": float(
                np.mean(confirmation_delta_e76[name])
            ),
            "maximum_fold_mean_delta_e76": float(
                np.max(confirmation_delta_e76[name])
            ),
        }
        for name, values in confirmation_rmse.items()
    }
    for name, values in development_rmse.items():
        aggregate[name]["mean_development_rgb_rmse"] = float(np.mean(values))
        aggregate[name]["mean_train_test_rgb_rmse_gap"] = (
            aggregate[name]["mean_confirmation_rgb_rmse"]
            - aggregate[name]["mean_development_rgb_rmse"]
        )
    one_rmse = aggregate["one_matrix"]["mean_confirmation_rgb_rmse"]
    two_rmse = aggregate["two_matrix"]["mean_confirmation_rgb_rmse"]
    identity_rmse = aggregate["identity"]["mean_confirmation_rgb_rmse"]
    affine_rmse = aggregate["full_affine"]["mean_confirmation_rgb_rmse"]
    aggregate["one_matrix"]["gain_over_identity"] = 1.0 - one_rmse / identity_rmse
    aggregate["one_matrix"]["gain_over_full_affine"] = 1.0 - one_rmse / affine_rmse
    aggregate["one_matrix"]["rmse_ratio_to_two_matrix"] = one_rmse / two_rmse

    gates = config["gates"]
    checks = [
        {"name": "all_six_folds_present", "passed": len(folds) == 6},
        {"name": "all_positive_film_fits_converged", "passed": all_converged},
        {
            "name": "one_matrix_mean_confirmation_rgb_rmse",
            "passed": one_rmse
            <= float(gates["one_matrix_mean_confirmation_rgb_rmse_maximum"]),
        },
        {
            "name": "one_matrix_maximum_fold_rgb_rmse",
            "passed": aggregate["one_matrix"]["maximum_fold_confirmation_rgb_rmse"]
            <= float(gates["one_matrix_maximum_fold_rgb_rmse"]),
        },
        {
            "name": "one_matrix_gain_over_identity",
            "passed": aggregate["one_matrix"]["gain_over_identity"]
            >= float(gates["one_matrix_minimum_gain_over_identity"]),
        },
        {
            "name": "one_matrix_gain_over_full_affine",
            "passed": aggregate["one_matrix"]["gain_over_full_affine"]
            >= float(gates["one_matrix_minimum_gain_over_full_affine"]),
        },
        {
            "name": "one_matrix_capacity_sufficiency",
            "passed": aggregate["one_matrix"]["rmse_ratio_to_two_matrix"]
            <= float(gates["one_matrix_rmse_ratio_to_two_matrix_maximum"]),
        },
        {
            "name": "one_matrix_train_test_gap",
            "passed": aggregate["one_matrix"]["mean_train_test_rgb_rmse_gap"]
            <= float(gates["one_matrix_train_test_rmse_gap_maximum"]),
        },
        {"name": "positive_film_outputs_in_cube", "passed": all_positive_in_cube},
        {
            "name": "source_nonmutation",
            "passed": np.array_equal(source, source_before),
        },
    ]
    automatic_pass = all(bool(check["passed"]) for check in checks)
    return {
        "folds": folds,
        "aggregate": aggregate,
        "automatic_checks": checks,
        "automatic_pass": automatic_pass,
        "selected_model": "one_matrix" if automatic_pass else "none",
        "decision": (
            "retain_one_matrix_as_small_chart_display_proxy_champion"
            if automatic_pass
            else "close_positive_film_family_on_this_chart_proxy"
        ),
    }


__all__ = ["evaluate_chart_proxy", "load_exact_pairs"]
