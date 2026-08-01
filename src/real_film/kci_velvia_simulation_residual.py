"""Controlled real-Velvia versus digital-simulation chart residual pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _extract_grid(path: Path, spec: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw = path.read_bytes()
    if len(raw) != int(spec["bytes"]) or _sha256(raw) != spec["sha256"]:
        raise ValueError("BR0 plot byte identity mismatch")
    with Image.open(path) as image:
        if image.size != (int(spec["width"]), int(spec["height"])):
            raise ValueError("BR0 plot dimensions mismatch")
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)

    extraction = config["extraction"]
    x0 = int(extraction["chart_horizontal_start"])
    x1 = int(extraction["chart_horizontal_end_inclusive"])
    y0 = int(extraction["chart_vertical_start"])
    y1 = int(extraction["chart_vertical_end_inclusive"])
    rows = int(extraction["rows"])
    columns = int(extraction["columns"])
    half_width = int(extraction["centre_half_width"])
    half_height = int(extraction["centre_half_height"])
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    centres = [
        (
            round(x0 + (column + 0.5) * width / columns),
            round(y0 + (row + 0.5) * height / rows),
        )
        for row in range(rows)
        for column in range(columns)
    ]
    values: list[np.ndarray] = []
    maximum_range = 0
    for centre_x, centre_y in centres:
        block = pixels[
            centre_y - half_height : centre_y + half_height + 1,
            centre_x - half_width : centre_x + half_width + 1,
        ]
        if block.shape != (2 * half_height + 1, 2 * half_width + 1, 3):
            raise ValueError("BR0 chart sample escapes plot image")
        maximum_range = max(
            maximum_range,
            int(np.max(np.ptp(block.astype(np.int16), axis=(0, 1)))),
        )
        values.append(np.median(block, axis=(0, 1)).astype(np.uint8))
    return np.asarray(values, dtype=np.uint8), maximum_range


def extract_exact_pair(
    digital_path: Path,
    film_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Re-extract and verify the two exact embedded ColorChecker thumbnails."""

    source = config["source"]
    extraction = config["extraction"]
    digital, digital_range = _extract_grid(
        digital_path, source["digital_plot_png"], config
    )
    film, film_range = _extract_grid(film_path, source["film_plot_png"], config)
    expected_digital = np.asarray(extraction["digital_patch_rgb_u8"], dtype=np.uint8)
    expected_film = np.asarray(extraction["film_patch_rgb_u8"], dtype=np.uint8)
    paired = np.concatenate((digital, film), axis=1)
    checks = [
        digital.shape == (24, 3),
        film.shape == (24, 3),
        _sha256(digital.tobytes())
        == extraction["expected_digital_patch_u8_sha256"],
        _sha256(film.tobytes()) == extraction["expected_film_patch_u8_sha256"],
        _sha256(paired.tobytes()) == extraction["expected_paired_patch_u8_sha256"],
        np.array_equal(digital, expected_digital),
        np.array_equal(film, expected_film),
        max(digital_range, film_range)
        <= int(extraction["maximum_centre_channel_range"]),
    ]
    if not all(checks):
        raise ValueError("BR0 exact chart extraction failed")
    return {
        "digital_rgb_u8": digital,
        "film_rgb_u8": film,
        "digital_maximum_centre_channel_range": digital_range,
        "film_maximum_centre_channel_range": film_range,
        "digital_patch_u8_sha256": _sha256(digital.tobytes()),
        "film_patch_u8_sha256": _sha256(film.tobytes()),
        "paired_patch_u8_sha256": _sha256(paired.tobytes()),
    }


def _fit_models(source: np.ndarray, target: np.ndarray) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    denominator = float(np.sum(np.square(source)))
    scale = max(0.0, float(np.sum(source * target)) / denominator)
    axis_scale = np.maximum(
        0.0,
        np.sum(source * target, axis=0) / np.sum(np.square(source), axis=0),
    )
    matrix = np.linalg.lstsq(source, target, rcond=None)[0].T
    affine_parameters = np.linalg.lstsq(
        np.column_stack((source, np.ones(len(source)))), target, rcond=None
    )[0]
    affine_matrix = affine_parameters[:2].T
    affine_bias = affine_parameters[2]
    return {
        "global_chroma": (source * scale, {"scale": scale}),
        "axis_scale": (source * axis_scale, {"scale": axis_scale.tolist()}),
        "full_matrix": (
            source @ matrix.T,
            {
                "matrix": matrix.tolist(),
                "determinant": float(np.linalg.det(matrix)),
                "condition_number": float(np.linalg.cond(matrix)),
            },
        ),
        "affine_ceiling": (
            source @ affine_matrix.T + affine_bias,
            {"matrix": affine_matrix.tolist(), "bias": affine_bias.tolist()},
        ),
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(prediction - target), axis=1))))


def evaluate_residual(
    digital_rgb_u8: np.ndarray,
    film_rgb_u8: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate held-column neutral-corrected chroma models."""

    digital = np.asarray(digital_rgb_u8)
    film = np.asarray(film_rgb_u8)
    if (
        digital.shape != (24, 3)
        or film.shape != (24, 3)
        or digital.dtype.kind not in "iu"
        or film.dtype.kind not in "iu"
        or np.any(digital < 0)
        or np.any(digital > 255)
        or np.any(film < 0)
        or np.any(film > 255)
    ):
        raise ValueError("BR0 requires two RGB8 24x3 arrays")
    digital_before = digital.copy()
    film_before = film.copy()
    digital_lab = rgb2lab(digital.astype(np.float64).reshape(4, 6, 3) / 255.0).reshape(24, 3)
    film_lab = rgb2lab(film.astype(np.float64).reshape(4, 6, 3) / 255.0).reshape(24, 3)
    chromatic = np.asarray(config["extraction"]["chromatic_patch_indices"], dtype=np.int64)
    neutral = np.asarray(config["extraction"]["neutral_patch_indices"], dtype=np.int64)
    digital_neutral = np.mean(digital_lab[neutral, 1:3], axis=0)
    film_neutral = np.mean(film_lab[neutral, 1:3], axis=0)
    source = digital_lab[chromatic, 1:3] - digital_neutral
    target = film_lab[chromatic, 1:3] - film_neutral
    source_before = source.copy()
    model_names = ("identity", "global_chroma", "axis_scale", "full_matrix", "affine_ceiling")
    confirmation: dict[str, list[float]] = {name: [] for name in model_names}
    development: dict[str, list[float]] = {name: [] for name in model_names[1:]}
    folds: list[dict[str, Any]] = []
    determinants: list[float] = []
    conditions: list[float] = []

    for fold_index, held in enumerate(config["split"]["confirmation_indices_by_fold"]):
        confirmation_indices = np.asarray(held, dtype=np.int64)
        development_indices = np.setdiff1d(np.arange(18), confirmation_indices)
        fitted = _fit_models(source[development_indices], target[development_indices])
        predictions = {"identity": source}
        parameters: dict[str, Any] = {}
        for name, (_, model_parameters) in fitted.items():
            parameters[name] = model_parameters
            if name == "global_chroma":
                predictions[name] = source * float(model_parameters["scale"])
            elif name == "axis_scale":
                predictions[name] = source * np.asarray(model_parameters["scale"])
            else:
                predictions[name] = source @ np.asarray(model_parameters["matrix"]).T
                if name == "affine_ceiling":
                    predictions[name] += np.asarray(model_parameters["bias"])
        determinants.append(float(parameters["full_matrix"]["determinant"]))
        conditions.append(float(parameters["full_matrix"]["condition_number"]))
        fold_models: dict[str, Any] = {}
        for name, prediction in predictions.items():
            confirmation_rmse = _rmse(
                prediction[confirmation_indices], target[confirmation_indices]
            )
            confirmation[name].append(confirmation_rmse)
            record: dict[str, Any] = {"confirmation_chroma_rmse": confirmation_rmse}
            if name != "identity":
                development_rmse = _rmse(
                    prediction[development_indices], target[development_indices]
                )
                development[name].append(development_rmse)
                record["development_chroma_rmse"] = development_rmse
                record["parameters"] = parameters[name]
            fold_models[name] = record
        folds.append(
            {
                "fold": fold_index,
                "development_indices": development_indices.tolist(),
                "confirmation_indices": confirmation_indices.tolist(),
                "models": fold_models,
            }
        )

    aggregate = {
        name: {
            "mean_confirmation_chroma_rmse": float(np.mean(values)),
            "maximum_fold_confirmation_chroma_rmse": float(np.max(values)),
        }
        for name, values in confirmation.items()
    }
    for name, values in development.items():
        aggregate[name]["mean_development_chroma_rmse"] = float(np.mean(values))
        aggregate[name]["mean_train_test_chroma_rmse_gap"] = (
            aggregate[name]["mean_confirmation_chroma_rmse"]
            - aggregate[name]["mean_development_chroma_rmse"]
        )
    full_rmse = aggregate["full_matrix"]["mean_confirmation_chroma_rmse"]
    global_rmse = aggregate["global_chroma"]["mean_confirmation_chroma_rmse"]
    affine_rmse = aggregate["affine_ceiling"]["mean_confirmation_chroma_rmse"]
    aggregate["full_matrix"].update(
        {
            "gain_over_global_chroma": 1.0 - full_rmse / global_rmse,
            "rmse_ratio_to_affine_ceiling": full_rmse / affine_rmse,
            "minimum_fold_determinant": float(np.min(determinants)),
            "maximum_fold_condition_number": float(np.max(conditions)),
        }
    )
    gates = config["gates"]
    checks = [
        {"name": "all_six_folds_present", "passed": len(folds) == 6},
        {
            "name": "identity_signal",
            "passed": aggregate["identity"]["mean_confirmation_chroma_rmse"]
            >= float(gates["identity_mean_confirmation_chroma_rmse_minimum"]),
        },
        {
            "name": "full_matrix_gain_over_global_chroma",
            "passed": aggregate["full_matrix"]["gain_over_global_chroma"]
            >= float(gates["full_matrix_minimum_gain_over_global_chroma"]),
        },
        {
            "name": "full_matrix_mean_confirmation_chroma_rmse",
            "passed": full_rmse
            <= float(gates["full_matrix_mean_confirmation_chroma_rmse_maximum"]),
        },
        {
            "name": "full_matrix_maximum_fold_chroma_rmse",
            "passed": aggregate["full_matrix"]["maximum_fold_confirmation_chroma_rmse"]
            <= float(gates["full_matrix_maximum_fold_chroma_rmse"]),
        },
        {
            "name": "full_matrix_train_test_chroma_rmse_gap",
            "passed": aggregate["full_matrix"]["mean_train_test_chroma_rmse_gap"]
            <= float(gates["full_matrix_train_test_chroma_rmse_gap_maximum"]),
        },
        {
            "name": "full_matrix_capacity_sufficiency",
            "passed": aggregate["full_matrix"]["rmse_ratio_to_affine_ceiling"]
            <= float(gates["full_matrix_rmse_ratio_to_affine_ceiling_maximum"]),
        },
        {
            "name": "full_matrix_positive_determinant",
            "passed": aggregate["full_matrix"]["minimum_fold_determinant"]
            >= float(gates["full_matrix_determinant_minimum"]),
        },
        {
            "name": "full_matrix_condition_number",
            "passed": aggregate["full_matrix"]["maximum_fold_condition_number"]
            <= float(gates["full_matrix_condition_number_maximum"]),
        },
        {
            "name": "source_nonmutation",
            "passed": np.array_equal(digital, digital_before)
            and np.array_equal(film, film_before)
            and np.array_equal(source, source_before),
        },
    ]
    automatic_pass = all(bool(check["passed"]) for check in checks)
    return {
        "digital_neutral_ab": digital_neutral.tolist(),
        "film_neutral_ab": film_neutral.tolist(),
        "folds": folds,
        "aggregate": aggregate,
        "automatic_checks": checks,
        "automatic_pass": automatic_pass,
        "selected_model": "full_matrix" if automatic_pass else "none",
        "decision": (
            "retain_small_nonbasic_chroma_direction_for_photo_stress"
            if automatic_pass
            else "close_controlled_chart_residual_without_capacity_rescue"
        ),
    }


__all__ = ["evaluate_residual", "extract_exact_pair"]
