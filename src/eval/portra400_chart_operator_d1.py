"""CHAM6 held-spatial-block Portra chart operator discriminant."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rawpy
from PIL import Image

from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.portra400_same_scene_registration import _raw_rgb
from src.real_film.velvia_chart_explainability import _fit_affine
from src.roll2film.emulating_emulsion_baseline import fit_emulating_emulsion_equation

SCHEMA = "neuro-film.u5-r2cham6-portra400-chart-operator-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham6-portra400-chart-operator-d1-result.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    sampling = value.get("sampling", {})
    if (
        value.get("schema") != SCHEMA
        or sampling.get("grid_rows") != 8
        or sampling.get("grid_columns") != 8
        or sampling.get("fold_count") != 4
        or value["models"].get("fit_domain") != "linear_srgb_from_exact_srgb_decodes"
        or value["models"]["candidate_parameterization"]["hard_output_clipping"]
    ):
        raise ValueError("unsupported CHAM6 contract")
    return value


def _candidate_options(contract: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    bounds = contract["models"]["candidate_parameterization"]
    fit = contract["models"]["fit"]
    return {
        "capture_matrix_entry_bounds": tuple(bounds["capture_matrix_entry_bounds"]),
        "scan_matrix_entry_bounds": tuple(bounds["scan_matrix_entry_bounds"]),
        "response_amplitude_bounds": tuple(bounds["response_amplitude_bounds"]),
        "response_slope_bounds": tuple(bounds["response_slope_bounds"]),
        "response_midpoint_bounds": tuple(bounds["response_midpoint_bounds"]),
        "response_offset_bounds": tuple(bounds["response_offset_bounds"]),
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(fit["maximum_function_evaluations"]),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "seed": seed,
    }


def _load_samples(
    contract: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    registration = json.loads(
        (root / contract["parents"]["registration_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    row = next(
        pair
        for pair in registration["pairs"]
        if pair["id"] == "chart_portra400_fuji_dpii"
    )
    homography = np.asarray(
        row["diagnostics"]["homography_digital_to_film"], dtype=np.float64
    )
    data_root = root / "data/quarantine/spektrafilm_portra400_same_scene_v1"
    raw_contract = {
        "raw_use_camera_wb": True,
        "raw_no_auto_bright": False,
        "raw_half_size": True,
    }
    source, source_meta = _raw_rgb(
        data_root / "Digital Lumix S5ii Color Chart.RW2", raw_contract
    )
    with Image.open(data_root / "Portra 400 + Fuji DPii Color Chart.tif") as image:
        target = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    scale = int(contract["sampling"]["target_downsample_divisor"])
    target_size = (target.shape[1] // scale, target.shape[0] // scale)
    scaled_h = np.diag((1.0 / scale, 1.0 / scale, 1.0)) @ homography
    warped = cv2.warpPerspective(source, scaled_h, target_size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(
        np.ones(source.shape[:2], dtype=np.uint8),
        scaled_h,
        target_size,
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    target_small = cv2.resize(target, target_size, interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = contract["sampling"]["normalized_roi_xyxy"]
    xs = (int(x0 * target_size[0]), int(x1 * target_size[0]))
    ys = (int(y0 * target_size[1]), int(y1 * target_size[1]))
    source_code = warped.astype(np.float64) / 255.0
    target_code = target_small.astype(np.float64) / 255.0
    source_f = np.asarray(encoded_srgb_to_linear(source_code), dtype=np.float64)
    target_f = np.asarray(encoded_srgb_to_linear(target_code), dtype=np.float64)
    source_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float64)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gradient = np.hypot(
        cv2.Sobel(source_gray, cv2.CV_64F, 1, 0),
        cv2.Sobel(source_gray, cv2.CV_64F, 0, 1),
    )
    gradient += np.hypot(
        cv2.Sobel(target_gray, cv2.CV_64F, 1, 0),
        cv2.Sobel(target_gray, cv2.CV_64F, 0, 1),
    )
    rows = int(contract["sampling"]["grid_rows"])
    columns = int(contract["sampling"]["grid_columns"])
    per_block = int(contract["sampling"]["samples_per_block"])
    rng = np.random.default_rng(int(contract["sampling"]["seed"]))
    source_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    block_ids: list[np.ndarray] = []
    counts: list[int] = []
    for row_index in range(rows):
        ya = ys[0] + (ys[1] - ys[0]) * row_index // rows
        yb = ys[0] + (ys[1] - ys[0]) * (row_index + 1) // rows
        for column_index in range(columns):
            xa = xs[0] + (xs[1] - xs[0]) * column_index // columns
            xb = xs[0] + (xs[1] - xs[0]) * (column_index + 1) // columns
            block_valid = valid[ya:yb, xa:xb].copy()
            block_gradient = gradient[ya:yb, xa:xb]
            threshold = float(
                np.quantile(
                    block_gradient[block_valid],
                    contract["sampling"]["gradient_quantile"],
                )
            )
            block_valid &= block_gradient <= threshold
            block_source_code = source_code[ya:yb, xa:xb]
            block_target_code = target_code[ya:yb, xa:xb]
            block_source = source_f[ya:yb, xa:xb]
            block_target = target_f[ya:yb, xa:xb]
            low = float(contract["sampling"]["minimum_code"])
            high = float(contract["sampling"]["maximum_code"])
            block_valid &= np.all(
                (block_source_code >= low) & (block_source_code <= high), axis=2
            )
            block_valid &= np.all(
                (block_target_code >= low) & (block_target_code <= high), axis=2
            )
            indexes = np.flatnonzero(block_valid)
            if len(indexes) < per_block:
                raise ValueError(
                    f"CHAM6 insufficient block support: {row_index},{column_index}"
                )
            chosen = np.sort(rng.choice(indexes, per_block, replace=False))
            source_samples.append(block_source.reshape(-1, 3)[chosen])
            target_samples.append(block_target.reshape(-1, 3)[chosen])
            block = row_index * columns + column_index
            block_ids.append(np.full(per_block, block, dtype=np.int64))
            counts.append(len(indexes))
    return (
        np.concatenate(source_samples),
        np.concatenate(target_samples),
        np.concatenate(block_ids),
        {
            "raw_decode_sha256": source_meta["decoded_sha256"],
            "target_decoded_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            "sample_count": sum(len(row) for row in source_samples),
            "minimum_eligible_samples_per_block": min(counts),
            "roi_pixels_xyxy": [xs[0], ys[0], xs[1], ys[1]],
            "downsampled_target_shape": list(target_small.shape),
        },
    )


def _rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate - target))))


def _affine_apply(fit: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    return values @ np.asarray(fit["matrix"], dtype=np.float64).T + np.asarray(
        fit["bias"], dtype=np.float64
    )


def _cube(size: int = 7) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("CHAM6 parent drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            "required_decision" in binding
            and payload.get("decision") != binding["required_decision"]
        ):
            raise ValueError("CHAM6 parent decision drift")
        if (
            "required_stable_evidence_id" in binding
            and payload.get("stable_evidence_id")
            != binding["required_stable_evidence_id"]
        ):
            raise ValueError("CHAM6 parent stable identity drift")
    source, target, blocks, dataset = _load_samples(contract, root)
    fold_count = int(contract["sampling"]["fold_count"])
    fits: list[dict[str, Any]] = []
    all_target: list[np.ndarray] = []
    predictions = {
        name: []
        for name in (
            "identity",
            "full_affine",
            "candidate",
            "correspondence_shuffled_candidate",
        )
    }
    cube = _cube()
    jacobian_minima: list[float] = []
    repeat_errors: list[float] = []
    for fold in range(fold_count):
        held = blocks % fold_count == fold
        development = ~held
        _, affine = _fit_affine(
            source[development], target[development], per_channel=False
        )
        candidate = fit_emulating_emulsion_equation(
            source[development],
            target[development],
            **_candidate_options(
                contract, seed=int(contract["models"]["fit"]["seed"]) + fold
            ),
        )
        shifted = np.roll(
            target[development],
            int(contract["models"]["fit"]["wrong_target_roll"]),
            axis=0,
        )
        wrong = fit_emulating_emulsion_equation(
            source[development],
            shifted,
            **_candidate_options(
                contract, seed=int(contract["models"]["fit"]["seed"]) + 100 + fold
            ),
        )
        held_source = source[held]
        held_target = target[held]
        all_target.append(held_target)
        predictions["identity"].append(held_source)
        predictions["full_affine"].append(_affine_apply(affine, held_source))
        candidate_prediction = candidate.operator.apply(held_source)
        candidate_replay = candidate.operator.apply(held_source)
        predictions["candidate"].append(candidate_prediction)
        predictions["correspondence_shuffled_candidate"].append(
            wrong.operator.apply(held_source)
        )
        repeat_errors.append(
            float(np.max(np.abs(candidate_prediction - candidate_replay)))
        )
        determinant = candidate.operator.jacobian_determinants(cube)
        jacobian_minima.append(float(np.min(determinant)))
        fits.append(
            {
                "fold": fold,
                "held_blocks": sorted({int(value) for value in blocks[held]}),
                "held_rows": int(np.count_nonzero(held)),
                "candidate_development_rmse": candidate.development_rgb_rmse,
                "candidate_converged": candidate.converged,
                "candidate_minimum_cube_jacobian_determinant": jacobian_minima[-1],
            }
        )
    combined_target = np.concatenate(all_target)
    combined = {name: np.concatenate(rows) for name, rows in predictions.items()}
    rmse = {name: _rmse(value, combined_target) for name, value in combined.items()}
    fold_candidate = [
        _rmse(predictions["candidate"][i], all_target[i]) for i in range(fold_count)
    ]
    fold_affine = [
        _rmse(predictions["full_affine"][i], all_target[i]) for i in range(fold_count)
    ]
    fold_improvement = [
        1.0 - candidate / affine
        for candidate, affine in zip(fold_candidate, fold_affine, strict=True)
    ]
    candidate = combined["candidate"]
    metrics = {
        "row_count": len(combined_target),
        "identity_rmse": rmse["identity"],
        "full_affine_rmse": rmse["full_affine"],
        "candidate_rmse": rmse["candidate"],
        "correspondence_shuffled_candidate_rmse": rmse[
            "correspondence_shuffled_candidate"
        ],
        "improvement_over_full_affine_fraction": 1.0
        - rmse["candidate"] / rmse["full_affine"],
        "fold_win_fraction_over_full_affine": float(
            np.mean(np.asarray(fold_candidate) < np.asarray(fold_affine))
        ),
        "worst_fold_improvement_over_full_affine_fraction": min(fold_improvement),
        "improvement_over_correspondence_shuffled_fraction": 1.0
        - rmse["candidate"] / rmse["correspondence_shuffled_candidate"],
        "raw_out_of_cube_fraction": float(
            np.mean((candidate < 0.0) | (candidate > 1.0))
        ),
        "minimum_cube_jacobian_determinant": min(jacobian_minima),
        "maximum_repeat_error": max(repeat_errors),
    }
    gates = contract["gates"]
    checks = {
        "support": dataset["minimum_eligible_samples_per_block"]
        >= gates["minimum_samples_per_block"],
        "mean_improvement": metrics["improvement_over_full_affine_fraction"]
        >= gates["minimum_improvement_over_full_affine_fraction"],
        "fold_wins": metrics["fold_win_fraction_over_full_affine"]
        >= gates["minimum_fold_win_fraction_over_full_affine"],
        "tail": metrics["worst_fold_improvement_over_full_affine_fraction"]
        >= gates["minimum_worst_fold_improvement_over_full_affine_fraction"],
        "correspondence": metrics["improvement_over_correspondence_shuffled_fraction"]
        >= gates["minimum_improvement_over_correspondence_shuffled_fraction"],
        "accuracy": metrics["candidate_rmse"] <= gates["maximum_candidate_rmse"],
        "cube": metrics["raw_out_of_cube_fraction"]
        <= gates["maximum_raw_out_of_cube_fraction"],
        "jacobian": metrics["minimum_cube_jacobian_determinant"]
        > gates["minimum_cube_jacobian_determinant_exclusive"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
    }
    finite = all(np.isfinite(float(value)) for value in metrics.values())
    checks["finite"] = finite
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "dataset": dataset,
        "folds": fits,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
        "rawpy_version": rawpy.__version__,
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
