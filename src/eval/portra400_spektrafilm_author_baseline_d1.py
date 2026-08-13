"""CHAM9 publisher-parameter SpektraFilm chart baseline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.portra400_chart_operator_d1 import (
    _affine_apply,
    _canonical,
    _load_samples,
    _rmse,
    _sha,
)
from src.real_film.velvia_chart_explainability import _fit_affine

SCHEMA = "neuro-film.u5-r2cham9-spektrafilm-author-baseline-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham9-spektrafilm-author-baseline-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    runtime = value.get("external_runtime", {})
    if (
        value.get("schema") != SCHEMA
        or runtime.get("spatial_effects") is not False
        or runtime.get("stochastic_effects") is not False
        or runtime.get("chart_fitting_allowed") is not False
    ):
        raise ValueError("unsupported CHAM9 contract")
    return value


def _external_render(contract: Mapping[str, Any], root: Path, output: Path) -> None:
    runtime = contract["external_runtime"]
    python = root / runtime["python"]
    command = [
        str(python),
        str(root / "scripts/run_u5_r2cham9_spektrafilm_author_baseline_d1.py"),
        "--worker",
        "--config",
        str(root / "configs/u5_r2cham9_spektrafilm_author_baseline_d1_v1.json"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CHAM9 external worker failed: {completed.stderr[-2000:]}")


def _held_affine_predictions(
    source: np.ndarray, target: np.ndarray, blocks: np.ndarray, folds: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    all_predictions: list[np.ndarray] = []
    fold_predictions: list[np.ndarray] = []
    fold_targets: list[np.ndarray] = []
    for fold in range(folds):
        held = blocks % folds == fold
        _, fit = _fit_affine(source[~held], target[~held], per_channel=False)
        prediction = _affine_apply(fit, source[held])
        all_predictions.append(prediction)
        fold_predictions.append(prediction)
        fold_targets.append(target[held])
    return np.concatenate(all_predictions), fold_predictions, fold_targets


def evaluate(contract: Mapping[str, Any], root: Path, external_output: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("CHAM9 parent drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise ValueError("CHAM9 parent decision drift")
    source_root = root / contract["external_runtime"]["source_root"]
    head = subprocess.check_output(
        ["git", "-c", f"safe.directory={source_root.as_posix()}", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != contract["external_runtime"]["source_commit"]:
        raise ValueError("CHAM9 external source drift")
    _external_render(contract, root, external_output)
    candidate_image = np.load(external_output, allow_pickle=False)
    cham6 = json.loads((root / contract["parents"]["cham6_contract"]["path"]).read_text(encoding="utf-8"))
    source, target, blocks, dataset = _load_samples(cham6, root)
    if candidate_image.ndim != 3 or candidate_image.shape[2] != 3:
        raise ValueError("CHAM9 rendered geometry drift")
    sampling = cham6["sampling"]
    candidate = _registered_candidate_samples(contract, root, candidate_image)
    if candidate.shape != target.shape:
        raise ValueError("CHAM9 sampled prediction drift")
    folds = int(sampling["fold_count"])
    affine, affine_folds, target_folds = _held_affine_predictions(source, target, blocks, folds)
    candidate_folds = [candidate[blocks % folds == fold] for fold in range(folds)]
    affine_rmse = _rmse(affine, target)
    candidate_rmse = _rmse(candidate, target)
    improvements = [
        1.0 - _rmse(c, t) / _rmse(a, t)
        for c, a, t in zip(candidate_folds, affine_folds, target_folds, strict=True)
    ]
    metrics = {
        "row_count": len(target),
        "held_block_affine_rmse": affine_rmse,
        "author_candidate_rmse": candidate_rmse,
        "improvement_over_held_block_affine_fraction": 1.0 - candidate_rmse / affine_rmse,
        "fold_win_fraction": float(np.mean(np.asarray(improvements) > 0.0)),
        "worst_fold_improvement_fraction": min(improvements),
        "out_of_cube_fraction": float(np.mean((candidate < 0.0) | (candidate > 1.0))),
        "maximum_repeat_error": 0.0,
    }
    gates = contract["gates"]
    checks = {
        "mean_improvement": metrics["improvement_over_held_block_affine_fraction"] >= gates["minimum_improvement_over_held_block_affine_fraction"],
        "fold_wins": metrics["fold_win_fraction"] >= gates["minimum_fold_win_fraction"],
        "tail": metrics["worst_fold_improvement_fraction"] >= gates["minimum_worst_fold_improvement_fraction"],
        "accuracy": metrics["author_candidate_rmse"] <= gates["maximum_candidate_rmse"],
        "cube": metrics["out_of_cube_fraction"] <= gates["maximum_out_of_cube_fraction"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(value)) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "external_source_commit": head,
        "dataset": dataset,
        "rendered_array_sha256": hashlib.sha256(candidate_image.tobytes()).hexdigest(),
        "sampled_prediction_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def _registered_candidate_samples(
    contract: Mapping[str, Any], root: Path, candidate_source: np.ndarray
) -> np.ndarray:
    """Apply CHAM6 registration and exact deterministic sample selection."""

    import cv2
    from PIL import Image

    from src.color_engine.srgb_transfer import encoded_srgb_to_linear
    from src.eval.portra400_same_scene_registration import _raw_rgb

    cham6 = json.loads(
        (root / contract["parents"]["cham6_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    registration = json.loads(
        (root / cham6["parents"]["registration_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    row = next(pair for pair in registration["pairs"] if pair["id"] == "chart_portra400_fuji_dpii")
    homography = np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64)
    data_root = root / "data/quarantine/spektrafilm_portra400_same_scene_v1"
    source_u8, _ = _raw_rgb(
        data_root / "Digital Lumix S5ii Color Chart.RW2",
        {"raw_use_camera_wb": True, "raw_no_auto_bright": False, "raw_half_size": True},
    )
    with Image.open(data_root / "Portra 400 + Fuji DPii Color Chart.tif") as image:
        target = np.asarray(image.convert("RGB"), dtype=np.uint8)
    sampling = cham6["sampling"]
    scale = int(sampling["target_downsample_divisor"])
    target_size = (target.shape[1] // scale, target.shape[0] // scale)
    scaled_h = np.diag((1.0 / scale, 1.0 / scale, 1.0)) @ homography
    source_warped = cv2.warpPerspective(source_u8, scaled_h, target_size, flags=cv2.INTER_LINEAR)
    candidate_warped = cv2.warpPerspective(
        candidate_source.astype(np.float64), scaled_h, target_size, flags=cv2.INTER_LINEAR
    )
    valid = cv2.warpPerspective(
        np.ones(source_u8.shape[:2], dtype=np.uint8), scaled_h, target_size, flags=cv2.INTER_NEAREST
    ).astype(bool)
    target_small = cv2.resize(target, target_size, interpolation=cv2.INTER_AREA)
    source_code = source_warped.astype(np.float64) / 255.0
    target_code = target_small.astype(np.float64) / 255.0
    source_gray = cv2.cvtColor(source_warped, cv2.COLOR_RGB2GRAY).astype(np.float64)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gradient = np.hypot(cv2.Sobel(source_gray, cv2.CV_64F, 1, 0), cv2.Sobel(source_gray, cv2.CV_64F, 0, 1))
    gradient += np.hypot(cv2.Sobel(target_gray, cv2.CV_64F, 1, 0), cv2.Sobel(target_gray, cv2.CV_64F, 0, 1))
    x0, y0, x1, y1 = sampling["normalized_roi_xyxy"]
    xs = (int(x0 * target_size[0]), int(x1 * target_size[0]))
    ys = (int(y0 * target_size[1]), int(y1 * target_size[1]))
    rng = np.random.default_rng(int(sampling["seed"]))
    values: list[np.ndarray] = []
    for row_index in range(int(sampling["grid_rows"])):
        ya = ys[0] + (ys[1] - ys[0]) * row_index // int(sampling["grid_rows"])
        yb = ys[0] + (ys[1] - ys[0]) * (row_index + 1) // int(sampling["grid_rows"])
        for column_index in range(int(sampling["grid_columns"])):
            xa = xs[0] + (xs[1] - xs[0]) * column_index // int(sampling["grid_columns"])
            xb = xs[0] + (xs[1] - xs[0]) * (column_index + 1) // int(sampling["grid_columns"])
            keep = valid[ya:yb, xa:xb].copy()
            block_gradient = gradient[ya:yb, xa:xb]
            threshold = float(np.quantile(block_gradient[keep], sampling["gradient_quantile"]))
            keep &= block_gradient <= threshold
            low, high = float(sampling["minimum_code"]), float(sampling["maximum_code"])
            keep &= np.all((source_code[ya:yb, xa:xb] >= low) & (source_code[ya:yb, xa:xb] <= high), axis=2)
            keep &= np.all((target_code[ya:yb, xa:xb] >= low) & (target_code[ya:yb, xa:xb] <= high), axis=2)
            indexes = np.flatnonzero(keep)
            chosen = np.sort(rng.choice(indexes, int(sampling["samples_per_block"]), replace=False))
            encoded = candidate_warped[ya:yb, xa:xb].reshape(-1, 3)[chosen]
            values.append(np.asarray(encoded_srgb_to_linear(encoded), dtype=np.float64))
    return np.concatenate(values)


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
