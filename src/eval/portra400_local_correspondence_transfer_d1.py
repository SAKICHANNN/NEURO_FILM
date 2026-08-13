"""CHAM10 chart-affine transfer on natural local correspondences."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.portra400_chart_operator_d1 import (
    _affine_apply,
    _canonical,
    _load_samples,
    _rmse,
    _sha,
)
from src.eval.portra400_same_scene_registration import _raster_rgb, _raw_rgb
from src.real_film.velvia_chart_explainability import _fit_affine

SCHEMA = "neuro-film.u5-r2cham10-portra400-local-correspondence-transfer-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham10-portra400-local-correspondence-transfer-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    pairs = value.get("natural_pairs", [])
    if (
        value.get("schema") != SCHEMA
        or len(pairs) != 2
        or sum(bool(row.get("primary")) for row in pairs) != 1
        or value["correspondence"].get("spatial_fold_count") != 4
    ):
        raise ValueError("unsupported CHAM10 contract")
    return value


def _mutual_inliers(
    source: np.ndarray, target: np.ndarray, spec: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    clahe = cv2.createCLAHE(
        clipLimit=float(spec["clahe_clip_limit"]),
        tileGridSize=tuple(int(v) for v in spec["clahe_tile_grid"]),
    )
    src_gray = clahe.apply(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY))
    dst_gray = clahe.apply(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY))
    sift = cv2.SIFT_create(nfeatures=int(spec["maximum_features"]))
    src_kp, src_desc = sift.detectAndCompute(src_gray, None)
    dst_kp, dst_desc = sift.detectAndCompute(dst_gray, None)
    if src_desc is None or dst_desc is None:
        raise ValueError("CHAM10 missing descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ratio = float(spec["ratio_threshold"])
    forward = {
        m.queryIdx: m
        for m, n in matcher.knnMatch(src_desc, dst_desc, k=2)
        if m.distance < ratio * n.distance
    }
    reverse = {
        m.queryIdx: m
        for m, n in matcher.knnMatch(dst_desc, src_desc, k=2)
        if m.distance < ratio * n.distance
    }
    mutual = [
        m
        for query, m in sorted(forward.items())
        if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == query
    ]
    if len(mutual) < int(spec["minimum_mutual_good_matches"]):
        raise ValueError("CHAM10 insufficient mutual matches")
    src_xy = np.float32([src_kp[m.queryIdx].pt for m in mutual])
    dst_xy = np.float32([dst_kp[m.trainIdx].pt for m in mutual])
    homography, mask = cv2.findHomography(
        src_xy,
        dst_xy,
        cv2.RANSAC,
        float(spec["ransac_reprojection_threshold_pixels"]),
    )
    if homography is None or mask is None:
        raise ValueError("CHAM10 homography failed")
    keep = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src_xy[:, None], homography)[:, 0]
    errors = np.linalg.norm(projected - dst_xy, axis=1)[keep]
    return src_xy[keep], dst_xy[keep], {
        "mutual_matches": len(mutual),
        "inliers": int(np.count_nonzero(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "median_reprojection_error_pixels": float(np.median(errors)),
        "p95_reprojection_error_pixels": float(np.quantile(errors, 0.95)),
    }


def _patch_rows(
    source: np.ndarray,
    target: np.ndarray,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = int(spec["patch_radius_pixels"])
    low, high = float(spec["minimum_code"]), float(spec["maximum_code"])
    maximum_mad = float(spec["maximum_patch_channel_mad"])
    src_rows: list[np.ndarray] = []
    dst_rows: list[np.ndarray] = []
    folds: list[int] = []
    for src_point, dst_point in zip(src_xy, dst_xy, strict=True):
        sx, sy = (round(float(v)) for v in src_point)
        tx, ty = (round(float(v)) for v in dst_point)
        if (
            sx - radius < 0
            or sy - radius < 0
            or tx - radius < 0
            or ty - radius < 0
            or sx + radius >= source.shape[1]
            or sy + radius >= source.shape[0]
            or tx + radius >= target.shape[1]
            or ty + radius >= target.shape[0]
        ):
            continue
        src = source[sy - radius : sy + radius + 1, sx - radius : sx + radius + 1]
        dst = target[ty - radius : ty + radius + 1, tx - radius : tx + radius + 1]
        src_code = src.astype(np.float64) / 255.0
        dst_code = dst.astype(np.float64) / 255.0
        src_median = np.median(src_code.reshape(-1, 3), axis=0)
        dst_median = np.median(dst_code.reshape(-1, 3), axis=0)
        src_mad = np.median(np.abs(src_code.reshape(-1, 3) - src_median), axis=0)
        dst_mad = np.median(np.abs(dst_code.reshape(-1, 3) - dst_median), axis=0)
        if (
            np.any(src_median < low)
            or np.any(src_median > high)
            or np.any(dst_median < low)
            or np.any(dst_median > high)
            or np.any(src_mad > maximum_mad)
            or np.any(dst_mad > maximum_mad)
        ):
            continue
        src_rows.append(src_median)
        dst_rows.append(dst_median)
        folds.append(2 * int(ty >= target.shape[0] / 2) + int(tx >= target.shape[1] / 2))
    if not src_rows:
        raise ValueError("CHAM10 no eligible local correspondences")
    source_linear = np.asarray(encoded_srgb_to_linear(np.asarray(src_rows)[:, None]), dtype=np.float64)[:, 0]
    target_linear = np.asarray(encoded_srgb_to_linear(np.asarray(dst_rows)[:, None]), dtype=np.float64)[:, 0]
    return source_linear, target_linear, np.asarray(folds, dtype=np.int64)


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("CHAM10 parent drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise ValueError("CHAM10 parent decision drift")
    cham6 = json.loads((root / contract["parents"]["cham6_contract"]["path"]).read_text(encoding="utf-8"))
    chart_source, chart_target, _blocks, chart_data = _load_samples(cham6, root)
    _, affine = _fit_affine(chart_source, chart_target, per_channel=False)
    data_root = root / "data/quarantine/spektrafilm_portra400_same_scene_v1"
    raw_spec = {"raw_use_camera_wb": True, "raw_no_auto_bright": False, "raw_half_size": True}
    source, source_meta = _raw_rgb(data_root / "RAW.RW2", raw_spec)
    role_paths = {
        "same_scene_portra400_fuji_dpii_ra4_epson_scan": data_root / "RA4 Print.jpg",
        "same_scene_portra400_noritsu_scan": data_root / "Film Scan.jpg",
    }
    rows: list[dict[str, Any]] = []
    for pair in contract["natural_pairs"]:
        target, target_meta = _raster_rgb(role_paths[pair["target_role"]])
        src_xy, dst_xy, registration = _mutual_inliers(source, target, contract["correspondence"])
        natural_source, natural_target, folds = _patch_rows(source, target, src_xy, dst_xy, contract["correspondence"])
        prediction = _affine_apply(affine, natural_source)
        replay = _affine_apply(affine, natural_source)
        wrong = np.roll(natural_target, int(contract["correspondence"]["wrong_target_roll"]), axis=0)
        identity_rmse = _rmse(natural_source, natural_target)
        candidate_rmse = _rmse(prediction, natural_target)
        wrong_rmse = _rmse(prediction, wrong)
        fold_improvements = []
        for fold in range(int(contract["correspondence"]["spatial_fold_count"])):
            held = folds == fold
            if np.any(held):
                fold_improvements.append(1.0 - _rmse(prediction[held], natural_target[held]) / _rmse(natural_source[held], natural_target[held]))
        rows.append({
            "id": pair["id"],
            "primary": bool(pair["primary"]),
            "source_decoded_sha256": source_meta["decoded_sha256"],
            "target_decoded_sha256": target_meta["decoded_sha256"],
            "registration": registration,
            "eligible_correspondences": len(natural_target),
            "occupied_spatial_folds": len(fold_improvements),
            "metrics": {
                "identity_rmse": identity_rmse,
                "candidate_rmse": candidate_rmse,
                "wrong_pairing_rmse": wrong_rmse,
                "improvement_over_identity_fraction": 1.0 - candidate_rmse / identity_rmse,
                "improvement_over_wrong_pairing_fraction": 1.0 - candidate_rmse / wrong_rmse,
                "spatial_fold_win_fraction": float(np.mean(np.asarray(fold_improvements) > 0.0)),
                "worst_spatial_fold_improvement_fraction": min(fold_improvements),
                "out_of_cube_fraction": float(np.mean((prediction < 0.0) | (prediction > 1.0))),
                "maximum_repeat_error": float(np.max(np.abs(prediction - replay))),
            },
        })
    primary = next(row for row in rows if row["primary"])
    metrics = primary["metrics"]
    gates = contract["gates"]
    checks = {
        "support": primary["eligible_correspondences"] >= gates["minimum_primary_eligible_correspondences"],
        "spatial_coverage": primary["occupied_spatial_folds"] == int(contract["correspondence"]["spatial_fold_count"]),
        "improvement": metrics["improvement_over_identity_fraction"] >= gates["minimum_primary_improvement_over_identity_fraction"],
        "pairing": metrics["improvement_over_wrong_pairing_fraction"] >= gates["minimum_primary_improvement_over_wrong_pairing_fraction"],
        "folds": metrics["spatial_fold_win_fraction"] >= gates["minimum_primary_spatial_fold_win_fraction"],
        "tail": metrics["worst_spatial_fold_improvement_fraction"] >= gates["minimum_primary_worst_spatial_fold_improvement_fraction"],
        "accuracy": metrics["candidate_rmse"] <= gates["maximum_primary_rmse"],
        "cube": metrics["out_of_cube_fraction"] <= gates["maximum_out_of_cube_fraction"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(v)) for v in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "chart_dataset": chart_data,
        "chart_affine": affine,
        "rows": rows,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
