"""Same-slide scanner/software nuisance quantification.

This module operates only on the frozen ColorReference device-RGB test lane.
It deliberately does not expose a stock operator or a production renderer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.optimize import lsq_linear
from skimage.color import deltaE_ciede2000, rgb2lab
import tifffile


class ScannerNuisanceError(RuntimeError):
    """Raised when frozen lineage, alignment, or evaluation is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_rgb(payload: bytes) -> np.ndarray:
    array = np.asarray(tifffile.imread(BytesIO(payload)))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ScannerNuisanceError(f"expected RGB TIFF, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ScannerNuisanceError(f"expected integer TIFF samples, got {array.dtype}")
    maximum = float(np.iinfo(array.dtype).max)
    return array.astype(np.float64) / maximum


def _alignment_gray(rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(rgb, 0.0, 1.0)
    encoded = np.rint(clipped * 255.0).astype(np.uint8)
    return cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY)


def align_source_to_scan(
    source_rgb: np.ndarray,
    scan_rgb: np.ndarray,
    alignment: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a deterministic source-to-scan homography and diagnostics."""

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=int(alignment["maximum_features"]))
    source_keypoints, source_descriptors = sift.detectAndCompute(
        _alignment_gray(source_rgb), None
    )
    scan_keypoints, scan_descriptors = sift.detectAndCompute(
        _alignment_gray(scan_rgb), None
    )
    if source_descriptors is None or scan_descriptors is None:
        raise ScannerNuisanceError("SIFT returned no descriptors")
    matches = cv2.BFMatcher().knnMatch(
        source_descriptors, scan_descriptors, k=2
    )
    ratio = float(alignment["ratio_test"])
    good = [left for left, right in matches if left.distance < ratio * right.distance]
    if len(good) < int(alignment["minimum_good_matches"]):
        raise ScannerNuisanceError(f"only {len(good)} good SIFT matches")
    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good]
    )
    scan_points = np.float32(
        [scan_keypoints[match.trainIdx].pt for match in good]
    )
    homography, mask = cv2.findHomography(
        source_points,
        scan_points,
        cv2.RANSAC,
        float(alignment["ransac_reprojection_threshold_px"]),
    )
    if homography is None or mask is None:
        raise ScannerNuisanceError("homography estimation failed")
    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(np.sum(inliers))
    if inlier_count < int(alignment["minimum_ransac_inliers"]):
        raise ScannerNuisanceError(f"only {inlier_count} RANSAC inliers")
    projected = cv2.perspectiveTransform(
        source_points[inliers].reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - scan_points[inliers], axis=1)
    median_error = float(np.median(errors))
    if median_error > float(
        alignment["maximum_median_inlier_reprojection_error_px"]
    ):
        raise ScannerNuisanceError(
            f"median alignment error {median_error:.6f}px exceeds gate"
        )
    return homography.astype(np.float64), {
        "source_keypoints": len(source_keypoints),
        "scan_keypoints": len(scan_keypoints),
        "good_matches": len(good),
        "ransac_inliers": inlier_count,
        "median_inlier_reprojection_error_px": median_error,
        "maximum_inlier_reprojection_error_px": float(np.max(errors)),
    }


def patch_medians(
    scan_rgb: np.ndarray,
    homography: np.ndarray,
    grid: Mapping[str, Any],
) -> np.ndarray:
    """Extract native-resolution medians from fixed source-grid inner cells."""

    columns = int(grid["columns"])
    rows = int(grid["rows"])
    left = float(grid["left_edge_px"])
    right = float(grid["right_edge_px"])
    top = float(grid["top_edge_px"])
    bottom = float(grid["bottom_edge_px"])
    inner = float(grid["inner_patch_fraction"])
    cell_width = (right - left) / columns
    cell_height = (bottom - top) / rows
    inset_x = 0.5 * (1.0 - inner) * cell_width
    inset_y = 0.5 * (1.0 - inner) * cell_height
    height, width = scan_rgb.shape[:2]
    medians = []
    for row in range(rows):
        for column in range(columns):
            x0 = left + column * cell_width + inset_x
            x1 = left + (column + 1) * cell_width - inset_x
            y0 = top + row * cell_height + inset_y
            y1 = top + (row + 1) * cell_height - inset_y
            source_quad = np.float32(
                [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]
            )
            scan_quad = cv2.perspectiveTransform(
                source_quad, homography
            ).reshape(4, 2)
            polygon = np.rint(scan_quad).astype(np.int32)
            if (
                np.any(polygon[:, 0] < 0)
                or np.any(polygon[:, 0] >= width)
                or np.any(polygon[:, 1] < 0)
                or np.any(polygon[:, 1] >= height)
            ):
                raise ScannerNuisanceError(
                    f"mapped patch {row},{column} leaves scan bounds"
                )
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon, 1)
            values = scan_rgb[mask.astype(bool)]
            if values.shape[0] < 25:
                raise ScannerNuisanceError(
                    f"mapped patch {row},{column} has only {values.shape[0]} pixels"
                )
            medians.append(np.median(values, axis=0))
    return np.asarray(medians, dtype=np.float64)


def _fit_channel_affine(
    source: np.ndarray,
    target: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    scale_bounds = tuple(float(value) for value in config["per_channel_scale_bounds"])
    bias_bounds = tuple(float(value) for value in config["per_channel_bias_bounds"])
    matrix = np.zeros((3, 3), dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        design = np.column_stack(
            [source[:, channel], np.ones(source.shape[0], dtype=np.float64)]
        )
        fit = lsq_linear(
            design,
            target[:, channel],
            bounds=(
                [scale_bounds[0], bias_bounds[0]],
                [scale_bounds[1], bias_bounds[1]],
            ),
            method="trf",
            lsmr_tol="auto",
        )
        if not fit.success:
            raise ScannerNuisanceError(f"per-channel fit failed: {fit.message}")
        matrix[channel, channel] = fit.x[0]
        bias[channel] = fit.x[1]
    return matrix, bias


def _fit_full_affine(
    source: np.ndarray,
    target: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    coefficient_bounds = tuple(
        float(value) for value in config["full_affine_coefficient_bounds"]
    )
    bias_bounds = tuple(float(value) for value in config["full_affine_bias_bounds"])
    design = np.column_stack(
        [source, np.ones(source.shape[0], dtype=np.float64)]
    )
    matrix = np.zeros((3, 3), dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        fit = lsq_linear(
            design,
            target[:, channel],
            bounds=(
                [coefficient_bounds[0]] * 3 + [bias_bounds[0]],
                [coefficient_bounds[1]] * 3 + [bias_bounds[1]],
            ),
            method="trf",
            lsmr_tol="auto",
        )
        if not fit.success:
            raise ScannerNuisanceError(f"full-affine fit failed: {fit.message}")
        matrix[channel] = fit.x[:3]
        bias[channel] = fit.x[3]
    return matrix, bias


def apply_affine(
    source: np.ndarray, matrix: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    return np.clip(source @ matrix.T + bias, 0.0, 1.0)


def distance_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rgb_distance = np.linalg.norm(predicted - target, axis=1)
    predicted_lab = rgb2lab(
        np.clip(predicted, 0.0, 1.0).reshape(-1, 1, 3)
    ).reshape(-1, 3)
    target_lab = rgb2lab(
        np.clip(target, 0.0, 1.0).reshape(-1, 1, 3)
    ).reshape(-1, 3)
    assumed_delta = deltaE_ciede2000(predicted_lab, target_lab)
    return {
        "median_rgb_euclidean": float(np.median(rgb_distance)),
        "p90_rgb_euclidean": float(np.percentile(rgb_distance, 90)),
        "maximum_rgb_euclidean": float(np.max(rgb_distance)),
        "median_delta_e00_srgb_assumption": float(np.median(assumed_delta)),
        "p90_delta_e00_srgb_assumption": float(np.percentile(assumed_delta, 90)),
    }


def evaluate_leave_one_slide_out(
    patch_bank: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    roles = [str(value) for value in config["pipeline_roles"]]
    slides = [str(value) for value in config["slide_ids"]]
    evaluation = config["evaluation"]
    records: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {
        model: [] for model in evaluation["models"]
    }
    unordered_raw: dict[str, list[np.ndarray]] = {}
    for left, right in itertools.combinations(roles, 2):
        pair_key = f"{left}__vs__{right}"
        unordered_raw[pair_key] = []
        for held_out in slides:
            train_slides = [slide for slide in slides if slide != held_out]
            for source_role, target_role in ((left, right), (right, left)):
                train_source = np.concatenate(
                    [patch_bank[source_role][slide] for slide in train_slides]
                )
                train_target = np.concatenate(
                    [patch_bank[target_role][slide] for slide in train_slides]
                )
                test_source = patch_bank[source_role][held_out]
                test_target = patch_bank[target_role][held_out]
                raw_distance = np.linalg.norm(test_source - test_target, axis=1)
                pooled["identity"].append(raw_distance)
                if source_role == left:
                    unordered_raw[pair_key].append(raw_distance)
                record: dict[str, Any] = {
                    "source_pipeline": source_role,
                    "target_pipeline": target_role,
                    "held_out_slide": held_out,
                    "models": {
                        "identity": distance_metrics(test_source, test_target)
                    },
                }
                channel_matrix, channel_bias = _fit_channel_affine(
                    train_source, train_target, evaluation
                )
                channel_output = apply_affine(
                    test_source, channel_matrix, channel_bias
                )
                pooled["per_channel_affine"].append(
                    np.linalg.norm(channel_output - test_target, axis=1)
                )
                record["models"]["per_channel_affine"] = {
                    **distance_metrics(channel_output, test_target),
                    "matrix": channel_matrix.tolist(),
                    "bias": channel_bias.tolist(),
                }
                full_matrix, full_bias = _fit_full_affine(
                    train_source, train_target, evaluation
                )
                full_output = apply_affine(test_source, full_matrix, full_bias)
                pooled["bounded_full_affine_3x3_plus_bias"].append(
                    np.linalg.norm(full_output - test_target, axis=1)
                )
                record["models"]["bounded_full_affine_3x3_plus_bias"] = {
                    **distance_metrics(full_output, test_target),
                    "matrix": full_matrix.tolist(),
                    "bias": full_bias.tolist(),
                }
                records.append(record)
    aggregate = {}
    for model, chunks in pooled.items():
        values = np.concatenate(chunks)
        aggregate[model] = {
            "median_rgb_euclidean": float(np.median(values)),
            "p90_rgb_euclidean": float(np.percentile(values, 90)),
            "maximum_rgb_euclidean": float(np.max(values)),
            "patch_predictions": int(values.size),
        }
    pair_raw = {}
    for pair, chunks in unordered_raw.items():
        values = np.concatenate(chunks)
        pair_raw[pair] = {
            "median_rgb_euclidean": float(np.median(values)),
            "p90_rgb_euclidean": float(np.percentile(values, 90)),
            "maximum_rgb_euclidean": float(np.max(values)),
            "paired_patches": int(values.size),
        }
    raw_median = float(aggregate["identity"]["median_rgb_euclidean"])
    candidate_models = [
        model for model in evaluation["models"] if model != "identity"
    ]
    best_model = min(
        candidate_models,
        key=lambda model: aggregate[model]["median_rgb_euclidean"],
    )
    best_median = float(aggregate[best_model]["median_rgb_euclidean"])
    thresholds = config["pre_registered_interpretation"]
    reduction = 1.0 - best_median / raw_median if raw_median > 0 else 0.0
    interpretation = {
        "best_global_canonicalizer": best_model,
        "aggregate_median_reduction_fraction": reduction,
        "material_raw_nuisance": (
            raw_median
            >= float(
                thresholds["material_raw_nuisance_aggregate_median_rgb_minimum"]
            )
            and all(
                row["median_rgb_euclidean"]
                >= float(
                    thresholds[
                        "material_raw_nuisance_each_pair_median_rgb_minimum"
                    ]
                )
                for row in pair_raw.values()
            )
        ),
        "canonicalizer_material_reduction": reduction
        >= float(thresholds["canonicalizer_material_reduction_fraction_minimum"]),
        "small_residual": best_median
        <= float(thresholds["small_residual_aggregate_median_rgb_maximum"]),
    }
    return {
        "directed_fold_records": records,
        "unordered_pair_raw": pair_raw,
        "aggregate": aggregate,
        "interpretation": interpretation,
    }


def build_report(root: Path, config_path: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    parent_config = root / str(config["parent_config"])
    parent_audit = root / str(config["parent_audit"])
    if sha256_file(parent_config) != str(config["parent_config_sha256"]):
        raise ScannerNuisanceError("parent config hash mismatch")
    if sha256_file(parent_audit) != str(config["parent_audit_sha256"]):
        raise ScannerNuisanceError("parent audit hash mismatch")
    parent_report = json.loads(parent_audit.read_text(encoding="utf-8"))
    if not parent_report.get("all_checks_passed"):
        raise ScannerNuisanceError("parent source audit did not pass")
    data_root = root / str(config["data_root"])
    source_by_slide = {
        slide: data_root / "source" / f"slidescale{slide}.tif"
        for slide in config["slide_ids"]
    }
    role_assets = {
        str(record["role"]): record
        for record in parent_report["assets"]
        if str(record["role"]).startswith("nikon_")
    }
    patch_bank: dict[str, dict[str, np.ndarray]] = {}
    alignments = []
    for role in config["pipeline_roles"]:
        record = role_assets[str(role)]
        patch_bank[str(role)] = {}
        with ZipFile(data_root / str(record["path"])) as archive:
            members = {
                str(member["slide_id"]): str(member["name"])
                for member in record["members"]
                if member.get("slide_id") is not None
            }
            for slide in config["slide_ids"]:
                source_rgb = _native_rgb(source_by_slide[str(slide)].read_bytes())
                scan_rgb = _native_rgb(archive.read(members[str(slide)]))
                homography, diagnostics = align_source_to_scan(
                    source_rgb, scan_rgb, config["alignment"]
                )
                patches = patch_medians(
                    scan_rgb, homography, config["source_grid"]
                )
                expected_patches = int(config["evaluation"]["patches_per_slide"])
                if patches.shape != (expected_patches, 3):
                    raise ScannerNuisanceError(
                        f"{role}/{slide} yielded {patches.shape}"
                    )
                patch_bank[str(role)][str(slide)] = patches
                alignments.append(
                    {
                        "pipeline": role,
                        "slide_id": slide,
                        **diagnostics,
                    }
                )
    evaluation = evaluate_leave_one_slide_out(patch_bank, config)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "parent_config_sha256_verified": True,
        "parent_audit_sha256_verified": True,
        "colour_state": config["decode"]["colour_state"],
        "pipeline_count": len(config["pipeline_roles"]),
        "slide_count": len(config["slide_ids"]),
        "patches_per_slide": int(config["evaluation"]["patches_per_slide"]),
        "alignment_records": alignments,
        "all_alignment_gates_passed": True,
        **evaluation,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ScannerNuisanceError",
    "align_source_to_scan",
    "apply_affine",
    "build_report",
    "distance_metrics",
    "evaluate_leave_one_slide_out",
    "patch_medians",
    "sha256_file",
]
