"""Bounded natural RAW/Sony geometry preflight for the NTIRE night pairs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.real_film.ntire_night_metadata_preflight import extract_member
from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    canonical_sha256,
    http_range_get,
    parse_central_directory,
    sha256_bytes,
)


class NTIRENightGeometryPreflightError(ValueError):
    """Raised when the frozen natural-pair geometry contract is invalid."""


RangeReader = Callable[[str, int, int, int], bytes]

OFFICIAL_ORIENTATION_STR_TO_ID = {
    "Horizontal (normal)": 1,
    "Mirror horizontal": 2,
    "Rotate 180": 3,
    "Mirror vertical": 4,
    "Mirror horizontal and rotate 270 CW": 5,
    "Rotate 90 CW": 6,
    "Mirror horizontal and rotate 90 CW": 7,
    "Rotate 270 CW": 8,
}


def _decode(payload: bytes, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
    if image is None:
        raise NTIRENightGeometryPreflightError("image decode failed")
    return image


def _shape(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def validate_metadata(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    contract = config["metadata_contract"]
    if not all(key in metadata for key in contract["required_exact_keys"]):
        raise NTIRENightGeometryPreflightError("required exact metadata key missing")
    for key in ("as_shot_neutral", "black_level", "cfa_pattern", "huawei_bounds"):
        if _shape(metadata[key]) != contract[f"{key}_shape"]:
            raise NTIRENightGeometryPreflightError(f"{key} shape differs")
    if metadata["orientation"] not in contract["orientation_allowed"]:
        raise NTIRENightGeometryPreflightError("orientation is unsupported")
    white = np.asarray(metadata["white_level"])
    if white.size != 1 or not np.isfinite(white.astype(np.float64)).all():
        raise NTIRENightGeometryPreflightError("white level is not finite scalar")
    if np.asarray(metadata["noise_profile"]).size < contract[
        "noise_profile_minimum_values"
    ]:
        raise NTIRENightGeometryPreflightError("noise profile is underspecified")
    numeric = np.concatenate(
        [
            np.asarray(metadata[key], dtype=np.float64).reshape(-1)
            for key in (
                "as_shot_neutral",
                "black_level",
                "cfa_pattern",
                "huawei_bounds",
                "noise_profile",
                "white_level",
            )
        ]
    )
    if not np.isfinite(numeric).all():
        raise NTIRENightGeometryPreflightError("metadata contains nonfinite values")


def structural_view(
    raw: np.ndarray, metadata: dict[str, Any], geometry: dict[str, Any]
) -> np.ndarray:
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise NTIRENightGeometryPreflightError("RAW PNG is not uint16 single plane")
    black = np.asarray(metadata["black_level"], dtype=np.float32)
    white = float(np.asarray(metadata["white_level"]).reshape(-1)[0])
    black_mask = np.empty(raw.shape, dtype=np.float32)
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        black_mask[row::2, column::2] = black[index]
    denominator = white - black_mask
    if np.any(denominator <= 0):
        raise NTIRENightGeometryPreflightError("white level does not exceed black")
    normalized = np.maximum(raw.astype(np.float32) - black_mask, 0.0) / denominator
    pattern = np.asarray(metadata["cfa_pattern"], dtype=np.int64).reshape(2, 2)
    if sorted(pattern.reshape(-1).tolist()) != [0, 1, 1, 2]:
        raise NTIRENightGeometryPreflightError("CFA is not one R/two G/one B")
    height, width = raw.shape[0] // 2, raw.shape[1] // 2
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for row in range(2):
        for column in range(2):
            channel = int(pattern[row, column])
            contribution = normalized[row::2, column::2]
            rgb[..., channel] += contribution / (2.0 if channel == 1 else 1.0)
    neutral = np.asarray(metadata["as_shot_neutral"], dtype=np.float32)
    if np.any(neutral <= 0):
        raise NTIRENightGeometryPreflightError("as-shot neutral is nonpositive")
    rgb = np.clip(rgb / neutral.reshape(1, 1, 3), 0.0, 1.0)

    orientation = metadata["orientation"]
    if isinstance(orientation, str):
        orientation = OFFICIAL_ORIENTATION_STR_TO_ID.get(orientation)
    if orientation == 2:
        rgb = cv2.flip(rgb, 0)
    elif orientation == 3:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    elif orientation == 4:
        rgb = cv2.flip(rgb, 1)
    elif orientation == 5:
        rgb = cv2.flip(rgb, 0)
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        rgb = cv2.flip(rgb, 0)
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation != 1:
        raise NTIRENightGeometryPreflightError("orientation mapping differs")

    factor = int(geometry["projective_pre_resize_factor"])
    resized = cv2.resize(
        rgb,
        (rgb.shape[1] // factor, rgb.shape[0] // factor),
        interpolation=cv2.INTER_LINEAR,
    )
    projected = cv2.warpPerspective(
        resized,
        np.asarray(geometry["projective_matrix"], dtype=np.float64),
        (geometry["projective_output_width"], geometry["projective_output_height"]),
        flags=cv2.INTER_LINEAR,
    )
    if geometry["horizontal_flip"]:
        projected = cv2.flip(projected, 1)
    resized = cv2.resize(
        projected,
        (geometry["pre_bounds_resize_width"], geometry["pre_bounds_resize_height"]),
        interpolation=cv2.INTER_LINEAR,
    )
    h_start, h_end, w_start, w_end = [int(value) for value in metadata["huawei_bounds"]]
    if not (
        0 <= h_start < h_end <= resized.shape[0]
        and 0 <= w_start < w_end <= resized.shape[1]
    ):
        raise NTIRENightGeometryPreflightError("Huawei bounds exceed geometry")
    bounded = resized[h_start:h_end, w_start:w_end]
    upper = int(geometry["upper_crop_start"])
    final_height = int(geometry["final_height"])
    final_width = int(geometry["final_width"])
    middle = bounded.shape[1] // 2
    output = bounded[
        upper : upper + final_height,
        middle - final_width // 2 : middle + final_width // 2,
    ]
    if output.shape != (final_height, final_width, 3):
        raise NTIRENightGeometryPreflightError("official geometry output shape differs")
    return output


def _rank_gray(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 3:
        gray = image.astype(np.float32).mean(axis=2)
    else:
        gray = image.astype(np.float32)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    flat = small.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.uint8)
    ranks[order] = np.linspace(0, 255, flat.size, dtype=np.uint8)
    return ranks.reshape(size, size)


def match_facts(
    reference: np.ndarray, target: np.ndarray, registration: dict[str, Any]
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260821)
    detector = cv2.SIFT_create(nfeatures=int(registration["sift_nfeatures"]))
    key_a, desc_a = detector.detectAndCompute(reference, None)
    key_b, desc_b = detector.detectAndCompute(target, None)
    if desc_a is None or desc_b is None or len(desc_b) < 2:
        return {
            "keypoints_reference": len(key_a),
            "keypoints_target": len(key_b),
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_pixels": None,
            "maximum_corner_displacement_pixels": None,
        }
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [
        first
        for first, second in pairs
        if first.distance < float(registration["ratio_test"]) * second.distance
    ]
    if len(good) < 4:
        return {
            "keypoints_reference": len(key_a),
            "keypoints_target": len(key_b),
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_pixels": None,
            "maximum_corner_displacement_pixels": None,
        }
    source = np.float32([key_a[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([key_b[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        float(registration["ransac_reprojection_pixels"]),
    )
    if homography is None or mask is None:
        return {
            "keypoints_reference": len(key_a),
            "keypoints_target": len(key_b),
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_pixels": None,
            "maximum_corner_displacement_pixels": None,
        }
    inlier = mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(source, homography)
    errors = np.linalg.norm(projected - destination, axis=2).reshape(-1)
    size = float(registration["working_size"] - 1)
    corners = np.float32([[[0, 0]], [[size, 0]], [[size, size]], [[0, size]]])
    warped = cv2.perspectiveTransform(corners, homography)
    corner_displacement = np.linalg.norm(warped - corners, axis=2).reshape(-1)
    return {
        "keypoints_reference": len(key_a),
        "keypoints_target": len(key_b),
        "good_matches": len(good),
        "inliers": int(inlier.sum()),
        "inlier_ratio": float(inlier.mean()),
        "median_reprojection_pixels": float(np.median(errors[inlier])),
        "maximum_corner_displacement_pixels": float(corner_displacement.max()),
    }


def _members(config: dict[str, Any], range_reader: RangeReader) -> tuple[dict[str, ZipMember], int, list[str]]:
    by_name: dict[str, ZipMember] = {}
    bytes_read = 0
    hashes: list[str] = []
    for key in ("raw", "target"):
        archive = config["archives"][key]
        central = range_reader(
            archive["url"],
            archive["central_offset"],
            archive["central_offset"] + archive["central_size"] - 1,
            archive["bytes"],
        )
        digest = sha256_bytes(central)
        if digest != archive["central_sha256"]:
            raise NTIRENightGeometryPreflightError("central directory differs")
        hashes.append(digest)
        for member in parse_central_directory(central):
            by_name[f"{key}:{member.name}"] = member
        bytes_read += len(central)
    return by_name, bytes_read, hashes


def run_preflight(
    config_path: Path,
    *,
    reverse_row_order: bool = False,
    range_reader: RangeReader = http_range_get,
) -> dict[str, Any]:
    contract_bytes = config_path.read_bytes()
    contract = json.loads(contract_bytes)
    contract_schema = contract.get("schema")
    base_contract_sha256 = None
    if contract_schema == (
        "neuro-film.sf3-a0x-ntire-night-official-orientation-preflight-contract.v1"
    ):
        base_path = config_path.parents[1] / contract["base_contract"]["path"]
        base_bytes = base_path.read_bytes()
        base_contract_sha256 = sha256_bytes(base_bytes)
        if base_contract_sha256 != contract["base_contract"]["sha256"]:
            raise NTIRENightGeometryPreflightError("base contract identity differs")
        config = json.loads(base_bytes)
        config["experiment_id"] = contract["experiment_id"]
        config["metadata_contract"]["orientation_allowed"] = [
            *contract["allowed_numeric_orientations"],
            *contract["official_orientation_mapping"].keys(),
        ]
        config["decision_if_pass"] = contract["decision_if_pass"]
        config["claim_ceiling"] = contract["claim_ceiling"]
        config["bounded_final_candidate_counter_before"] = contract[
            "bounded_final_candidate_counter_before"
        ]
        config["bounded_final_candidate_counter_after"] = contract[
            "bounded_final_candidate_counter_after"
        ]
        report_schema = (
            "neuro-film.sf3-a0x-ntire-night-official-orientation-preflight-report.v1"
        )
    elif contract_schema == (
        "neuro-film.sf3-a0w-ntire-night-geometry-preflight-contract.v1"
    ):
        config = contract
        report_schema = (
            "neuro-film.sf3-a0w-ntire-night-geometry-preflight-report.v1"
        )
    else:
        raise NTIRENightGeometryPreflightError("contract schema differs")
    by_name, archive_bytes, central_hashes = _members(config, range_reader)
    selected = config["rows"]["selected_ids"]
    sequence = list(reversed(selected)) if reverse_row_order else selected
    views: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    row_inputs: dict[int, dict[str, Any]] = {}
    for numeric_id in sequence:
        raw_archive = config["archives"]["raw"]
        target_archive = config["archives"]["target"]
        names = {
            "metadata": f"raw:raw/{numeric_id}.json",
            "raw": f"raw:raw/{numeric_id}.png",
            "target": f"target:sony/{numeric_id}.JPG",
        }
        if any(name not in by_name for name in names.values()):
            raise NTIRENightGeometryPreflightError("selected member is missing")
        metadata_payload, read = extract_member(
            raw_archive["url"], by_name[names["metadata"]], raw_archive["bytes"], range_reader
        )
        archive_bytes += read
        metadata = json.loads(metadata_payload)
        validate_metadata(metadata, config)
        raw_member = by_name[names["raw"]]
        target_member = by_name[names["target"]]
        if raw_member.uncompressed_size > config["read_budget"][
            "maximum_raw_png_uncompressed_bytes_per_row"
        ]:
            raise NTIRENightGeometryPreflightError("RAW member exceeds byte cap")
        if target_member.uncompressed_size > config["read_budget"][
            "maximum_target_jpeg_uncompressed_bytes_per_row"
        ]:
            raise NTIRENightGeometryPreflightError("target member exceeds byte cap")
        raw_payload, read = extract_member(
            raw_archive["url"], raw_member, raw_archive["bytes"], range_reader
        )
        archive_bytes += read
        target_payload, read = extract_member(
            target_archive["url"], target_member, target_archive["bytes"], range_reader
        )
        archive_bytes += read
        raw = _decode(raw_payload, cv2.IMREAD_UNCHANGED)
        target = _decode(target_payload, cv2.IMREAD_COLOR)
        view = structural_view(raw, metadata, config["official_baseline"]["geometry"])
        if target.shape != (2000, 2000, 3):
            raise NTIRENightGeometryPreflightError("Sony target shape differs")
        size = int(config["registration"]["working_size"])
        views[numeric_id] = _rank_gray(view, size)
        targets[numeric_id] = _rank_gray(target, size)
        row_inputs[numeric_id] = {
            "numeric_id": numeric_id,
            "metadata_sha256": sha256_bytes(metadata_payload),
            "raw_png_sha256": sha256_bytes(raw_payload),
            "target_jpeg_sha256": sha256_bytes(target_payload),
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "target_shape": list(target.shape),
            "huawei_bounds": [int(value) for value in metadata["huawei_bounds"]],
            "orientation": metadata["orientation"],
        }
    if archive_bytes > config["read_budget"]["maximum_total_archive_bytes_per_run"]:
        raise NTIRENightGeometryPreflightError("total archive byte cap exceeded")

    registration = config["registration"]
    rows = []
    for index, numeric_id in enumerate(selected):
        wrong_id = selected[(index + 1) % len(selected)]
        correct = match_facts(views[numeric_id], targets[numeric_id], registration)
        wrong = match_facts(views[numeric_id], targets[wrong_id], registration)
        passed = (
            correct["good_matches"] >= registration["minimum_good_matches"]
            and correct["inliers"] >= registration["minimum_inliers"]
            and correct["inlier_ratio"] >= registration["minimum_inlier_ratio"]
            and correct["inlier_ratio"] - wrong["inlier_ratio"]
            >= registration["minimum_correct_minus_wrong_inlier_ratio"]
            and correct["median_reprojection_pixels"] is not None
            and correct["median_reprojection_pixels"]
            <= registration["maximum_median_reprojection_pixels"]
            and correct["maximum_corner_displacement_pixels"] is not None
            and correct["maximum_corner_displacement_pixels"]
            <= registration["maximum_corner_displacement_pixels"]
        )
        rows.append(
            {
                **row_inputs[numeric_id],
                "wrong_target_id": wrong_id,
                "correct": correct,
                "wrong": wrong,
                "correct_minus_wrong_inlier_ratio": (
                    correct["inlier_ratio"] - wrong["inlier_ratio"]
                ),
                "passed": passed,
            }
        )
    passing = sum(row["passed"] for row in rows)
    gates = {
        "official_source_identity_frozen": True,
        "exact_metadata_shapes": True,
        "selected_rows_complete": len(rows) == len(selected),
        "required_registration_rows": passing >= registration["required_passing_rows"],
        "pair_specific_wrong_target_separation": all(
            row["correct_minus_wrong_inlier_ratio"]
            >= registration["minimum_correct_minus_wrong_inlier_ratio"]
            for row in rows
        ),
        "bounded_archive_reads": archive_bytes
        <= config["read_budget"]["maximum_total_archive_bytes_per_run"],
        "zero_persistent_member_bytes": True,
        "zero_operator_fit_render_score": True,
    }
    passed = all(gates.values())
    report = {
        "schema": report_schema,
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(contract_bytes),
        "base_contract_sha256": base_contract_sha256,
        "official_repository": config["official_baseline"]["repository"],
        "official_commit": config["official_baseline"]["commit"],
        "central_sha256": central_hashes,
        "selected_ids": selected,
        "rows": rows,
        "row_count": len(rows),
        "passing_row_count": passing,
        "archive_bytes_read": archive_bytes,
        "persistent_member_bytes": 0,
        "operator_fits": 0,
        "renders": 0,
        "quality_scores": 0,
        "bounded_final_candidate_counter_before": config[
            "bounded_final_candidate_counter_before"
        ],
        "bounded_final_candidate_counter_after": config[
            "bounded_final_candidate_counter_after"
        ],
        "gates": gates,
        "automatic_pass": passed,
        "decision": (
            config["decision_if_pass"]
            if passed
            else "FAIL_CLOSED_NATURAL_GEOMETRY_PREFLIGHT"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
