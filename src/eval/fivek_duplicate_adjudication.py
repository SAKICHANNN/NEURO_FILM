"""Source-only multi-evidence adjudication of perceptual duplicate candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image


class FiveKDuplicateAdjudicationError(ValueError):
    """Raised when frozen source-only duplicate evidence cannot be reproduced."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FiveKDuplicateAdjudicationError(f"missing source image: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 32:
        raise FiveKDuplicateAdjudicationError("invalid source image")
    return rgb


def _dhash64(rgb: np.ndarray) -> int:
    gray = np.asarray(
        Image.fromarray(rgb, mode="RGB")
        .convert("L")
        .resize((9, 8), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def _phash64(rgb: np.ndarray) -> int:
    gray = np.asarray(
        Image.fromarray(rgb, mode="RGB")
        .convert("L")
        .resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    low = cv2.dct(gray)[:8, :8]
    threshold = float(np.median(low.ravel()[1:]))
    value = 0
    for bit in (low > threshold).ravel():
        value = (value << 1) | int(bit)
    return value


def _normalized_luma_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def normalized(rgb: np.ndarray) -> np.ndarray:
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(luma, (128, 128), interpolation=cv2.INTER_AREA)
        vector = resized.ravel().astype(np.float64)
        vector -= vector.mean()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1.0e-12 else np.zeros_like(vector)

    return float(np.dot(normalized(first), normalized(second)))


def _orb_geometry(
    first: np.ndarray,
    second: np.ndarray,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    def gray_for_orb(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        maximum_side = int(protocol["maximum_orb_side"])
        scale = min(1.0, maximum_side / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (
                    max(1, int(round(gray.shape[1] * scale))),
                    max(1, int(round(gray.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    previous_threads = cv2.getNumThreads()
    previous_optimized = cv2.useOptimized()
    previous_opencl = cv2.ocl.useOpenCL()
    try:
        cv2.setNumThreads(1)
        cv2.setUseOptimized(False)
        cv2.ocl.setUseOpenCL(False)
        cv2.setRNGSeed(int(protocol["opencv_rng_seed"]))
        orb = cv2.ORB_create(
            nfeatures=int(protocol["maximum_features"]),
            scaleFactor=float(protocol["scale_factor"]),
            nlevels=int(protocol["levels"]),
        )
        keypoints_a, descriptors_a = orb.detectAndCompute(gray_for_orb(first), None)
        keypoints_b, descriptors_b = orb.detectAndCompute(gray_for_orb(second), None)
        if descriptors_a is None or descriptors_b is None:
            return {
                "keypoints_first": len(keypoints_a),
                "keypoints_second": len(keypoints_b),
                "mutual_ratio_matches": 0,
                "ransac_inliers": 0,
                "ransac_inlier_fraction": 0.0,
                "confirmed": False,
            }
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        def ratio_pairs(left: np.ndarray, right: np.ndarray) -> set[tuple[int, int]]:
            pairs: set[tuple[int, int]] = set()
            for matches in matcher.knnMatch(left, right, k=2):
                if len(matches) == 2 and matches[0].distance < (
                    float(protocol["ratio_test"]) * matches[1].distance
                ):
                    pairs.add((matches[0].queryIdx, matches[0].trainIdx))
            return pairs

        forward = ratio_pairs(descriptors_a, descriptors_b)
        backward = ratio_pairs(descriptors_b, descriptors_a)
        mutual = sorted(pair for pair in forward if (pair[1], pair[0]) in backward)
        inliers = 0
        if len(mutual) >= 4:
            points_a = np.float32([keypoints_a[a].pt for a, _ in mutual])
            points_b = np.float32([keypoints_b[b].pt for _, b in mutual])
            _, mask = cv2.findHomography(
                points_a,
                points_b,
                cv2.RANSAC,
                float(protocol["ransac_reprojection_pixels"]),
                maxIters=int(protocol["ransac_maximum_iterations"]),
                confidence=float(protocol["ransac_confidence"]),
            )
            inliers = int(mask.sum()) if mask is not None else 0
        fraction = inliers / max(len(mutual), 1)
        confirmed = (
            len(mutual) >= int(protocol["minimum_mutual_ratio_matches"])
            and inliers >= int(protocol["minimum_ransac_inliers"])
            and fraction >= float(protocol["minimum_ransac_inlier_fraction"])
        )
        return {
            "keypoints_first": len(keypoints_a),
            "keypoints_second": len(keypoints_b),
            "mutual_ratio_matches": len(mutual),
            "ransac_inliers": inliers,
            "ransac_inlier_fraction": fraction,
            "confirmed": confirmed,
        }
    finally:
        cv2.setUseOptimized(previous_optimized)
        cv2.ocl.setUseOpenCL(previous_opencl)
        cv2.setNumThreads(previous_threads)


def compare_source_images(
    first_path: Path,
    second_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    first = _load_rgb(first_path)
    second = _load_rgb(second_path)
    dhash_distance = (_dhash64(first) ^ _dhash64(second)).bit_count()
    phash_distance = (_phash64(first) ^ _phash64(second)).bit_count()
    luma_correlation = _normalized_luma_correlation(first, second)
    geometry = _orb_geometry(first, second, protocol["orb"])
    phash_confirmed = (
        phash_distance <= int(protocol["maximum_phash_hamming"])
        and luma_correlation
        >= float(protocol["minimum_normalized_luma_correlation"])
    )
    confirmed = phash_confirmed or bool(geometry["confirmed"])
    return {
        "first_path": str(first_path.as_posix()),
        "first_sha256": _sha256(first_path),
        "second_path": str(second_path.as_posix()),
        "second_sha256": _sha256(second_path),
        "dhash_hamming": dhash_distance,
        "phash_hamming": phash_distance,
        "normalized_luma_correlation": luma_correlation,
        "phash_confirmed": phash_confirmed,
        "orb": geometry,
        "confirmed_duplicate": confirmed,
    }


def _load_hashed_json(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if not path.is_file() or _sha256(path) != str(spec["sha256"]):
        raise FiveKDuplicateAdjudicationError("parent JSON drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FiveKDuplicateAdjudicationError("parent JSON must be an object")
    return payload


def _prior_sources(root: Path, specs: list[Mapping[str, Any]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        path = root / str(spec["path"])
        if not path.is_file() or _sha256(path) != str(spec["sha256"]):
            raise FiveKDuplicateAdjudicationError("prior manifest drift")
        if spec["format"] == "csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        elif spec["format"] == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
        else:
            raise FiveKDuplicateAdjudicationError("unsupported prior manifest")
        if len(rows) != int(spec["expected_rows"]):
            raise FiveKDuplicateAdjudicationError("prior row-count drift")
        for row in rows:
            identifier = str(row[spec["id_field"]])
            candidate = Path(str(row[spec["source_path_field"]]))
            result[identifier] = candidate if candidate.is_absolute() else root / candidate
    return result


def run_adjudication(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or config.get("target_pixels_allowed") is not False
        or config.get("operator_fitting_allowed") is not False
        or config.get("adaptive_successor_after_bq0s2_failure") is not True
    ):
        raise FiveKDuplicateAdjudicationError("contract boundary drift")
    parent_report = _load_hashed_json(root, config["parent_report"])
    parent_manifest = _load_hashed_json(root, config["parent_manifest"])
    if parent_report.get("automatic_pass") is not False:
        raise FiveKDuplicateAdjudicationError("expected frozen parent failure")
    rows = {str(row["pair_id"]): row for row in parent_manifest.get("rows", [])}
    prior = _prior_sources(root, config["prior_manifests"])
    candidates: list[dict[str, Any]] = []
    for item in parent_report.get("perceptual_pairs", []):
        fresh_id = str(item["fresh_pair_id"])
        old_id = str(item["existing_pair_id"])
        if fresh_id not in rows or old_id not in prior:
            raise FiveKDuplicateAdjudicationError("unresolved prior candidate")
        evidence = compare_source_images(
            prior[old_id], Path(str(rows[fresh_id]["source_path"])), config["protocol"]
        )
        candidates.append(
            {"scope": "prior_pool", "first_id": old_id, "second_id": fresh_id, **evidence}
        )
    for item in parent_report.get("internal_perceptual_pairs", []):
        first_id = str(item["development_pair_id"])
        second_id = str(item["confirmation_pair_id"])
        if first_id not in rows or second_id not in rows:
            raise FiveKDuplicateAdjudicationError("unresolved internal candidate")
        evidence = compare_source_images(
            Path(str(rows[first_id]["source_path"])),
            Path(str(rows[second_id]["source_path"])),
            config["protocol"],
        )
        candidates.append(
            {"scope": "internal_split", "first_id": first_id, "second_id": second_id, **evidence}
        )
    expected = config["expected_candidates"]
    counts = {
        "prior_pool": sum(item["scope"] == "prior_pool" for item in candidates),
        "internal_split": sum(item["scope"] == "internal_split" for item in candidates),
    }
    gates = {
        "candidate_inventory": counts == expected,
        "dhash_candidate_reproduced": all(
            item["dhash_hamming"] <= int(config["protocol"]["maximum_dhash_hamming"])
            for item in candidates
        ),
        "no_confirmed_duplicates": not any(
            item["confirmed_duplicate"] for item in candidates
        ),
        "target_pixels_unused": True,
    }
    stable = {
        "parent_report_sha256": config["parent_report"]["sha256"],
        "parent_manifest_sha256": config["parent_manifest"]["sha256"],
        "candidate_counts": counts,
        "candidates": candidates,
        "gates": gates,
        "automatic_pass": all(gates.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq0s3_source_duplicate_adjudication.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "target_pixels_accessed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKDuplicateAdjudicationError",
    "compare_source_images",
    "run_adjudication",
]
