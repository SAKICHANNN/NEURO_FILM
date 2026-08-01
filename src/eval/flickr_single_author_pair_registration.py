"""Deterministic registration feasibility for the BO1 weak film/digital pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.flickr_single_author_pair_acquisition import atomic_json, canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bo2-flickr-single-author-pair-registration.v1"


class FlickrPairRegistrationError(ValueError):
    """Raised when the frozen registration inputs or contract drift."""


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrPairRegistrationError("invalid BO2 contract")
    loaded: dict[str, dict[str, Any]] = {}
    for name, parent in config["parents"].items():
        path = root / str(parent["path"])
        if sha256_file(path) != parent["sha256"]:
            raise FlickrPairRegistrationError(f"parent hash drift: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    integrity = loaded["integrity_report"]
    if (
        not integrity.get("automatic_pass")
        or integrity.get("stable_evidence_id")
        != config["parents"]["integrity_report"]["required_stable_evidence_id"]
    ):
        raise FlickrPairRegistrationError("integrity parent did not pass exactly")
    return integrity, loaded["download_manifest"]


def _load_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise FlickrPairRegistrationError(f"pixel hash drift: {path.as_posix()}")
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    return rgb


def _gray(rgb: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(
        clipLimit=float(contract["clahe_clip_limit"]),
        tileGridSize=tuple(int(value) for value in contract["clahe_tile_grid"]),
    ).apply(gray)


def register_pair(
    digital_rgb: np.ndarray, film_rgb: np.ndarray, contract: Mapping[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate one deterministic digital-to-film homography and diagnostics."""

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    digital_gray = _gray(digital_rgb, contract)
    film_gray = _gray(film_rgb, contract)
    sift = cv2.SIFT_create(nfeatures=int(contract["maximum_features"]))
    digital_points, digital_desc = sift.detectAndCompute(digital_gray, None)
    film_points, film_desc = sift.detectAndCompute(film_gray, None)
    base = {
        "digital_keypoints": len(digital_points),
        "film_keypoints": len(film_points),
    }
    if digital_desc is None or film_desc is None:
        return None, {**base, "failure_reason": "missing_descriptors", "registration_gate_passed": False}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(digital_desc, film_desc, k=2)
    reverse = matcher.knnMatch(film_desc, digital_desc, k=2)
    ratio = float(contract["ratio_threshold"])
    forward_good = {m.queryIdx: m for m, n in forward if m.distance < ratio * n.distance}
    reverse_good = {m.queryIdx: m for m, n in reverse if m.distance < ratio * n.distance}
    mutual = [
        match
        for query, match in sorted(forward_good.items())
        if match.trainIdx in reverse_good and reverse_good[match.trainIdx].trainIdx == query
    ]
    if len(mutual) < int(contract["minimum_mutual_good_matches"]):
        return None, {
            **base,
            "mutual_good_matches": len(mutual),
            "failure_reason": "insufficient_mutual_matches",
            "registration_gate_passed": False,
        }
    digital_xy = np.float32([digital_points[item.queryIdx].pt for item in mutual])
    film_xy = np.float32([film_points[item.trainIdx].pt for item in mutual])
    homography, mask = cv2.findHomography(
        digital_xy,
        film_xy,
        cv2.RANSAC,
        float(contract["ransac_reprojection_threshold_pixels"]),
    )
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None, {
            **base,
            "mutual_good_matches": len(mutual),
            "failure_reason": "homography_failed",
            "registration_gate_passed": False,
        }
    inlier = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(digital_xy[:, None, :], homography)[:, 0, :]
    errors = np.linalg.norm(projected - film_xy, axis=1)[inlier]
    source_mask = np.ones(digital_gray.shape, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask,
        homography,
        (film_gray.shape[1], film_gray.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    overlap = float(np.mean(warped_mask > 0))
    diagnostics = {
        **base,
        "mutual_good_matches": len(mutual),
        "inliers": int(np.count_nonzero(inlier)),
        "inlier_fraction": float(np.mean(inlier)),
        "median_inlier_reprojection_error_pixels": float(np.median(errors)),
        "p95_inlier_reprojection_error_pixels": float(np.quantile(errors, 0.95)),
        "target_overlap_fraction": overlap,
        "homography_digital_to_film": homography.tolist(),
    }
    checks = {
        "minimum_mutual_good_matches": len(mutual) >= int(contract["minimum_mutual_good_matches"]),
        "minimum_inliers": int(np.count_nonzero(inlier)) >= int(contract["minimum_inliers"]),
        "minimum_inlier_fraction": float(np.mean(inlier)) >= float(contract["minimum_inlier_fraction"]),
        "median_reprojection_error": float(np.median(errors))
        <= float(contract["maximum_median_inlier_reprojection_error_pixels"]),
        "p95_reprojection_error": float(np.quantile(errors, 0.95))
        <= float(contract["maximum_p95_inlier_reprojection_error_pixels"]),
        "target_overlap_fraction": overlap >= float(contract["minimum_target_overlap_fraction"]),
    }
    passed = all(checks.values())
    return homography.astype(np.float64), {
        **diagnostics,
        "checks": checks,
        "failure_reason": None if passed else "registration_quality_gate",
        "registration_gate_passed": passed,
    }


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all exact pairs without using colour pixels for fitting."""

    _, manifest = validate_inputs(root, config)
    data_root = root / str(config["data_root"])
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in manifest["rows"]:
        grouped.setdefault(str(row["pair_id"]), {})[str(row["role"])] = row
    if len(grouped) != int(config["dataset_gates"]["expected_pairs"]):
        raise FlickrPairRegistrationError("pair count drift")
    results: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        roles = grouped[pair_id]
        if set(roles) != {"digital", "film"}:
            raise FlickrPairRegistrationError("pair role drift")
        digital = roles["digital"]
        film = roles["film"]
        digital_rgb = _load_rgb(data_root / str(digital["local_path"]), str(digital["sha256"]))
        film_rgb = _load_rgb(data_root / str(film["local_path"]), str(film["sha256"]))
        homography, diagnostics = register_pair(digital_rgb, film_rgb, config["registration"])
        results.append(
            {
                "pair_id": pair_id,
                "family_id": digital["family_id"],
                "scene_id": digital["scene_id"],
                "digital_local_path": digital["local_path"],
                "digital_sha256": digital["sha256"],
                "film_local_path": film["local_path"],
                "film_sha256": film["sha256"],
                "diagnostics": diagnostics,
            }
        )
    passed_rows = [row for row in results if row["diagnostics"]["registration_gate_passed"]]
    per_family = Counter(row["family_id"] for row in passed_rows)
    gates = config["dataset_gates"]
    families = sorted({row["family_id"] for row in results})
    checks = {
        "expected_pairs": len(results) == int(gates["expected_pairs"]),
        "minimum_registered_pairs": len(passed_rows) >= int(gates["minimum_registered_pairs"]),
        "required_families": len(families) == int(gates["required_families"]),
        "minimum_registered_pairs_per_family": all(
            per_family[family] >= int(gates["minimum_registered_pairs_per_family"])
            for family in families
        ),
    }
    stable = {
        "schema": "neuro-film.u5-r2bo2-flickr-single-author-pair-registration-report.v1",
        "node": config["node"],
        "parent_report_sha256": config["parents"]["integrity_report"]["sha256"],
        "parent_manifest_sha256": config["parents"]["download_manifest"]["sha256"],
        "metrics": {
            "pairs": len(results),
            "registered_pairs": len(passed_rows),
            "registered_fraction": len(passed_rows) / len(results),
            "families": len(families),
            "registered_pairs_per_family": dict(sorted(per_family.items())),
            "failure_reasons": dict(
                sorted(
                    Counter(
                        row["diagnostics"]["failure_reason"] or "passed"
                        for row in results
                    ).items()
                )
            ),
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "branch": config["branches"]["pass" if all(checks.values()) else "fail"],
        "pairs": results,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(canonical_bytes(stable)).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def render_contact_sheets(root: Path, report: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render diagnostic digital/film/aligned-overlay sheets after auto pass."""

    if not report.get("automatic_pass"):
        raise FlickrPairRegistrationError("contact sheets require automatic pass")
    data_root = root / str(config["data_root"])
    output_root = root / str(config["visual_review"]["contact_sheet_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in report["pairs"]:
        rows_by_family.setdefault(str(row["family_id"]), []).append(row)
    outputs: list[dict[str, Any]] = []
    cell_w, cell_h = 300, 220
    for family, rows in sorted(rows_by_family.items()):
        canvas = Image.new("RGB", (cell_w * 3, cell_h * len(rows)), "#181818")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(sorted(rows, key=lambda item: int(item["scene_id"]))):
            y = index * cell_h
            digital = _load_rgb(data_root / str(row["digital_local_path"]), str(row["digital_sha256"]))
            film = _load_rgb(data_root / str(row["film_local_path"]), str(row["film_sha256"]))
            homography = row["diagnostics"].get("homography_digital_to_film")
            if homography is None:
                aligned = np.zeros_like(film)
            else:
                aligned = cv2.warpPerspective(
                    digital,
                    np.asarray(homography, dtype=np.float64),
                    (film.shape[1], film.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )
            overlay = np.rint(0.5 * aligned.astype(np.float32) + 0.5 * film.astype(np.float32)).astype(np.uint8)
            for column, (name, image) in enumerate((("digital", digital), ("film", film), ("overlay", overlay))):
                preview = ImageOps.contain(Image.fromarray(image), (cell_w - 8, cell_h - 30))
                canvas.paste(preview, (column * cell_w + (cell_w - preview.width) // 2, y + 4))
                draw.text((column * cell_w + 6, y + cell_h - 22), f"{row['scene_id']:02d} {name}", fill="white")
            status = "PASS" if row["diagnostics"]["registration_gate_passed"] else "FAIL"
            draw.text((cell_w * 3 - 56, y + cell_h - 22), status, fill="#6cff6c" if status == "PASS" else "#ff6c6c")
        path = output_root / f"{family}.jpg"
        canvas.save(path, quality=92, subsampling=0)
        outputs.append({"family_id": family, "path": path.as_posix(), "sha256": sha256_file(path), "pairs": len(rows)})
    return outputs


__all__ = [
    "FlickrPairRegistrationError",
    "SCHEMA",
    "evaluate",
    "register_pair",
    "render_contact_sheets",
    "validate_inputs",
]
