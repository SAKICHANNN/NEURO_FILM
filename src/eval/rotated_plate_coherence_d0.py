"""Rotated-plate transfer of the frozen P6AT coherence discriminator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.eval.scorpion_cross_scanner_coherence_d0 import _coherence, _spectra

SCHEMA = "neuro-film.u6-p6av-rotated-plate-coherence-d0-contract.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P6AV contract")
    return value


def _load_u16(path: Path, source: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    payload = path.read_bytes()
    if len(payload) != int(source["bytes"]):
        raise ValueError("P6AV source byte count drift")
    if "sha256" in source and _sha(payload) != source["sha256"]:
        raise ValueError("P6AV source SHA-256 drift")
    if hashlib.md5(payload).hexdigest() != source["md5"]:
        raise ValueError("P6AV source MD5 drift")
    with tifffile.TiffFile(path) as image:
        required_pages = int(source.get("required_pages", 1))
        primary_page_index = int(source.get("primary_page_index", 0))
        if len(image.pages) != required_pages:
            raise ValueError("P6AV source page count drift")
        if not 0 <= primary_page_index < required_pages:
            raise ValueError("P6AV primary page index drift")
        for index in source.get("required_reduced_page_indices", []):
            if not image.pages[int(index)].is_reduced:
                raise ValueError("P6AV reduced page structure drift")
        values = image.pages[primary_page_index].asarray()
    if values.dtype != np.uint16 or values.ndim != 2:
        raise ValueError("P6AV source is not grayscale uint16")
    return values, {
        "path": source["path"],
        "bytes": len(payload),
        "sha256": _sha(payload),
        "shape": [int(v) for v in values.shape],
        "dtype": str(values.dtype),
        "pages": required_pages,
        "primary_page_index": primary_page_index,
    }


def _registration_input(
    values: np.ndarray, maximum_side: int
) -> tuple[np.ndarray, np.ndarray]:
    gray = (values >> 8).astype(np.uint8)
    scale = min(1.0, maximum_side / max(gray.shape))
    shape = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    small = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA)
    transform = np.array(
        [
            [shape[0] / gray.shape[1], 0.0, 0.0],
            [0.0, shape[1] / gray.shape[0], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return small, transform


def _register_orientation(
    reference: np.ndarray, rotated: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    detector = cv2.SIFT_create(nfeatures=int(config["maximum_features"]))
    ref_small, ref_scale = _registration_input(
        reference, int(config["maximum_side_pixels"])
    )
    ref_keys, ref_desc = detector.detectAndCompute(ref_small, None)
    results: list[
        tuple[tuple[float, float, int], np.ndarray | None, dict[str, Any]]
    ] = []
    for degrees in config["orientation_candidates_degrees"]:
        candidate = np.rot90(rotated, k=1 if int(degrees) == 90 else -1)
        cand_small, cand_scale = _registration_input(
            candidate, int(config["maximum_side_pixels"])
        )
        cand_keys, cand_desc = detector.detectAndCompute(cand_small, None)
        diagnostics: dict[str, Any] = {"orientation_degrees": int(degrees)}
        homography = None
        if ref_desc is not None and cand_desc is not None:
            pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(cand_desc, ref_desc, k=2)
            good = [
                a
                for a, b in pairs
                if a.distance < float(config["ratio_threshold"]) * b.distance
            ]
            diagnostics["good_matches"] = len(good)
            if len(good) >= 4:
                source_xy = np.float32([cand_keys[m.queryIdx].pt for m in good])
                target_xy = np.float32([ref_keys[m.trainIdx].pt for m in good])
                small_h, mask = cv2.findHomography(
                    source_xy,
                    target_xy,
                    cv2.RANSAC,
                    float(config["ransac_reprojection_threshold_pixels"]),
                )
                if small_h is not None and mask is not None:
                    inliers = mask.ravel().astype(bool)
                    projected = cv2.perspectiveTransform(
                        source_xy[:, None, :], small_h
                    )[:, 0, :]
                    errors = np.linalg.norm(projected - target_xy, axis=1)[inliers]
                    diagnostics.update(
                        {
                            "inliers": int(inliers.sum()),
                            "inlier_fraction": float(inliers.mean()),
                            "median_inlier_error_pixels": float(np.median(errors)),
                            "p95_inlier_error_pixels": float(np.quantile(errors, 0.95)),
                        }
                    )
                    homography = np.linalg.inv(ref_scale) @ small_h @ cand_scale
        diagnostics.setdefault("good_matches", 0)
        diagnostics.setdefault("inliers", 0)
        diagnostics.setdefault("inlier_fraction", 0.0)
        diagnostics.setdefault("median_inlier_error_pixels", float("inf"))
        diagnostics.setdefault("p95_inlier_error_pixels", float("inf"))
        # The final term implements the frozen +90 tie break.
        key = (
            float(diagnostics["inliers"]),
            float(diagnostics["inlier_fraction"]),
            1 if int(degrees) == 90 else 0,
        )
        results.append((key, homography, diagnostics))
    _, homography, selected_row = max(results, key=lambda item: item[0])
    selected = dict(selected_row)
    selected["candidates"] = [dict(item[2]) for item in results]
    return homography, selected


def _midrank(values: np.ndarray) -> np.ndarray:
    counts = np.bincount(values.ravel(), minlength=65536)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    mapping = (starts + 0.5 * counts) / values.size
    return mapping[values]


def evaluate(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    parent_spec = contract["parents"]["method_evidence"]
    parent_bytes = (root / parent_spec["path"]).read_bytes()
    parent = json.loads(parent_bytes)
    if (
        _sha(parent_bytes) != parent_spec["sha256"]
        or parent.get("status") != parent_spec["required_status"]
        or parent.get("formal_replay", {}).get("stable_evidence_id")
        != parent_spec["required_stable_evidence_id"]
    ):
        raise ValueError("P6AV method parent drift")
    if "p6av_evidence" in contract["parents"]:
        prior_spec = contract["parents"]["p6av_evidence"]
        prior_bytes = (root / prior_spec["path"]).read_bytes()
        prior = json.loads(prior_bytes)
        if (
            _sha(prior_bytes) != prior_spec["sha256"]
            or prior.get("status") != prior_spec["required_status"]
        ):
            raise ValueError("P6AW predecessor evidence drift")
    source = contract["source"]
    reference, reference_facts = _load_u16(
        root / source["reference"]["path"], source["reference"]
    )
    rotated, rotated_facts = _load_u16(
        root / source["rotated"]["path"], source["rotated"]
    )
    reference_shape = tuple(
        int(v) for v in source.get("reference_shape", source.get("required_shape"))
    )
    rotated_shape = tuple(
        int(v) for v in source.get("rotated_shape", source.get("required_shape"))
    )
    if reference.shape != reference_shape or rotated.shape != rotated_shape:
        raise ValueError("P6AV source shape drift")
    homography, registration = _register_orientation(
        reference, rotated, contract["registration"]
    )
    reg = contract["registration"]
    registration_pass = bool(
        homography is not None
        and registration["good_matches"] >= int(reg["minimum_good_matches"])
        and registration["inlier_fraction"] >= float(reg["minimum_inlier_fraction"])
        and registration["median_inlier_error_pixels"]
        <= float(reg["maximum_median_inlier_reprojection_error_pixels"])
        and registration["p95_inlier_error_pixels"]
        <= float(reg["maximum_p95_inlier_reprojection_error_pixels"])
    )
    registration["automatic_pass"] = registration_pass
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {"patches": 0}
    if registration_pass and homography is not None:
        oriented = np.rot90(
            rotated, k=1 if registration["orientation_degrees"] == 90 else -1
        )
        aligned = cv2.warpPerspective(
            oriented,
            homography,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpPerspective(
            np.ones(oriented.shape, dtype=np.uint8),
            homography,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        erosion = int(reg["mask_erosion_pixels"])
        valid = cv2.erode(valid, np.ones((2 * erosion + 1, 2 * erosion + 1), np.uint8))
        analysis = contract["analysis"]
        crop = int(analysis["common_center_crop_size"])
        y0 = (reference.shape[0] - crop) // 2
        x0 = (reference.shape[1] - crop) // 2
        if not np.all(valid[y0 : y0 + crop, x0 : x0 + crop]):
            raise ValueError("P6AV common crop is outside registered support")
        ref_rank = _midrank(reference)[y0 : y0 + crop, x0 : x0 + crop]
        aligned_rank = _midrank(aligned)[y0 : y0 + crop, x0 : x0 + crop]
        patch = int(analysis["patch_size"])
        ref_patches = [
            ref_rank[y : y + patch, x : x + patch]
            for y in range(0, crop, patch)
            for x in range(0, crop, patch)
        ]
        aligned_patches = [
            aligned_rank[y : y + patch, x : x + patch]
            for y in range(0, crop, patch)
            for x in range(0, crop, patch)
        ]
        spectra_config = {**analysis, "crop_size": patch}
        ref_spectra = [_spectra(value, spectra_config) for value in ref_patches]
        aligned_spectra = [_spectra(value, spectra_config) for value in aligned_patches]
        dy, dx = map(int, analysis["shift_control_pixels"])
        shifted_spectra = [
            _spectra(np.roll(value, (dy, dx), axis=(0, 1)), spectra_config)
            for value in aligned_patches
        ]
        for index, left in enumerate(ref_spectra):
            rows.append(
                {
                    "patch_index": index,
                    "correct": _coherence(left, aligned_spectra[index], spectra_config),
                    "wrong": _coherence(
                        left,
                        aligned_spectra[(index + 1) % len(aligned_spectra)],
                        spectra_config,
                    ),
                    "shifted": _coherence(left, shifted_spectra[index], spectra_config),
                }
            )
        correct = np.array([row["correct"] for row in rows])
        wrong = np.array([row["wrong"] for row in rows])
        shifted = np.array([row["shifted"] for row in rows])
        tiny = np.finfo(np.float64).tiny
        aggregate = {
            "patches": len(rows),
            "correct_vs_wrong_win_rate": float(np.mean(correct > wrong)),
            "correct_vs_shifted_win_rate": float(np.mean(correct > shifted)),
            "median_correct_to_wrong_ratio": float(
                np.median(correct / np.maximum(wrong, tiny))
            ),
            "median_correct_to_shifted_ratio": float(
                np.median(correct / np.maximum(shifted, tiny))
            ),
            "median_correct_coherence": float(np.median(correct)),
        }
    gates = contract["gates"]
    coherence_pass = bool(
        registration_pass
        and aggregate["patches"] == int(gates["required_patches"])
        and aggregate["correct_vs_wrong_win_rate"]
        >= float(gates["minimum_correct_vs_wrong_win_rate"])
        and aggregate["correct_vs_shifted_win_rate"]
        >= float(gates["minimum_correct_vs_shifted_win_rate"])
        and aggregate["median_correct_to_wrong_ratio"]
        >= float(gates["minimum_median_correct_to_wrong_ratio"])
        and aggregate["median_correct_to_shifted_ratio"]
        >= float(gates["minimum_median_correct_to_shifted_ratio"])
        and aggregate["median_correct_coherence"]
        >= float(gates["minimum_median_correct_coherence"])
    )
    scientific = {
        "source": {"reference": reference_facts, "rotated": rotated_facts},
        "registration": registration,
        "rows": rows,
        "aggregate": aggregate,
        "automatic_pass": coherence_pass,
        "decision": contract["decision_if_pass"]
        if coherence_pass
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro-film.u6-p6av-rotated-plate-coherence-d0-report.v1",
        "experiment_id": contract["experiment_id"],
        **scientific,
        "stable_evidence_id": _sha(_canonical(scientific)),
    }
