"""BP2 basic-normalized B&W residual identifiability audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bp2-flickr-bw-residual-identifiability.v1"


class FlickrBwResidualError(ValueError):
    """Raised when BP2 inputs, support, or contract drift."""


def _load_json(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise FlickrBwResidualError(f"parent hash drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_luma(path: Path, digest: str) -> np.ndarray:
    if sha256_file(path) != digest:
        raise FlickrBwResidualError(f"pixel hash drift: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return (rgb[..., 0] + 2.0 * rgb[..., 1] + rgb[..., 2]) / 4.0


def _tone_map(source: np.ndarray, target: np.ndarray, valid: np.ndarray, knots: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, knots)
    source_knots = np.quantile(source[valid], quantiles)
    target_knots = np.maximum.accumulate(np.quantile(target[valid], quantiles))
    unique, indices = np.unique(source_knots, return_index=True)
    mapped = np.interp(source, unique, target_knots[indices], left=target_knots[indices][0], right=target_knots[indices][-1])
    return mapped


def _highpass(values: np.ndarray, prefilter_sigma: float, highpass_sigma: float) -> np.ndarray:
    base = cv2.GaussianBlur(values, (0, 0), prefilter_sigma, borderType=cv2.BORDER_REFLECT_101)
    low = cv2.GaussianBlur(base, (0, 0), highpass_sigma, borderType=cv2.BORDER_REFLECT_101)
    return base - low


def _radial_psd(patches: np.ndarray, edges: np.ndarray) -> np.ndarray:
    size = patches.shape[1]
    window = np.hanning(size)[:, None] * np.hanning(size)[None, :]
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    result = []
    for patch in patches:
        power = np.abs(np.fft.rfft2(patch * window)) ** 2 / max(float(np.sum(window * window)), 1e-12)
        result.append(
            [
                float(np.mean(power[(radius >= low) & (radius < high)]))
                for low, high in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
    return np.mean(np.asarray(result, dtype=np.float64), axis=0)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x, y = np.asarray(left).ravel(), np.asarray(right).ravel()
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-20 else float(np.dot(left, right) / denominator)


def _block_ratio(patches: np.ndarray) -> float:
    horizontal, vertical = np.abs(np.diff(patches, axis=2)), np.abs(np.diff(patches, axis=1))
    xb = np.arange(1, patches.shape[2]) % 8 == 0
    yb = np.arange(1, patches.shape[1]) % 8 == 0
    boundary = np.concatenate((horizontal[:, :, xb].ravel(), vertical[:, yb, :].ravel()))
    ordinary = np.concatenate((horizontal[:, :, ~xb].ravel(), vertical[:, ~yb, :].ravel()))
    return float(np.mean(boundary) / max(float(np.mean(ordinary)), 1e-12))


def analyze_pair(
    digital: np.ndarray, film: np.ndarray, homography: np.ndarray, contract: Mapping[str, Any]
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    height, width = film.shape
    warped = cv2.warpPerspective(digital, homography, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(
        np.ones(digital.shape, dtype=np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    erosion = int(contract["valid_mask_erosion_pixels"])
    if erosion:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((erosion, erosion), np.uint8)).astype(bool)
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    valid &= (warped > low) & (warped < high) & (film > low) & (film < high)
    basic = _tone_map(warped, film, valid, int(contract["tone_quantiles"]))
    floor = float(contract["log_floor"])
    basic_log, film_log = np.log(np.maximum(basic, floor)), np.log(np.maximum(film, floor))
    basic_hp = _highpass(basic_log, float(contract["prefilter_sigma_pixels"]), float(contract["highpass_sigma_pixels"]))
    film_hp = _highpass(film_log, float(contract["prefilter_sigma_pixels"]), float(contract["highpass_sigma_pixels"]))
    gx = cv2.Sobel(basic_log, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(basic_log, cv2.CV_64F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    size, stride = int(contract["patch_size_pixels"]), int(contract["patch_stride_pixels"])
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            if np.all(valid[y : y + size, x : x + size]):
                candidates.append((float(np.quantile(gradient[y : y + size, x : x + size], 0.9)), y, x))
    keep = min(
        int(contract["maximum_patches_per_scene"]),
        max(int(contract["minimum_patches_per_scene"]), int(np.ceil(len(candidates) * float(contract["maximum_patch_gradient_quantile"])))),
    )
    selected = sorted(candidates)[:keep]
    if len(selected) < int(contract["minimum_patches_per_scene"]):
        raise FlickrBwResidualError("insufficient flat-patch support")
    slices = [(slice(y, y + size), slice(x, x + size)) for _, y, x in selected]
    basic_patches = np.stack([basic_hp[yy, xx] for yy, xx in slices])
    film_patches = np.stack([film_hp[yy, xx] for yy, xx in slices])
    gradients = np.stack([gradient[yy, xx] for yy, xx in slices])
    scale = float(
        np.sum(basic_patches * film_patches) / max(float(np.sum(basic_patches * basic_patches)), 1e-20)
    )
    scale = float(np.clip(scale, 0.0, 2.0))
    residual = film_patches - scale * basic_patches
    shifted = np.roll(basic_patches, int(contract["shift_control_pixels"]), axis=2)
    shifted_residual = film_patches - scale * shifted
    edges = np.asarray(contract["radial_frequency_edges_nyquist"], dtype=np.float64)
    film_psd, basic_psd = _radial_psd(film_patches, edges), _radial_psd(basic_patches, edges)
    film_energy, basic_energy = float(np.mean(film_patches**2)), float(np.mean(basic_patches**2))
    return {
        "patches": len(selected),
        "predictive_scale": scale,
        "film_highpass_energy": film_energy,
        "basic_highpass_energy": basic_energy,
        "film_to_basic_highpass_energy_ratio": film_energy / max(basic_energy, 1e-20),
        "correct_to_shifted_residual_rms_ratio": float(
            np.sqrt(np.mean(residual**2)) / max(float(np.sqrt(np.mean(shifted_residual**2))), 1e-20)
        ),
        "residual_edge_correlation": _safe_corr(np.abs(residual), gradients),
        "film_jpeg_block_ratio": _block_ratio(film_patches),
        "film_radial_psd": film_psd.tolist(),
        "basic_radial_psd": basic_psd.tolist(),
        "film_minus_basic_radial_psd": (film_psd - basic_psd).tolist(),
    }


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrBwResidualError("invalid BP2 contract")
    if re.fullmatch(r"[0-9a-f]{40}", str(config.get("software_commit", ""))) is None:
        raise FlickrBwResidualError("software commit is not frozen")
    registration = _load_json(root, config["parents"]["registration_report"])
    manifest = _load_json(root, config["parents"]["download_manifest"])
    if (
        not registration.get("automatic_pass")
        or registration.get("stable_evidence_id")
        != config["parents"]["registration_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwResidualError("registration parent did not pass exactly")
    by_scene = {int(row["scene_id"]): row for row in manifest["rows"]}
    data_root = root / str(config["data_root"])
    rows = []
    for pair in registration["pairs"]:
        if not pair["diagnostics"]["registration_gate_passed"]:
            continue
        scene = int(pair["scene_id"])
        source = by_scene[scene]
        digital = _load_luma(data_root / source["digital_local_path"], source["digital_sha256"])
        film = _load_luma(data_root / source["film_local_path"], source["film_sha256"])
        residual = analyze_pair(
            digital,
            film,
            np.asarray(pair["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["analysis"],
        )
        rows.append({"scene_id": scene, "photo_id": pair["photo_id"], "residual": residual})
    deltas = np.asarray([row["residual"]["film_minus_basic_radial_psd"] for row in rows])
    pooled = np.median(deltas, axis=0)
    loo_cosines = []
    for index in range(len(rows)):
        other = np.median(np.delete(deltas, index, axis=0), axis=0)
        loo_cosines.append(_cosine(np.maximum(deltas[index], 0.0), np.maximum(other, 0.0)))
    ratios = np.asarray([row["residual"]["film_to_basic_highpass_energy_ratio"] for row in rows])
    gates = config["evaluation"]
    metrics = {
        "scenes": len(rows),
        "minimum_patches": min(row["residual"]["patches"] for row in rows),
        "median_correct_to_shifted_residual_rms_ratio": float(
            np.median([row["residual"]["correct_to_shifted_residual_rms_ratio"] for row in rows])
        ),
        "median_film_to_basic_luma_highpass_energy_ratio": float(np.median(ratios)),
        "scene_fraction_with_luma_excess": float(np.mean(ratios >= 1.0)),
        "pooled_film_minus_basic_radial_psd": pooled.tolist(),
        "positive_film_minus_basic_psd_bin_fraction": float(np.mean(pooled > 0.0)),
        "leave_one_scene_out_positive_delta_cosines": loo_cosines,
        "loo_stable_scene_fraction": float(
            np.mean(np.asarray(loo_cosines) >= float(gates["minimum_leave_one_scene_out_positive_delta_cosine"]))
        ),
        "median_residual_edge_correlation": float(
            np.median([row["residual"]["residual_edge_correlation"] for row in rows])
        ),
        "median_film_jpeg_block_ratio": float(
            np.median([row["residual"]["film_jpeg_block_ratio"] for row in rows])
        ),
    }
    checks = {
        "required_scenes": len(rows) == int(gates["required_scenes"]),
        "minimum_patches": metrics["minimum_patches"] >= int(config["analysis"]["minimum_patches_per_scene"]),
        "alignment_control": metrics["median_correct_to_shifted_residual_rms_ratio"]
        <= float(gates["maximum_median_correct_to_shifted_residual_rms_ratio"]),
        "luma_excess": metrics["median_film_to_basic_luma_highpass_energy_ratio"]
        >= float(gates["minimum_median_film_to_basic_luma_highpass_energy_ratio"]),
        "scene_luma_consistency": metrics["scene_fraction_with_luma_excess"]
        >= float(gates["minimum_scene_fraction_with_luma_excess"]),
        "positive_psd_support": metrics["positive_film_minus_basic_psd_bin_fraction"]
        >= float(gates["minimum_positive_film_minus_basic_psd_bin_fraction"]),
        "leave_one_scene_out_shape": metrics["loo_stable_scene_fraction"]
        >= float(gates["minimum_loo_stable_scene_fraction"]),
        "edge_nuisance_control": metrics["median_residual_edge_correlation"]
        <= float(gates["maximum_median_residual_edge_correlation"]),
        "jpeg_block_control": metrics["median_film_jpeg_block_ratio"]
        <= float(gates["maximum_median_film_jpeg_block_ratio"]),
    }
    passed = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bp2-flickr-bw-residual-identifiability-report.v1",
        "node": config["node"],
        "software_commit": config["software_commit"],
        "parent_registration_sha256": config["parents"]["registration_report"]["sha256"],
        "parent_manifest_sha256": config["parents"]["download_manifest"]["sha256"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "branch": config["branches"]["pass" if passed else "fail"],
        "rows": rows,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


__all__ = ["FlickrBwResidualError", "SCHEMA", "analyze_pair", "evaluate"]
