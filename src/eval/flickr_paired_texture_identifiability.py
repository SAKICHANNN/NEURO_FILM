"""BO5 paired capture-chain texture identifiability audit.

The audit deliberately measures only a display-chain residual.  It does not
identify emulsion grain, scanner noise, sharpening, compression, or a stock.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bo5-flickr-paired-texture-identifiability.v1"


class FlickrPairedTextureError(ValueError):
    """Raised when the frozen BO5 inputs or analysis contract drift."""


def _load_json_exact(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise FlickrPairedTextureError(f"parent hash drift: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrPairedTextureError("invalid BO5 contract")
    registration = _load_json_exact(root, config["parents"]["registration_report"])
    manifest = _load_json_exact(root, config["parents"]["download_manifest"])
    if (
        not registration.get("automatic_pass")
        or registration.get("stable_evidence_id")
        != config["parents"]["registration_report"]["required_stable_evidence_id"]
    ):
        raise FlickrPairedTextureError("registration parent did not pass exactly")
    return registration, manifest


def _load_rgb(path: Path, digest: str) -> np.ndarray:
    if sha256_file(path) != digest:
        raise FlickrPairedTextureError(f"pixel hash drift: {path.as_posix()}")
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def _opponent_log(encoded: np.ndarray, floor: float) -> np.ndarray:
    log_rgb = np.log(np.maximum(np.asarray(encoded, dtype=np.float64), floor))
    return np.stack(
        (
            (log_rgb[..., 0] + 2.0 * log_rgb[..., 1] + log_rgb[..., 2]) / 4.0,
            log_rgb[..., 0] - log_rgb[..., 1],
            log_rgb[..., 2] - log_rgb[..., 1],
        ),
        axis=-1,
    )


def _highpass(opponent: np.ndarray, prefilter_sigma: float, highpass_sigma: float) -> np.ndarray:
    values = np.empty_like(opponent, dtype=np.float64)
    for channel in range(3):
        base = cv2.GaussianBlur(
            opponent[..., channel], (0, 0), prefilter_sigma, borderType=cv2.BORDER_REFLECT_101
        )
        low = cv2.GaussianBlur(base, (0, 0), highpass_sigma, borderType=cv2.BORDER_REFLECT_101)
        values[..., channel] = base - low
    return values


def _radial_psd(patches: np.ndarray, edges: np.ndarray) -> np.ndarray:
    size = int(patches.shape[1])
    window_1d = np.hanning(size)
    window = window_1d[:, None] * window_1d[None, :]
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    spectra: list[np.ndarray] = []
    for patch in patches:
        fft = np.fft.rfft2(patch * window)
        power = np.square(np.abs(fft)) / max(float(np.sum(window * window)), 1e-12)
        bands = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = (radius >= low) & (radius < high)
            bands.append(float(np.mean(power[mask])) if np.any(mask) else 0.0)
        spectra.append(np.asarray(bands, dtype=np.float64))
    return np.mean(np.stack(spectra), axis=0)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-20 else float(np.dot(left, right) / denominator)


def _block_ratio(patches: np.ndarray) -> float:
    horizontal = np.abs(np.diff(patches, axis=2))
    vertical = np.abs(np.diff(patches, axis=1))
    x_index = np.arange(1, patches.shape[2])
    y_index = np.arange(1, patches.shape[1])
    boundary_x = x_index % 8 == 0
    boundary_y = y_index % 8 == 0
    boundary = np.concatenate((horizontal[:, :, boundary_x].ravel(), vertical[:, boundary_y, :].ravel()))
    ordinary = np.concatenate((horizontal[:, :, ~boundary_x].ravel(), vertical[:, ~boundary_y, :].ravel()))
    return float(np.mean(boundary) / max(float(np.mean(ordinary)), 1e-12))


def analyze_pair(
    digital_u8: np.ndarray,
    film_u8: np.ndarray,
    homography: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return deterministic scene-level texture facts for one registered pair."""

    cv2.setNumThreads(1)
    height, width = film_u8.shape[:2]
    digital = digital_u8.astype(np.float64) / 255.0
    film = film_u8.astype(np.float64) / 255.0
    warped = cv2.warpPerspective(digital, homography, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(
        np.ones(digital_u8.shape[:2], dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    erosion = int(contract["valid_mask_erosion_pixels"])
    if erosion:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((erosion, erosion), np.uint8)).astype(bool)
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    valid &= np.all((warped > low) & (warped < high), axis=2)
    valid &= np.all((film > low) & (film < high), axis=2)

    digital_opp = _opponent_log(warped, float(contract["log_floor"]))
    film_opp = _opponent_log(film, float(contract["log_floor"]))
    digital_hp = _highpass(digital_opp, float(contract["prefilter_sigma_pixels"]), float(contract["highpass_sigma_pixels"]))
    film_hp = _highpass(film_opp, float(contract["prefilter_sigma_pixels"]), float(contract["highpass_sigma_pixels"]))
    digital_grad = cv2.magnitude(
        cv2.Sobel(digital_opp[..., 0], cv2.CV_64F, 1, 0),
        cv2.Sobel(digital_opp[..., 0], cv2.CV_64F, 0, 1),
    )
    film_grad = cv2.magnitude(
        cv2.Sobel(film_opp[..., 0], cv2.CV_64F, 1, 0),
        cv2.Sobel(film_opp[..., 0], cv2.CV_64F, 0, 1),
    )
    gradient = np.maximum(digital_grad, film_grad)
    size = int(contract["patch_size_pixels"])
    stride = int(contract["patch_stride_pixels"])
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            patch_valid = valid[y : y + size, x : x + size]
            if bool(np.all(patch_valid)):
                score = float(np.quantile(gradient[y : y + size, x : x + size], 0.9))
                candidates.append((score, y, x))
    if len(candidates) < int(contract["minimum_patches_per_scene"]):
        raise FlickrPairedTextureError("too few fully valid texture patches")
    scores = np.asarray([row[0] for row in candidates], dtype=np.float64)
    ordered = sorted(candidates)
    quantile_count = int(np.ceil(len(ordered) * float(contract["maximum_patch_gradient_quantile"])))
    selected_count = max(int(contract["minimum_patches_per_scene"]), quantile_count)
    selected = ordered[: min(selected_count, int(contract["maximum_patches_per_scene"]))]
    cutoff = float(selected[-1][0])
    if len(selected) < int(contract["minimum_patches_per_scene"]):
        raise FlickrPairedTextureError("too few low-gradient texture patches")

    slices = [(slice(y, y + size), slice(x, x + size)) for _, y, x in selected]
    digital_patches = np.stack([digital_hp[yy, xx] for yy, xx in slices])
    film_patches = np.stack([film_hp[yy, xx] for yy, xx in slices])
    gradient_patches = np.stack([gradient[yy, xx] for yy, xx in slices])
    digital_flat = digital_patches.reshape(-1, 3)
    film_flat = film_patches.reshape(-1, 3)
    alpha = np.sum(digital_flat * film_flat, axis=0) / np.maximum(np.sum(digital_flat * digital_flat, axis=0), 1e-20)
    alpha = np.clip(alpha, 0.0, 2.0)
    residual = film_patches - digital_patches * alpha[None, None, None, :]
    shifted = np.roll(digital_patches, int(contract["shift_control_pixels"]), axis=2)
    shifted_residual = film_patches - shifted * alpha[None, None, None, :]
    residual_energy = np.mean(np.square(residual), axis=(0, 1, 2))
    film_energy = np.mean(np.square(film_patches), axis=(0, 1, 2))
    digital_energy = np.mean(np.square(digital_patches), axis=(0, 1, 2))
    edges = np.asarray(contract["radial_frequency_edges_nyquist"], dtype=np.float64)
    film_psd = _radial_psd(film_patches[..., 0], edges)
    digital_psd = _radial_psd(digital_patches[..., 0], edges)
    delta_psd = film_psd - digital_psd
    chroma_residual_energy = float(np.mean(residual_energy[1:]))
    return {
        "patches": len(selected),
        "gradient_cutoff": cutoff,
        "predictive_scale_log_luma_rg_bg": alpha.tolist(),
        "film_highpass_energy_log_luma_rg_bg": film_energy.tolist(),
        "digital_highpass_energy_log_luma_rg_bg": digital_energy.tolist(),
        "residual_energy_log_luma_rg_bg": residual_energy.tolist(),
        "film_to_digital_luma_highpass_energy_ratio": float(film_energy[0] / max(float(digital_energy[0]), 1e-20)),
        "correct_to_shifted_residual_rms_ratio": float(
            np.sqrt(np.mean(np.square(residual))) / max(float(np.sqrt(np.mean(np.square(shifted_residual)))), 1e-20)
        ),
        "residual_chroma_to_luma_energy_ratio": float(chroma_residual_energy / max(float(residual_energy[0]), 1e-20)),
        "residual_edge_correlation": _safe_corr(np.abs(residual[..., 0]), gradient_patches),
        "film_jpeg_block_ratio": _block_ratio(film_patches[..., 0]),
        "digital_jpeg_block_ratio": _block_ratio(digital_patches[..., 0]),
        "film_luma_radial_psd": film_psd.tolist(),
        "digital_luma_radial_psd": digital_psd.tolist(),
        "film_minus_digital_luma_radial_psd": delta_psd.tolist(),
        "positive_delta_psd_bin_fraction": float(np.mean(delta_psd > 0.0)),
    }


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(row["texture"][field]) for row in rows]))


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registration, manifest = validate_inputs(root, config)
    manifest_rows = {str(row["local_path"]): row for row in manifest["rows"]}
    accepted = [row for row in registration["pairs"] if row["diagnostics"]["registration_gate_passed"]]
    development = [row for row in accepted if int(row["scene_id"]) % 4 != 0]
    confirmation = [row for row in accepted if int(row["scene_id"]) % 4 == 0]
    data_root = root / str(config["data_root"])
    results: list[dict[str, Any]] = []
    for row in sorted(development, key=lambda item: str(item["pair_id"])):
        digital_meta = manifest_rows[str(row["digital_local_path"])]
        film_meta = manifest_rows[str(row["film_local_path"])]
        digital = _load_rgb(data_root / str(row["digital_local_path"]), str(digital_meta["sha256"]))
        film = _load_rgb(data_root / str(row["film_local_path"]), str(film_meta["sha256"]))
        texture = analyze_pair(
            digital,
            film,
            np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["analysis"],
        )
        results.append(
            {
                "pair_id": row["pair_id"],
                "family_id": row["family_id"],
                "scene_id": row["scene_id"],
                "digital_sha256": digital_meta["sha256"],
                "film_sha256": film_meta["sha256"],
                "texture": texture,
            }
        )

    families = sorted({str(row["family_id"]) for row in results})
    counts = Counter(str(row["family_id"]) for row in results)
    pooled_delta = np.median(
        np.asarray([row["texture"]["film_minus_digital_luma_radial_psd"] for row in results], dtype=np.float64),
        axis=0,
    )
    pooled_positive = np.maximum(pooled_delta, 0.0)
    family_delta: dict[str, list[float]] = {}
    family_cosine: dict[str, float] = {}
    family_energy_ratio: dict[str, float] = {}
    for family in families:
        rows = [row for row in results if row["family_id"] == family]
        delta = np.median(
            np.asarray([row["texture"]["film_minus_digital_luma_radial_psd"] for row in rows], dtype=np.float64),
            axis=0,
        )
        family_delta[family] = delta.tolist()
        family_cosine[family] = _cosine(np.maximum(delta, 0.0), pooled_positive)
        family_energy_ratio[family] = _median(rows, "film_to_digital_luma_highpass_energy_ratio")

    gates = config["evaluation"]
    metrics = {
        "development_scenes": len(results),
        "sealed_confirmation_scenes_not_loaded": len(confirmation),
        "scenes_per_family": dict(sorted(counts.items())),
        "minimum_patches_in_any_scene": min(int(row["texture"]["patches"]) for row in results),
        "median_correct_to_shifted_residual_rms_ratio": _median(results, "correct_to_shifted_residual_rms_ratio"),
        "median_film_to_digital_luma_highpass_energy_ratio": _median(results, "film_to_digital_luma_highpass_energy_ratio"),
        "per_family_median_film_to_digital_luma_highpass_energy_ratio": family_energy_ratio,
        "pooled_film_minus_digital_luma_radial_psd": pooled_delta.tolist(),
        "positive_film_minus_digital_luma_psd_bin_fraction": float(np.mean(pooled_delta > 0.0)),
        "per_family_positive_delta_psd_cosine_to_pooled": family_cosine,
        "per_family_film_minus_digital_luma_radial_psd": family_delta,
        "median_residual_edge_correlation": _median(results, "residual_edge_correlation"),
        "median_film_jpeg_block_ratio": _median(results, "film_jpeg_block_ratio"),
        "median_digital_jpeg_block_ratio": _median(results, "digital_jpeg_block_ratio"),
        "median_residual_chroma_to_luma_energy_ratio": _median(results, "residual_chroma_to_luma_energy_ratio"),
    }
    checks = {
        "minimum_scenes": len(results) >= int(gates["minimum_scenes"]),
        "minimum_scenes_per_family": all(counts[family] >= int(gates["minimum_scenes_per_family"]) for family in families),
        "minimum_patches_per_scene": metrics["minimum_patches_in_any_scene"] >= int(config["analysis"]["minimum_patches_per_scene"]),
        "alignment_control": metrics["median_correct_to_shifted_residual_rms_ratio"] <= float(gates["maximum_median_correct_to_shifted_residual_rms_ratio"]),
        "material_luma_excess": metrics["median_film_to_digital_luma_highpass_energy_ratio"] >= float(gates["minimum_median_film_to_digital_luma_highpass_energy_ratio"]),
        "every_family_luma_excess": all(value >= float(gates["minimum_each_family_median_film_to_digital_luma_highpass_energy_ratio"]) for value in family_energy_ratio.values()),
        "positive_delta_support": metrics["positive_film_minus_digital_luma_psd_bin_fraction"] >= float(gates["minimum_positive_film_minus_digital_luma_psd_bin_fraction"]),
        "cross_family_delta_shape": all(value >= float(gates["minimum_each_family_positive_delta_psd_cosine_to_pooled"]) for value in family_cosine.values()),
        "edge_nuisance_control": metrics["median_residual_edge_correlation"] <= float(gates["maximum_median_residual_edge_correlation"]),
        "jpeg_block_control": metrics["median_film_jpeg_block_ratio"] <= float(gates["maximum_median_film_jpeg_block_ratio"]),
        "chroma_lower_bound": metrics["median_residual_chroma_to_luma_energy_ratio"] >= float(gates["minimum_median_residual_chroma_to_luma_energy_ratio"]),
        "chroma_upper_bound": metrics["median_residual_chroma_to_luma_energy_ratio"] <= float(gates["maximum_median_residual_chroma_to_luma_energy_ratio"]),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bo5-flickr-paired-texture-identifiability-report.v1",
        "node": config["node"],
        "parent_registration_sha256": config["parents"]["registration_report"]["sha256"],
        "parent_manifest_sha256": config["parents"]["download_manifest"]["sha256"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "branch": config["branches"]["pass" if automatic_pass else "fail"],
        "rows": results,
        "confirmation_pixels_loaded": False,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}
