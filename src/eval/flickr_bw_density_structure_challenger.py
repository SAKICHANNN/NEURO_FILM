"""BP3 leave-one-scene-out density-domain B&W structure challenger."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.flickr_bw_residual_identifiability import (
    _highpass,
    _load_luma,
    _radial_psd,
    _safe_corr,
    _tone_map,
)
from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bp3-flickr-bw-density-structure-challenger.v1"


class FlickrBwDensityStructureError(ValueError):
    """Raised when BP3 inputs, information flow, or numerical contracts drift."""


def _load_json(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise FlickrBwDensityStructureError(f"parent hash drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed(namespace: str, scene_id: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def synthesize_spectral_field(
    shape: tuple[int, int],
    edges: np.ndarray,
    profile: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return deterministic zero-mean random-phase field with a radial profile."""

    if len(shape) != 2 or min(shape) < 8:
        raise FlickrBwDensityStructureError("invalid synthesis shape")
    edges = np.asarray(edges, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if (
        edges.ndim != 1
        or profile.shape != (edges.size - 1,)
        or not np.all(np.isfinite(profile))
        or np.any(profile < 0.0)
        or float(np.sum(profile)) <= 0.0
    ):
        raise FlickrBwDensityStructureError("invalid radial profile")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape, dtype=np.float64)
    spectrum = np.fft.rfft2(white)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    gain = np.zeros_like(radius)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        gain[(radius >= low) & (radius < high)] = np.sqrt(profile[index])
    gain[0, 0] = 0.0
    field = np.fft.irfft2(spectrum * gain, s=shape).real
    field -= float(np.mean(field))
    standard_deviation = float(np.std(field))
    if standard_deviation <= 1e-20:
        raise FlickrBwDensityStructureError("degenerate spectral field")
    return field / standard_deviation


def safe_density_candidate(
    basic: np.ndarray,
    valid: np.ndarray,
    log_density_field: np.ndarray,
    low: float,
    high: float,
    maximum_scale: float,
) -> tuple[np.ndarray, float, float]:
    """Apply a mean-preserving density perturbation with analytic no-clip scale."""

    basic = np.asarray(basic, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    field = np.asarray(log_density_field, dtype=np.float64)
    if basic.shape != valid.shape or basic.shape != field.shape or not np.any(valid):
        raise FlickrBwDensityStructureError("candidate shape or mask mismatch")
    if not np.all(np.isfinite(basic)) or not np.all(np.isfinite(field)):
        raise FlickrBwDensityStructureError("non-finite candidate input")
    residual = np.zeros_like(basic)
    residual[valid] = basic[valid] * np.expm1(field[valid])
    residual[valid] -= float(np.mean(residual[valid]))
    positive = valid & (residual > 0.0)
    negative = valid & (residual < 0.0)
    limits = [float(maximum_scale)]
    if np.any(positive):
        limits.append(float(np.min((high - basic[positive]) / residual[positive])))
    if np.any(negative):
        limits.append(float(np.min((basic[negative] - low) / (-residual[negative]))))
    scale = max(0.0, min(limits))
    if scale > 0.0:
        scale *= 1.0 - 1e-12
    candidate = basic + scale * residual
    mean_drift = float(np.mean(candidate[valid]) - np.mean(basic[valid]))
    if np.any(candidate[valid] < low) or np.any(candidate[valid] > high):
        raise FlickrBwDensityStructureError("analytic scale violated output bounds")
    return candidate, scale, mean_drift


def _prepare_scene(
    digital: np.ndarray,
    film: np.ndarray,
    homography: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    height, width = film.shape
    warped = cv2.warpPerspective(digital, homography, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(
        np.ones(digital.shape, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    erosion = int(contract["valid_mask_erosion_pixels"])
    if erosion:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((erosion, erosion), np.uint8)).astype(bool)
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    valid &= (warped > low) & (warped < high) & (film > low) & (film < high)
    basic = _tone_map(warped, film, valid, int(contract["tone_quantiles"]))
    floor = float(contract["log_floor"])
    basic_log = np.log(np.maximum(basic, floor))
    film_log = np.log(np.maximum(film, floor))
    basic_hp = _highpass(
        basic_log,
        float(contract["prefilter_sigma_pixels"]),
        float(contract["highpass_sigma_pixels"]),
    )
    film_hp = _highpass(
        film_log,
        float(contract["prefilter_sigma_pixels"]),
        float(contract["highpass_sigma_pixels"]),
    )
    gx = cv2.Sobel(basic_log, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(basic_log, cv2.CV_64F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    size = int(contract["patch_size_pixels"])
    stride = int(contract["patch_stride_pixels"])
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            if np.all(valid[y : y + size, x : x + size]):
                score = float(np.quantile(gradient[y : y + size, x : x + size], 0.9))
                candidates.append((score, y, x))
    keep = min(
        int(contract["maximum_patches_per_scene"]),
        max(
            int(contract["minimum_patches_per_scene"]),
            int(np.ceil(len(candidates) * float(contract["maximum_patch_gradient_quantile"]))),
        ),
    )
    selected = sorted(candidates)[:keep]
    if len(selected) < int(contract["minimum_patches_per_scene"]):
        raise FlickrBwDensityStructureError("insufficient flat-patch support")
    slices = [(slice(y, y + size), slice(x, x + size)) for _, y, x in selected]
    return {
        "basic": basic,
        "film": film,
        "valid": valid,
        "basic_hp": basic_hp,
        "film_hp": film_hp,
        "gradient": gradient,
        "slices": slices,
    }


def _patches(values: np.ndarray, slices: list[tuple[slice, slice]]) -> np.ndarray:
    return np.stack([values[yy, xx] for yy, xx in slices])


def _profile_error(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum(np.abs(candidate - target)) / max(float(np.sum(target)), 1e-20))


def _scaled_field(
    scene: Mapping[str, Any],
    edges: np.ndarray,
    profile: np.ndarray,
    seed: int,
    maximum_log_standard_deviation: float,
    contract: Mapping[str, Any],
) -> np.ndarray:
    field = synthesize_spectral_field(scene["basic"].shape, edges, profile, seed)
    field_hp = _highpass(
        field,
        float(contract["prefilter_sigma_pixels"]),
        float(contract["highpass_sigma_pixels"]),
    )
    unit_psd = _radial_psd(_patches(field_hp, scene["slices"]), edges)
    scale = np.sqrt(float(np.sum(profile)) / max(float(np.sum(unit_psd)), 1e-20))
    scale = min(scale, float(maximum_log_standard_deviation))
    return field * scale


def _analyze_candidate(
    scene: Mapping[str, Any],
    profile: np.ndarray,
    namespace: str,
    scene_id: int,
    contract: Mapping[str, Any],
) -> tuple[dict[str, float | list[float]], np.ndarray]:
    edges = np.asarray(contract["radial_frequency_edges_nyquist"], dtype=np.float64)
    field = _scaled_field(
        scene,
        edges,
        profile,
        _seed(namespace, scene_id),
        float(contract["maximum_log_density_standard_deviation"]),
        contract,
    )
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    candidate, scale, mean_drift = safe_density_candidate(
        scene["basic"], scene["valid"], field, low, high, float(contract["maximum_safe_scale"])
    )
    candidate_hp = _highpass(
        np.log(np.maximum(candidate, float(contract["log_floor"]))),
        float(contract["prefilter_sigma_pixels"]),
        float(contract["highpass_sigma_pixels"]),
    )
    candidate_patches = _patches(candidate_hp, scene["slices"])
    candidate_psd = _radial_psd(candidate_patches, edges)
    film_patches = _patches(scene["film_hp"], scene["slices"])
    candidate_energy = float(np.mean(candidate_patches**2))
    film_energy = float(np.mean(film_patches**2))
    boundary = (~((scene["basic"] <= low) | (scene["basic"] >= high))) & (
        (candidate <= low) | (candidate >= high)
    )
    added = candidate_hp - scene["basic_hp"]
    return (
        {
            "safe_scale": scale,
            "mean_luma_drift": mean_drift,
            "new_boundary_fraction": float(np.mean(boundary)),
            "radial_psd": candidate_psd.tolist(),
            "highpass_energy": candidate_energy,
            "candidate_to_film_highpass_energy_ratio": candidate_energy / max(film_energy, 1e-20),
            "added_structure_edge_correlation": _safe_corr(
                np.abs(_patches(added, scene["slices"])),
                _patches(scene["gradient"], scene["slices"]),
            ),
        },
        candidate,
    )


def _evaluate_internal(
    root: Path, config: Mapping[str, Any], return_previews: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrBwDensityStructureError("invalid BP3 contract")
    if re.fullmatch(r"[0-9a-f]{40}", str(config.get("software_commit", ""))) is None:
        raise FlickrBwDensityStructureError("software commit is not frozen")
    residual_parent = _load_json(root, config["parents"]["residual_report"])
    registration = _load_json(root, config["parents"]["registration_report"])
    manifest = _load_json(root, config["parents"]["download_manifest"])
    if (
        not residual_parent.get("automatic_pass")
        or residual_parent.get("stable_evidence_id")
        != config["parents"]["residual_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwDensityStructureError("BP2 parent did not pass exactly")
    if (
        not registration.get("automatic_pass")
        or registration.get("stable_evidence_id")
        != config["parents"]["registration_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwDensityStructureError("BP1 parent did not pass exactly")
    by_scene = {int(row["scene_id"]): row for row in manifest["rows"]}
    parent_rows = {int(row["scene_id"]): row for row in residual_parent["rows"]}
    data_root = root / str(config["data_root"])
    scenes: dict[int, dict[str, Any]] = {}
    for pair in registration["pairs"]:
        if not pair["diagnostics"]["registration_gate_passed"]:
            continue
        scene_id = int(pair["scene_id"])
        source = by_scene[scene_id]
        digital = _load_luma(data_root / source["digital_local_path"], source["digital_sha256"])
        film = _load_luma(data_root / source["film_local_path"], source["film_sha256"])
        scenes[scene_id] = _prepare_scene(
            digital,
            film,
            np.asarray(pair["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["analysis"],
        )
    edges = np.asarray(config["analysis"]["radial_frequency_edges_nyquist"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    for scene_id in sorted(scenes):
        scene = scenes[scene_id]
        training_ids = [other for other in sorted(scenes) if other != scene_id]
        training_deltas = np.asarray(
            [parent_rows[other]["residual"]["film_minus_basic_radial_psd"] for other in training_ids],
            dtype=np.float64,
        )
        profile = np.maximum(np.median(training_deltas, axis=0), 0.0)
        if float(np.sum(profile)) <= 0.0:
            raise FlickrBwDensityStructureError("LOO profile has no positive support")
        white_profile = np.full_like(profile, float(np.sum(profile)) / profile.size)
        wrong_profile = profile[::-1].copy()
        wrong_profile *= float(np.sum(profile)) / max(float(np.sum(wrong_profile)), 1e-20)
        candidate, candidate_image = _analyze_candidate(
            scene,
            profile,
            str(config["analysis"]["candidate_seed_namespace"]),
            scene_id,
            config["analysis"],
        )
        white, _ = _analyze_candidate(
            scene,
            white_profile,
            str(config["analysis"]["white_control_seed_namespace"]),
            scene_id,
            config["analysis"],
        )
        wrong, _ = _analyze_candidate(
            scene,
            wrong_profile,
            str(config["analysis"]["candidate_seed_namespace"]),
            scene_id,
            config["analysis"],
        )
        film_psd = np.asarray(parent_rows[scene_id]["residual"]["film_radial_psd"], dtype=np.float64)
        basic_psd = np.asarray(parent_rows[scene_id]["residual"]["basic_radial_psd"], dtype=np.float64)
        candidate_error = _profile_error(np.asarray(candidate["radial_psd"]), film_psd)
        white_error = _profile_error(np.asarray(white["radial_psd"]), film_psd)
        wrong_error = _profile_error(np.asarray(wrong["radial_psd"]), film_psd)
        basic_error = _profile_error(basic_psd, film_psd)
        film_energy = float(parent_rows[scene_id]["residual"]["film_highpass_energy"])
        basic_energy = float(parent_rows[scene_id]["residual"]["basic_highpass_energy"])
        candidate_energy = float(candidate["highpass_energy"])
        rows.append(
            {
                "scene_id": scene_id,
                "photo_id": parent_rows[scene_id]["photo_id"],
                "training_scene_ids": training_ids,
                "held_scene_used_for_profile": False,
                "loo_positive_profile": profile.tolist(),
                "basic_psd_error": basic_error,
                "candidate_psd_error": candidate_error,
                "white_psd_error": white_error,
                "wrong_shape_psd_error": wrong_error,
                "candidate_improvement_over_basic": (basic_error - candidate_error) / max(basic_error, 1e-20),
                "candidate_improvement_over_white": (white_error - candidate_error) / max(white_error, 1e-20),
                "candidate_improvement_over_wrong_shape": (wrong_error - candidate_error) / max(wrong_error, 1e-20),
                "basic_highpass_energy_error": abs(basic_energy - film_energy) / max(film_energy, 1e-20),
                "candidate_highpass_energy_error": abs(candidate_energy - film_energy) / max(film_energy, 1e-20),
                "candidate": candidate,
                "white_control": white,
                "wrong_shape_control": wrong,
            }
        )
        if return_previews:
            previews.append(
                {
                    "scene_id": scene_id,
                    "basic": scene["basic"],
                    "candidate": candidate_image,
                    "film": scene["film"],
                    "difference": np.abs(candidate_image - scene["basic"]),
                }
            )
    improvements = np.asarray([row["candidate_improvement_over_basic"] for row in rows])
    white_improvements = np.asarray([row["candidate_improvement_over_white"] for row in rows])
    wrong_improvements = np.asarray([row["candidate_improvement_over_wrong_shape"] for row in rows])
    energy_improvements = np.asarray(
        [
            (row["basic_highpass_energy_error"] - row["candidate_highpass_energy_error"])
            / max(row["basic_highpass_energy_error"], 1e-20)
            for row in rows
        ]
    )
    metrics = {
        "scenes": len(rows),
        "median_psd_error_improvement_over_basic": float(np.median(improvements)),
        "scene_win_fraction_over_basic": float(np.mean(improvements > 0.0)),
        "median_psd_error_improvement_over_white": float(np.median(white_improvements)),
        "median_psd_error_improvement_over_wrong_shape": float(np.median(wrong_improvements)),
        "median_highpass_energy_error_improvement_over_basic": float(np.median(energy_improvements)),
        "maximum_absolute_mean_luma_drift": float(
            np.max([abs(float(row["candidate"]["mean_luma_drift"])) for row in rows])
        ),
        "maximum_new_boundary_fraction": float(
            np.max([float(row["candidate"]["new_boundary_fraction"]) for row in rows])
        ),
        "median_candidate_edge_correlation": float(
            np.median([row["candidate"]["added_structure_edge_correlation"] for row in rows])
        ),
        "worst_candidate_to_film_highpass_energy_ratio": float(
            np.max([row["candidate"]["candidate_to_film_highpass_energy_ratio"] for row in rows])
        ),
        "minimum_safe_scale": float(np.min([row["candidate"]["safe_scale"] for row in rows])),
    }
    gates = config["evaluation"]
    checks = {
        "required_scenes": len(rows) == int(gates["required_scenes"]),
        "psd_gain_over_basic": metrics["median_psd_error_improvement_over_basic"]
        >= float(gates["minimum_median_psd_error_improvement_over_basic"]),
        "scene_wins_over_basic": metrics["scene_win_fraction_over_basic"]
        >= float(gates["minimum_scene_win_fraction_over_basic"]),
        "shape_gain_over_white": metrics["median_psd_error_improvement_over_white"]
        >= float(gates["minimum_median_psd_error_improvement_over_white"]),
        "shape_gain_over_wrong": metrics["median_psd_error_improvement_over_wrong_shape"]
        >= float(gates["minimum_median_psd_error_improvement_over_wrong_shape"]),
        "energy_gain": metrics["median_highpass_energy_error_improvement_over_basic"]
        >= float(gates["minimum_median_highpass_energy_error_improvement_over_basic"]),
        "mean_preserved": metrics["maximum_absolute_mean_luma_drift"]
        <= float(gates["maximum_absolute_mean_luma_drift"]),
        "no_new_boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "not_edge_locked": metrics["median_candidate_edge_correlation"]
        <= float(gates["maximum_median_candidate_edge_correlation"]),
        "energy_tail_bounded": metrics["worst_candidate_to_film_highpass_energy_ratio"]
        <= float(gates["maximum_worst_candidate_to_film_highpass_energy_ratio"]),
    }
    passed = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bp3-flickr-bw-density-structure-challenger-report.v1",
        "node": config["node"],
        "software_commit": config["software_commit"],
        "parent_residual_sha256": config["parents"]["residual_report"]["sha256"],
        "parent_registration_sha256": config["parents"]["registration_report"]["sha256"],
        "parent_manifest_sha256": config["parents"]["download_manifest"]["sha256"],
        "information_flow": {
            "held_film_used_for_profile": False,
            "held_film_used_for_candidate": False,
            "held_film_used_for_evaluation": True,
            "target_informed_basic_tone_baseline": True,
        },
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "branch": config["branches"]["pass" if passed else "fail"],
        "rows": rows,
        "training_allowed": False,
        "product_integration_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}
    return report, previews


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_internal(root, config, return_previews=False)[0]


def render_contact_sheet(root: Path, config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    report, previews = _evaluate_internal(root, config, return_previews=True)
    if not report["automatic_pass"]:
        raise FlickrBwDensityStructureError("visual review is gated on automatic pass")
    cell_w, cell_h = 260, 210
    canvas = Image.new("RGB", (cell_w * 4, cell_h * len(previews)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(previews):
        y = row_index * cell_h
        images = (
            ("basic", row["basic"]),
            ("candidate", row["candidate"]),
            ("film", row["film"]),
            ("|candidate-basic| x8", np.minimum(row["difference"] * 8.0, 1.0)),
        )
        for column, (label, values) in enumerate(images):
            encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
            preview = ImageOps.contain(Image.fromarray(encoded, mode="L").convert("RGB"), (cell_w - 8, cell_h - 28))
            canvas.paste(preview, (column * cell_w + (cell_w - preview.width) // 2, y + 4))
            draw.text((column * cell_w + 5, y + cell_h - 20), f"{row['scene_id']:02d} {label}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94, subsampling=0)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "scenes": len(previews)}


__all__ = [
    "FlickrBwDensityStructureError",
    "SCHEMA",
    "evaluate",
    "render_contact_sheet",
    "safe_density_candidate",
    "synthesize_spectral_field",
]
