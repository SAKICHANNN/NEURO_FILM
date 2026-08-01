"""BP4 leave-one-scene-out deterministic density-domain B&W acutance audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.flickr_bw_density_structure_challenger import (
    _load_json,
    _patches,
    _prepare_scene,
    _profile_error,
    safe_density_candidate,
)
from src.eval.flickr_bw_residual_identifiability import _highpass, _load_luma, _radial_psd
from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file


SCHEMA = "neuro-film.u5-r2bp4-flickr-bw-density-acutance-challenger.v1"


class FlickrBwDensityAcutanceError(ValueError):
    """Raised when BP4 inputs, LOO selection, or numerical contracts drift."""


def _bounded_linear_candidate(
    basic: np.ndarray,
    valid: np.ndarray,
    residual: np.ndarray,
    low: float,
    high: float,
    maximum_scale: float,
    maximum_delta: float,
) -> tuple[np.ndarray, float, float]:
    residual = np.asarray(residual, dtype=np.float64).copy()
    residual[~valid] = 0.0
    residual[valid] -= float(np.mean(residual[valid]))
    positive = valid & (residual > 0.0)
    negative = valid & (residual < 0.0)
    limits = [float(maximum_scale)]
    if np.any(positive):
        limits.append(float(np.min((high - basic[positive]) / residual[positive])))
    if np.any(negative):
        limits.append(float(np.min((basic[negative] - low) / (-residual[negative]))))
    maximum_raw_delta = float(np.max(np.abs(residual[valid])))
    if maximum_raw_delta > 0.0:
        limits.append(float(maximum_delta) / maximum_raw_delta)
    scale = max(0.0, min(limits))
    if scale > 0.0:
        scale *= 1.0 - 1e-12
    candidate = basic + scale * residual
    if np.any(candidate[valid] < low) or np.any(candidate[valid] > high):
        raise FlickrBwDensityAcutanceError("bounded linear candidate escaped")
    return candidate, scale, float(np.mean(candidate[valid]) - np.mean(basic[valid]))


def render_acutance_candidate(
    basic: np.ndarray,
    valid: np.ndarray,
    inner_sigma: float,
    outer_sigma: float,
    strength: float,
    low: float,
    high: float,
    maximum_scale: float,
    maximum_delta: float,
    density_domain: bool,
) -> tuple[np.ndarray, float, float]:
    """Render one bounded mean-preserving DoG acutance candidate."""

    if not (0.0 < inner_sigma < outer_sigma and strength > 0.0):
        raise FlickrBwDensityAcutanceError("invalid acutance parameters")
    values = np.log(np.maximum(basic, 1.0 / 255.0)) if density_domain else basic
    inner = cv2.GaussianBlur(values, (0, 0), inner_sigma, borderType=cv2.BORDER_REFLECT_101)
    outer = cv2.GaussianBlur(values, (0, 0), outer_sigma, borderType=cv2.BORDER_REFLECT_101)
    dog = strength * (inner - outer)
    if density_domain:
        candidate, scale, drift = safe_density_candidate(
            basic, valid, dog, low, high, maximum_scale
        )
        peak = float(np.max(np.abs(candidate[valid] - basic[valid])))
        if peak > maximum_delta:
            factor = maximum_delta / peak
            candidate = basic + factor * (candidate - basic)
            scale *= factor
            drift = float(np.mean(candidate[valid]) - np.mean(basic[valid]))
        return candidate, scale, drift
    return _bounded_linear_candidate(
        basic, valid, dog, low, high, maximum_scale, maximum_delta
    )


def _analyze(
    scene: Mapping[str, Any],
    candidate: np.ndarray,
    scale: float,
    mean_drift: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    floor = float(contract["log_floor"])
    candidate_hp = _highpass(
        np.log(np.maximum(candidate, floor)),
        float(contract["prefilter_sigma_pixels"]),
        float(contract["highpass_sigma_pixels"]),
    )
    candidate_patches = _patches(candidate_hp, scene["slices"])
    film_patches = _patches(scene["film_hp"], scene["slices"])
    edges = np.asarray(contract["radial_frequency_edges_nyquist"], dtype=np.float64)
    candidate_psd = _radial_psd(candidate_patches, edges)
    film_psd = _radial_psd(film_patches, edges)
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    basic_interior = (scene["basic"] > low) & (scene["basic"] < high)
    new_boundary = basic_interior & ((candidate <= low) | (candidate >= high))
    return {
        "safe_scale": scale,
        "mean_luma_drift": mean_drift,
        "new_boundary_fraction": float(np.mean(new_boundary)),
        "maximum_absolute_luma_delta": float(np.max(np.abs(candidate - scene["basic"]))),
        "radial_psd": candidate_psd.tolist(),
        "psd_error": _profile_error(candidate_psd, film_psd),
        "highpass_energy": float(np.mean(candidate_patches**2)),
        "candidate_to_film_highpass_energy_ratio": float(
            np.mean(candidate_patches**2) / max(float(np.mean(film_patches**2)), 1e-20)
        ),
    }


def _render_and_analyze(
    scene: Mapping[str, Any],
    params: tuple[float, float, float],
    contract: Mapping[str, Any],
    density_domain: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    inner, outer, strength = params
    low = float(contract["encoded_boundary_code_minimum"]) / 255.0
    high = float(contract["encoded_boundary_code_maximum"]) / 255.0
    image, scale, drift = render_acutance_candidate(
        scene["basic"],
        scene["valid"],
        inner,
        outer,
        strength,
        low,
        high,
        float(contract["maximum_safe_scale"]),
        float(contract["maximum_absolute_luma_delta"]),
        density_domain,
    )
    return _analyze(scene, image, scale, drift, contract), image


def _evaluate_internal(
    root: Path, config: Mapping[str, Any], return_previews: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrBwDensityAcutanceError("invalid BP4 contract")
    if re.fullmatch(r"[0-9a-f]{40}", str(config.get("software_commit", ""))) is None:
        raise FlickrBwDensityAcutanceError("software commit is not frozen")
    bp2 = _load_json(root, config["parents"]["residual_report"])
    bp3 = _load_json(root, config["parents"]["stochastic_close_report"])
    registration = _load_json(root, config["parents"]["registration_report"])
    manifest = _load_json(root, config["parents"]["download_manifest"])
    if (
        not bp2.get("automatic_pass")
        or bp2.get("stable_evidence_id")
        != config["parents"]["residual_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwDensityAcutanceError("BP2 parent did not pass exactly")
    if (
        bp3.get("automatic_pass")
        or bp3.get("stable_evidence_id")
        != config["parents"]["stochastic_close_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwDensityAcutanceError("BP3 close parent drift")
    if (
        not registration.get("automatic_pass")
        or registration.get("stable_evidence_id")
        != config["parents"]["registration_report"]["required_stable_evidence_id"]
    ):
        raise FlickrBwDensityAcutanceError("BP1 parent did not pass exactly")
    by_scene = {int(row["scene_id"]): row for row in manifest["rows"]}
    bp2_rows = {int(row["scene_id"]): row for row in bp2["rows"]}
    data_root = root / str(config["data_root"])
    scenes: dict[int, dict[str, Any]] = {}
    for pair in registration["pairs"]:
        if not pair["diagnostics"]["registration_gate_passed"]:
            continue
        scene_id = int(pair["scene_id"])
        source = by_scene[scene_id]
        scenes[scene_id] = _prepare_scene(
            _load_luma(data_root / source["digital_local_path"], source["digital_sha256"]),
            _load_luma(data_root / source["film_local_path"], source["film_sha256"]),
            np.asarray(pair["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["analysis"],
        )
    grid = [
        (float(inner), float(outer), float(strength))
        for inner, outer, strength in itertools.product(
            config["analysis"]["inner_sigma_grid_pixels"],
            config["analysis"]["outer_sigma_grid_pixels"],
            config["analysis"]["strength_grid"],
        )
        if float(inner) < float(outer)
    ]
    if len(grid) != 27:
        raise FlickrBwDensityAcutanceError("parameter grid drift")
    cache: dict[tuple[int, tuple[float, float, float]], tuple[dict[str, Any], np.ndarray]] = {}
    for scene_id, scene in scenes.items():
        for params in grid:
            cache[(scene_id, params)] = _render_and_analyze(
                scene, params, config["analysis"], density_domain=True
            )
    rows: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    for held_id in sorted(scenes):
        training_ids = [scene_id for scene_id in sorted(scenes) if scene_id != held_id]
        scored = [
            (
                float(np.median([cache[(scene_id, params)][0]["psd_error"] for scene_id in training_ids])),
                params,
            )
            for params in grid
        ]
        selected = min(scored, key=lambda item: (item[0], item[1]))[1]
        candidate, candidate_image = cache[(held_id, selected)]
        linear, _ = _render_and_analyze(
            scenes[held_id], selected, config["analysis"], density_domain=False
        )
        wrong_params = (selected[0] * 2.0, selected[1] * 2.0, selected[2])
        wrong, _ = _render_and_analyze(
            scenes[held_id], wrong_params, config["analysis"], density_domain=True
        )
        basic_psd = np.asarray(bp2_rows[held_id]["residual"]["basic_radial_psd"], dtype=np.float64)
        film_psd = np.asarray(bp2_rows[held_id]["residual"]["film_radial_psd"], dtype=np.float64)
        basic_error = _profile_error(basic_psd, film_psd)
        film_energy = float(bp2_rows[held_id]["residual"]["film_highpass_energy"])
        basic_energy = float(bp2_rows[held_id]["residual"]["basic_highpass_energy"])
        candidate_error = float(candidate["psd_error"])
        linear_error = float(linear["psd_error"])
        wrong_error = float(wrong["psd_error"])
        row = {
            "scene_id": held_id,
            "photo_id": bp2_rows[held_id]["photo_id"],
            "training_scene_ids": training_ids,
            "held_scene_used_for_selection": False,
            "selected_parameters": {
                "inner_sigma_pixels": selected[0],
                "outer_sigma_pixels": selected[1],
                "strength": selected[2],
            },
            "training_median_psd_error": min(scored)[0],
            "basic_psd_error": basic_error,
            "candidate_psd_error": candidate_error,
            "linear_luma_psd_error": linear_error,
            "wrong_radius_psd_error": wrong_error,
            "candidate_improvement_over_basic": (basic_error - candidate_error) / max(basic_error, 1e-20),
            "candidate_improvement_over_linear_luma": (linear_error - candidate_error) / max(linear_error, 1e-20),
            "candidate_improvement_over_wrong_radius": (wrong_error - candidate_error) / max(wrong_error, 1e-20),
            "basic_highpass_energy_error": abs(basic_energy - film_energy) / max(film_energy, 1e-20),
            "candidate_highpass_energy_error": abs(float(candidate["highpass_energy"]) - film_energy)
            / max(film_energy, 1e-20),
            "candidate": candidate,
            "linear_luma_control": linear,
            "wrong_radius_control": wrong,
        }
        rows.append(row)
        if return_previews:
            previews.append(
                {
                    "scene_id": held_id,
                    "basic": scenes[held_id]["basic"],
                    "candidate": candidate_image,
                    "film": scenes[held_id]["film"],
                    "difference": np.abs(candidate_image - scenes[held_id]["basic"]),
                }
            )
    improvements = np.asarray([row["candidate_improvement_over_basic"] for row in rows])
    linear_improvements = np.asarray([row["candidate_improvement_over_linear_luma"] for row in rows])
    wrong_improvements = np.asarray([row["candidate_improvement_over_wrong_radius"] for row in rows])
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
        "median_psd_error_improvement_over_linear_luma": float(np.median(linear_improvements)),
        "median_psd_error_improvement_over_wrong_radius": float(np.median(wrong_improvements)),
        "median_highpass_energy_error_improvement_over_basic": float(np.median(energy_improvements)),
        "minimum_safe_scale": float(np.min([row["candidate"]["safe_scale"] for row in rows])),
        "maximum_absolute_mean_luma_drift": float(
            np.max([abs(float(row["candidate"]["mean_luma_drift"])) for row in rows])
        ),
        "maximum_new_boundary_fraction": float(
            np.max([float(row["candidate"]["new_boundary_fraction"]) for row in rows])
        ),
        "maximum_absolute_luma_delta": float(
            np.max([float(row["candidate"]["maximum_absolute_luma_delta"]) for row in rows])
        ),
        "worst_candidate_to_film_highpass_energy_ratio": float(
            np.max([row["candidate"]["candidate_to_film_highpass_energy_ratio"] for row in rows])
        ),
        "selected_parameter_counts": {
            f"{row['selected_parameters']['inner_sigma_pixels']}/"
            f"{row['selected_parameters']['outer_sigma_pixels']}/"
            f"{row['selected_parameters']['strength']}": sum(
                other["selected_parameters"] == row["selected_parameters"] for other in rows
            )
            for row in rows
        },
    }
    gates = config["evaluation"]
    checks = {
        "required_scenes": len(rows) == int(gates["required_scenes"]),
        "psd_gain_over_basic": metrics["median_psd_error_improvement_over_basic"]
        >= float(gates["minimum_median_psd_error_improvement_over_basic"]),
        "scene_wins_over_basic": metrics["scene_win_fraction_over_basic"]
        >= float(gates["minimum_scene_win_fraction_over_basic"]),
        "density_gain_over_linear": metrics["median_psd_error_improvement_over_linear_luma"]
        >= float(gates["minimum_median_psd_error_improvement_over_linear_luma"]),
        "radius_specificity": metrics["median_psd_error_improvement_over_wrong_radius"]
        >= float(gates["minimum_median_psd_error_improvement_over_wrong_radius"]),
        "energy_gain": metrics["median_highpass_energy_error_improvement_over_basic"]
        >= float(gates["minimum_median_highpass_energy_error_improvement_over_basic"]),
        "nontrivial_safe_scale": metrics["minimum_safe_scale"] >= float(gates["minimum_safe_scale"]),
        "mean_preserved": metrics["maximum_absolute_mean_luma_drift"]
        <= float(gates["maximum_absolute_mean_luma_drift"]),
        "no_new_boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "bounded_luma_delta": metrics["maximum_absolute_luma_delta"]
        <= float(gates["maximum_absolute_luma_delta"]) + 1e-12,
        "energy_tail_bounded": metrics["worst_candidate_to_film_highpass_energy_ratio"]
        <= float(gates["maximum_worst_candidate_to_film_highpass_energy_ratio"]),
    }
    passed = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bp4-flickr-bw-density-acutance-challenger-report.v1",
        "node": config["node"],
        "software_commit": config["software_commit"],
        "parent_residual_sha256": config["parents"]["residual_report"]["sha256"],
        "parent_stochastic_close_sha256": config["parents"]["stochastic_close_report"]["sha256"],
        "information_flow": {
            "held_film_used_for_parameter_selection": False,
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
        raise FlickrBwDensityAcutanceError("visual review is gated on automatic pass")
    cell_w, cell_h = 260, 210
    canvas = Image.new("RGB", (cell_w * 4, cell_h * len(previews)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(previews):
        y = row_index * cell_h
        images = (
            ("basic", row["basic"]),
            ("density acutance", row["candidate"]),
            ("film", row["film"]),
            ("|candidate-basic| x8", np.minimum(row["difference"] * 8.0, 1.0)),
        )
        for column, (label, values) in enumerate(images):
            encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
            preview = ImageOps.contain(
                Image.fromarray(encoded, mode="L").convert("RGB"), (cell_w - 8, cell_h - 28)
            )
            canvas.paste(preview, (column * cell_w + (cell_w - preview.width) // 2, y + 4))
            draw.text((column * cell_w + 5, y + cell_h - 20), f"{row['scene_id']:02d} {label}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94, subsampling=0)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "scenes": len(previews)}


__all__ = [
    "FlickrBwDensityAcutanceError",
    "SCHEMA",
    "evaluate",
    "render_acutance_candidate",
    "render_contact_sheet",
]
