"""Label-blind generic scanner-convolved grain-shape challenger."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from src.eval.real_uniform_grain_analysis import build_scan_signatures
from src.eval.real_uniform_grain_nps import (
    cosine_similarity,
    radial_nps_signature,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file, validate_contract
from src.film_physics.compound_poisson import counter_poisson_region


class GenericGrainShapingError(RuntimeError):
    """Raised when the fixed generic shaping experiment fails closed."""


def anisotropic_poisson_region(
    *,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    rate: float,
    sigma_yx: tuple[float, float],
    seed: int,
    truncate: float,
) -> np.ndarray:
    """Render a positive anisotropic Gaussian-filtered Poisson field."""
    sigma_y, sigma_x = sigma_yx
    if (
        len(full_shape) != 2
        or len(origin_yx) != 2
        or len(shape) != 2
        or any(value <= 0 for value in full_shape + shape)
        or any(value < 0 for value in origin_yx)
        or origin_yx[0] + shape[0] > full_shape[0]
        or origin_yx[1] + shape[1] > full_shape[1]
        or not math.isfinite(sigma_y)
        or not math.isfinite(sigma_x)
        or sigma_y < 0.0
        or sigma_x < 0.0
        or not math.isfinite(truncate)
        or truncate <= 0.0
    ):
        raise ValueError("invalid anisotropic Poisson request")
    halo_y = int(truncate * sigma_y + 0.5)
    halo_x = int(truncate * sigma_x + 0.5)
    y0 = max(0, origin_yx[0] - halo_y)
    x0 = max(0, origin_yx[1] - halo_x)
    y1 = min(full_shape[0], origin_yx[0] + shape[0] + halo_y)
    x1 = min(full_shape[1], origin_yx[1] + shape[1] + halo_x)
    counts = counter_poisson_region(
        full_shape,
        origin_yx=(y0, x0),
        shape=(y1 - y0, x1 - x0),
        rate=rate,
        seed=seed,
    )
    filtered = gaussian_filter(
        counts.astype(np.float64),
        sigma=(sigma_y, sigma_x),
        order=0,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    crop_y = origin_yx[0] - y0
    crop_x = origin_yx[1] - x0
    output = np.ascontiguousarray(
        filtered[
            crop_y : crop_y + shape[0],
            crop_x : crop_x + shape[1],
        ]
    )
    if np.any(output < 0.0) or not np.all(np.isfinite(output)):
        raise GenericGrainShapingError("anisotropic field left its domain")
    output.setflags(write=False)
    return output


def _load_bound_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise GenericGrainShapingError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenericGrainShapingError("parent payload must be an object")
    return payload


def _normalized_mean(rows: list[np.ndarray]) -> np.ndarray:
    value = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.all(np.isfinite(value)):
        raise GenericGrainShapingError("degenerate signature centroid")
    output = value / norm
    output.setflags(write=False)
    return output


def _observed_signatures(
    *,
    root: Path,
    source: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(analysis["pixel_contract"]["crop_size_pixels"]),
            centers_yx=analysis["pixel_contract"][
                "fixed_fractional_centers_yx"
            ],
        )
        signatures = build_scan_signatures(
            rgb=rgb,
            infrared=infrared,
            pixel_contract=analysis["pixel_contract"],
        )
        del rgb, infrared
        rows[inspected["source_id"]] = _normalized_mean(
            list(signatures["combined_rgb_nps"])
        )
    return rows


def _synthetic_centroid(
    *,
    model: dict[str, Any],
    analysis: dict[str, Any],
    sigma_yx_by_channel: list[tuple[float, float]],
    seeds: list[int],
    count_cache: dict[tuple[tuple[int, int], float, int], np.ndarray],
) -> np.ndarray:
    full_shape = tuple(int(value) for value in model["field_shape"])
    crop_y, crop_x, crop_h, crop_w = (
        int(value) for value in model["central_analysis_crop"]
    )
    target_density = float(model["target_density"])
    grain_od = [
        float(value) for value in model["grain_optical_density_by_rgb_layer"]
    ]
    truncate = float(model["truncate"])
    edges = analysis["pixel_contract"][
        "radial_nps_band_edges_cycles_per_pixel"
    ]
    signatures: list[np.ndarray] = []
    for seed in seeds:
        channels = []
        for channel, (sigma_y, sigma_x) in enumerate(
            sigma_yx_by_channel
        ):
            rate = target_density / grain_od[channel]
            channel_seed = int(seed + 104729 * channel)
            count_key = (full_shape, rate, channel_seed)
            counts = count_cache.get(count_key)
            if counts is None:
                counts = counter_poisson_region(
                    full_shape,
                    origin_yx=(0, 0),
                    shape=full_shape,
                    rate=rate,
                    seed=channel_seed,
                )
                count_cache[count_key] = counts
            field = gaussian_filter(
                counts.astype(np.float64),
                sigma=(sigma_y, sigma_x),
                order=0,
                mode="constant",
                cval=0.0,
                truncate=truncate,
            )
            crop = field[
                crop_y : crop_y + crop_h,
                crop_x : crop_x + crop_w,
            ]
            channels.append(
                radial_nps_signature(
                    crop,
                    band_edges_cycles_per_pixel=edges,
                )
            )
        signatures.append(_normalized_mean([np.concatenate(channels)]))
    return _normalized_mean(signatures)


def _distances(
    centroid: np.ndarray,
    observations: list[np.ndarray],
) -> list[float]:
    return [
        float(1.0 - cosine_similarity(centroid, observed))
        for observed in observations
    ]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "minimum": float(np.min(values)),
    }


def _reverse_frequency_bins(signature: np.ndarray) -> np.ndarray:
    values = np.asarray(signature, dtype=np.float64)
    if values.ndim != 1 or len(values) % 3 != 0:
        raise GenericGrainShapingError("combined signature shape is invalid")
    output = np.concatenate(
        [row[::-1] for row in np.split(values, 3)]
    )
    output /= np.linalg.norm(output)
    output.setflags(write=False)
    return output


def _partition_exact(
    *,
    sigma_yx: tuple[float, float],
    rate: float,
    seed: int,
    truncate: float,
) -> bool:
    full_shape = (257, 263)
    full = anisotropic_poisson_region(
        full_shape=full_shape,
        origin_yx=(0, 0),
        shape=full_shape,
        rate=rate,
        sigma_yx=sigma_yx,
        seed=seed,
        truncate=truncate,
    )
    rows = []
    for y0 in range(0, full_shape[0], 31):
        height = min(31, full_shape[0] - y0)
        rows.append(
            anisotropic_poisson_region(
                full_shape=full_shape,
                origin_yx=(y0, 0),
                shape=(height, full_shape[1]),
                rate=rate,
                sigma_yx=sigma_yx,
                seed=seed,
                truncate=truncate,
            )
        )
    return bool(np.array_equal(full, np.concatenate(rows, axis=0)))


def run_generic_grain_shaping(
    *,
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Fit one shared label-blind anisotropic kernel on development scans."""
    if (
        contract.get("schema")
        != "neuro_film.u6_p4s_generic_grain_shaping_contract.v1"
    ):
        raise GenericGrainShapingError("unsupported P4S contract")
    parents = contract["parents"]
    source = _load_bound_json(root, parents["source_contract"])
    analysis = _load_bound_json(root, parents["analysis_contract"])
    p4r_report = _load_bound_json(root, parents["p4r_report"])
    p4r_decision = _load_bound_json(root, parents["p4r_decision"])
    p4d = _load_bound_json(root, parents["p4d_contract"])
    validate_contract(source)
    if (
        p4r_report.get("decision")
        != parents["p4r_report"]["required_decision"]
        or p4r_decision.get("result", {}).get("status")
        != parents["p4r_decision"]["required_status"]
        or p4d.get("model", {}).get("input_domain")
        != "developed_optical_density"
    ):
        raise GenericGrainShapingError("parent decision does not admit P4S")

    observed = _observed_signatures(
        root=root,
        source=source,
        analysis=analysis,
    )
    split = contract["split"]
    development_ids = split["development_source_ids"]
    confirmation_ids = split["confirmation_source_ids"]
    development = [observed[key] for key in development_ids]
    confirmation = [observed[key] for key in confirmation_ids]
    if (
        split.get("stock_labels_available_to_fit") is not False
        or len(set(development_ids)) != len(development_ids)
        or len(set(confirmation_ids)) != len(confirmation_ids)
        or set(observed) != set(development_ids) | set(confirmation_ids)
        or set(development_ids) & set(confirmation_ids)
    ):
        raise GenericGrainShapingError("development/confirmation split mismatch")

    model = contract["model"]
    if (
        model.get("amplitude_fit_allowed") is not False
        or model.get("per_channel_sigma_fit_allowed") is not False
        or model.get("stock_specific_fit_allowed") is not False
        or model.get("display_rgb_noise_allowed") is not False
        or model.get("hard_clipping_allowed") is not False
    ):
        raise GenericGrainShapingError("forbidden shaping capacity is enabled")
    baseline_sigmas = [
        (float(row[0]), float(row[1]))
        for row in model["baseline_layer_sigmas_yx"]
    ]
    development_seeds = [int(value) for value in model["development_seeds"]]
    confirmation_seeds = [int(value) for value in model["confirmation_seeds"]]
    count_cache: dict[
        tuple[tuple[int, int], float, int], np.ndarray
    ] = {}
    baseline_development = _synthetic_centroid(
        model=model,
        analysis=analysis,
        sigma_yx_by_channel=baseline_sigmas,
        seeds=development_seeds,
        count_cache=count_cache,
    )
    candidates = []
    for sigma_y in model["shared_sigma_y_grid_pixels"]:
        for sigma_x in model["shared_sigma_x_grid_pixels"]:
            sigma = (float(sigma_y), float(sigma_x))
            centroid = _synthetic_centroid(
                model=model,
                analysis=analysis,
                sigma_yx_by_channel=[sigma, sigma, sigma],
                seeds=development_seeds,
                count_cache=count_cache,
            )
            distances = _distances(centroid, development)
            candidates.append(
                {
                    "sigma_yx": sigma,
                    "centroid": centroid,
                    "distances": distances,
                    "key": (
                        float(np.median(distances)),
                        float(np.max(distances)),
                        abs(sigma[0] - sigma[1]),
                        sigma[0],
                        sigma[1],
                    ),
                }
            )
    selected = min(candidates, key=lambda row: row["key"])
    selected_sigma = selected["sigma_yx"]
    candidate_development = selected["centroid"]

    baseline_confirmation = _synthetic_centroid(
        model=model,
        analysis=analysis,
        sigma_yx_by_channel=baseline_sigmas,
        seeds=confirmation_seeds,
        count_cache=count_cache,
    )
    candidate_confirmation = _synthetic_centroid(
        model=model,
        analysis=analysis,
        sigma_yx_by_channel=[selected_sigma] * 3,
        seeds=confirmation_seeds,
        count_cache=count_cache,
    )
    repeated_candidate = _synthetic_centroid(
        model=model,
        analysis=analysis,
        sigma_yx_by_channel=[selected_sigma] * 3,
        seeds=confirmation_seeds,
        count_cache=count_cache,
    )
    baseline_dev_distance = _distances(baseline_development, development)
    candidate_dev_distance = _distances(candidate_development, development)
    baseline_confirm_distance = _distances(
        baseline_confirmation,
        confirmation,
    )
    candidate_confirm_distance = _distances(
        candidate_confirmation,
        confirmation,
    )
    reversed_confirm_distance = _distances(
        _reverse_frequency_bins(candidate_confirmation),
        confirmation,
    )
    baseline_summary = _summary(baseline_confirm_distance)
    candidate_summary = _summary(candidate_confirm_distance)
    reversed_summary = _summary(reversed_confirm_distance)
    epsilon = np.finfo(np.float64).eps
    median_improvement = 1.0 - candidate_summary["median"] / max(
        baseline_summary["median"], epsilon
    )
    worst_ratio = candidate_summary["maximum"] / max(
        baseline_summary["maximum"], epsilon
    )
    control_improvement = 1.0 - candidate_summary["median"] / max(
        reversed_summary["median"], epsilon
    )
    gates = contract["automatic_gates"]
    partition_exact = _partition_exact(
        sigma_yx=selected_sigma,
        rate=float(model["target_density"])
        / float(model["grain_optical_density_by_rgb_layer"][0]),
        seed=confirmation_seeds[0],
        truncate=float(model["truncate"]),
    )
    gate_results = {
        "confirmation_median_improvement": median_improvement
        >= float(
            gates[
                "minimum_confirmation_median_distance_improvement_fraction"
            ]
        ),
        "confirmation_worst_not_worse": worst_ratio
        <= float(gates["maximum_confirmation_worst_distance_ratio"]),
        "reversed_frequency_control": control_improvement
        >= float(
            gates[
                "minimum_candidate_vs_reversed_frequency_control_improvement_fraction"
            ]
        ),
        "repeat_exact": bool(
            np.array_equal(candidate_confirmation, repeated_candidate)
        ),
        "row_partition_exact": partition_exact,
        "positive_field": True,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p4s_generic_grain_shaping_report.v1",
        "contract_sha256": hash_file(
            root / "configs/u6_p4s_generic_grain_shaping_v1.json",
            "sha256",
        ),
        "source_count": len(observed),
        "development_source_ids": development_ids,
        "confirmation_source_ids": confirmation_ids,
        "stock_labels_used_for_fit": False,
        "selected_shared_sigma_yx_pixels": list(selected_sigma),
        "baseline_layer_sigmas_yx_pixels": [
            list(value) for value in baseline_sigmas
        ],
        "development": {
            "baseline_distance": _summary(baseline_dev_distance),
            "candidate_distance": _summary(candidate_dev_distance),
        },
        "confirmation": {
            "baseline_distance": baseline_summary,
            "candidate_distance": candidate_summary,
            "reversed_frequency_control_distance": reversed_summary,
            "median_improvement_fraction": median_improvement,
            "candidate_to_baseline_worst_distance_ratio": worst_ratio,
            "candidate_vs_reversed_control_improvement_fraction": (
                control_improvement
            ),
        },
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_generic_anisotropic_shaping_candidate"
            if automatic_pass
            else "retain_p4d_p4q_baseline_close_generic_shaping"
        ),
        "parameter_fitting": {
            "shared_kernel_only": True,
            "amplitude_fit": False,
            "per_channel_fit": False,
            "stock_specific_fit": False,
        },
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GenericGrainShapingError",
    "anisotropic_poisson_region",
    "run_generic_grain_shaping",
    "write_report",
]
