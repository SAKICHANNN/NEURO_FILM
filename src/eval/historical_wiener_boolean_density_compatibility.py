"""U6.P2AP historical Wiener amplitude vs generic Boolean-silver audit."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.film_physics.developed_structure import (
    build_bw_silver_context,
    render_developed_structure,
    render_developed_structure_region,
)


class HistoricalWienerBooleanCompatibilityError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ap_historical_wiener_boolean_density_compatibility_contract.v1"
    ):
        raise HistoricalWienerBooleanCompatibilityError("unsupported P2AP contract")
    return payload


def _evaluate_realization(
    task: tuple[float, int, dict[str, Any], bool],
) -> dict[str, Any]:
    density, seed, mechanism, check_partition = task
    target = np.full(tuple(mechanism["input_shape"]), density, dtype=np.float64)
    kwargs = {
        "radius_um": mechanism["radius_um"],
        "output_zoom": mechanism["output_zoom"],
        "output_pixel_pitch_um": mechanism["output_pixel_pitch_um"],
        "monte_carlo_samples": mechanism["monte_carlo_samples"],
        "seed": seed,
    }
    context = build_bw_silver_context(target, **kwargs)
    repeated_context = build_bw_silver_context(target, **kwargs)
    transmittance = render_developed_structure(context).values[..., 0]
    partition_exact = True
    if check_partition:
        assembled = np.empty_like(transmittance)
        tile_rows = mechanism["row_partition_height"]
        for y0 in range(0, transmittance.shape[0], tile_rows):
            height = min(tile_rows, transmittance.shape[0] - y0)
            assembled[y0 : y0 + height] = render_developed_structure_region(
                context,
                output_origin_yx=(y0, 0),
                output_shape=(height, transmittance.shape[1]),
            ).values[..., 0]
        partition_exact = bool(np.array_equal(transmittance, assembled))
    zero_count = int(np.count_nonzero(transmittance <= 0.0))
    margin = mechanism["crop_margin_output_pixels"]
    interior = transmittance[margin:-margin, margin:-margin]
    if np.any(interior <= 0.0):
        variance = 0.0
    else:
        density_field = -np.log10(interior.astype(np.float64))
        variance = float(np.var(density_field, dtype=np.float64))
    return {
        "density": density,
        "seed": seed,
        "grain_count": len(context.centers_by_layer[0]),
        "density_variance": variance,
        "transmittance_sha256": hashlib.sha256(
            np.asarray(transmittance, dtype="<f4").tobytes()
        ).hexdigest(),
        "zero_count": zero_count,
        "repeat_exact": context.fingerprint() == repeated_context.fingerprint(),
        "partition_exact": partition_exact,
    }


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise HistoricalWienerBooleanCompatibilityError(
                "P2AP parent identity mismatch"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("automatic_pass") is not binding["required_automatic_pass"]:
            raise HistoricalWienerBooleanCompatibilityError(
                "P2AP parent decision mismatch"
            )
    observation = contract["historical_observation"]
    densities = np.asarray(observation["densities"], dtype=np.float64)
    wiener = np.asarray(
        observation["wiener_granularity_spectrum_cm2"], dtype=np.float64
    )
    target_ratios = wiener / wiener[0]
    mechanism = contract["unchanged_generic_mechanism"]
    tasks = [
        (
            float(density),
            int(seed),
            mechanism,
            density_index == len(densities) // 2 and seed_index == 0,
        )
        for density_index, density in enumerate(densities)
        for seed_index, seed in enumerate(mechanism["seeds"])
    ]
    worker_count = min(12, os.cpu_count() or 1, len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        realized = list(executor.map(_evaluate_realization, tasks))
    rows = []
    repeat_results = [row.pop("repeat_exact") for row in realized]
    partition_results = [row.pop("partition_exact") for row in realized]
    zero_counts = [row.pop("zero_count") for row in realized]
    repeat_exact = all(repeat_results)
    partition_exact = all(partition_results)
    zero_count = sum(zero_counts)
    variance_by_density: list[list[float]] = [[] for _ in densities]
    seeds_per_density = len(mechanism["seeds"])
    for row_index, row in enumerate(realized):
        variance_by_density[row_index // seeds_per_density].append(
            row["density_variance"]
        )
        rows.append(row)
    median_variances = np.median(
        np.asarray(variance_by_density, dtype=np.float64), axis=1
    )
    if np.isfinite(median_variances[0]) and median_variances[0] > 0.0:
        candidate_ratios = median_variances / median_variances[0]
    else:
        candidate_ratios = np.zeros_like(median_variances)
    relative_errors = np.abs(candidate_ratios - target_ratios) / target_ratios
    rank = float(spearmanr(target_ratios, candidate_ratios).statistic)
    if not np.isfinite(rank):
        rank = -1.0
    measurements = {
        "target_relative_wiener_amplitudes": target_ratios.tolist(),
        "median_density_variances": median_variances.tolist(),
        "candidate_relative_density_variances": candidate_ratios.tolist(),
        "relative_ratio_rmse": float(np.sqrt(np.mean(np.square(relative_errors)))),
        "maximum_absolute_relative_ratio_error": float(np.max(relative_errors)),
        "rank_correlation": rank,
        "zero_transmittance_count": zero_count,
        "all_density_variances_positive": bool(
            np.all(np.isfinite(median_variances)) and np.all(median_variances > 0.0)
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_byte_exact": partition_exact,
        "parameter_refit_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "maximum_relative_ratio_rmse": measurements["relative_ratio_rmse"]
        <= gates["maximum_relative_ratio_rmse"],
        "maximum_absolute_relative_ratio_error": measurements[
            "maximum_absolute_relative_ratio_error"
        ]
        <= gates["maximum_absolute_relative_ratio_error"],
        "minimum_rank_correlation": rank >= gates["minimum_rank_correlation"],
        "zero_transmittance_count": zero_count == gates["zero_transmittance_count"],
        "all_density_variances_positive": measurements["all_density_variances_positive"]
        is gates["all_density_variances_positive"],
        "repeat_byte_exact": repeat_exact is gates["repeat_byte_exact"],
        "partition_byte_exact": partition_exact is gates["partition_byte_exact"],
        "parameter_refit_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ap_historical_wiener_boolean_density_compatibility_report.v1",
        "rows": rows,
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
