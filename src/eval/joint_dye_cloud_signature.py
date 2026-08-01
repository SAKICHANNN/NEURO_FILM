"""Joint signal-transfer and noise-spectrum dye-cloud audit.

This module deliberately remains an offline scientific evaluator.  It tests
whether one positive Gaussian cloud scale can explain both signal MTF and the
NPS shape of an independently realized marked-Poisson density field.  It does
not render display RGB or define a stock profile.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != (
        "neuro_film.u6_p4as_joint_dye_cloud_mtf_nps_contract.v1"
    ):
        raise ValueError("unexpected U6.P4AS contract schema")
    return contract


def _gaussian_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not math.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive")
    if sigma == 0.0:
        return np.ones(1, dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma))
    return kernel / np.sum(kernel, dtype=np.float64)


def gaussian_mtf(
    frequencies_cycles_per_pixel: np.ndarray,
    sigma_pixels: float,
    truncate: float,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_cycles_per_pixel, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or not len(frequencies)
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0.0)
        or np.any(frequencies >= 0.5)
    ):
        raise ValueError("frequencies must be finite one-dimensional Nyquist values")
    kernel = _gaussian_kernel_1d(float(sigma_pixels), float(truncate))
    radius = len(kernel) // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    response = np.sum(
        kernel[None, :]
        * np.cos(2.0 * np.pi * frequencies[:, None] * coordinates[None, :]),
        axis=1,
        dtype=np.float64,
    )
    return response


def _sigma_grid(spec: dict[str, float]) -> np.ndarray:
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["step"])
    if (
        not all(math.isfinite(value) for value in (minimum, maximum, step))
        or minimum <= 0.0
        or maximum < minimum
        or step <= 0.0
    ):
        raise ValueError("invalid sigma grid")
    count = round((maximum - minimum) / step)
    grid = minimum + step * np.arange(count + 1, dtype=np.float64)
    if abs(float(grid[-1]) - maximum) > 1e-12:
        raise ValueError("sigma grid endpoints are not exactly representable by step")
    return grid


def fit_sigma_from_mtf(
    measured_mtf: np.ndarray,
    frequencies_cycles_per_pixel: np.ndarray,
    *,
    sigma_grid: np.ndarray,
    truncate: float,
) -> tuple[float, float]:
    measured = np.asarray(measured_mtf, dtype=np.float64)
    frequencies = np.asarray(frequencies_cycles_per_pixel, dtype=np.float64)
    if measured.shape != frequencies.shape or not np.all(np.isfinite(measured)):
        raise ValueError("measured MTF must match finite frequencies")
    losses = np.asarray(
        [
            np.mean(
                np.square(gaussian_mtf(frequencies, sigma, truncate) - measured),
                dtype=np.float64,
            )
            for sigma in sigma_grid
        ],
        dtype=np.float64,
    )
    selected = int(np.argmin(losses))
    return float(sigma_grid[selected]), float(math.sqrt(losses[selected]))


def _radial_layout(
    shape: tuple[int, int], edges: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    radius = np.sqrt(np.square(fx) + np.square(fy))
    masks = [
        (radius >= edges[index]) & (radius < edges[index + 1])
        for index in range(len(edges) - 1)
    ]
    if any(not np.any(mask) for mask in masks):
        raise ValueError("NPS radial band has no samples")
    return radius, masks


def predicted_nps_shape(
    shape: tuple[int, int],
    edges: np.ndarray,
    sigma_pixels: float,
    truncate: float,
) -> np.ndarray:
    _, masks = _radial_layout(shape, edges)
    kernel = _gaussian_kernel_1d(float(sigma_pixels), float(truncate))
    radius = len(kernel) // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    frequency_y = np.fft.fftfreq(shape[0])
    frequency_x = np.fft.fftfreq(shape[1])
    transfer_y = np.sum(
        kernel[None, :]
        * np.cos(2.0 * np.pi * frequency_y[:, None] * coordinates[None, :]),
        axis=1,
        dtype=np.float64,
    )
    transfer_x = np.sum(
        kernel[None, :]
        * np.cos(2.0 * np.pi * frequency_x[:, None] * coordinates[None, :]),
        axis=1,
        dtype=np.float64,
    )
    power = np.square(transfer_y[:, None] * transfer_x[None, :])
    bands = np.asarray([np.mean(power[mask]) for mask in masks], dtype=np.float64)
    return bands / np.sum(bands, dtype=np.float64)


def realized_poisson_nps_shape(
    *,
    shape: tuple[int, int],
    edges: np.ndarray,
    sigma_pixels: float,
    truncate: float,
    event_rate: float,
    seeds: list[int],
) -> tuple[np.ndarray, bool]:
    if not math.isfinite(event_rate) or event_rate <= 0.0:
        raise ValueError("event rate must be finite and positive")
    _, masks = _radial_layout(shape, edges)
    band_powers = np.zeros(len(masks), dtype=np.float64)
    finite_nonnegative = True
    for seed in seeds:
        counts = (
            np.random.default_rng(seed)
            .poisson(event_rate, size=shape)
            .astype(np.float64)
        )
        density = gaussian_filter(
            counts,
            sigma=float(sigma_pixels),
            mode="wrap",
            truncate=float(truncate),
        )
        finite_nonnegative = bool(
            finite_nonnegative
            and np.all(np.isfinite(density))
            and np.all(density >= 0.0)
        )
        centered = density - np.mean(density, dtype=np.float64)
        power = np.square(np.abs(np.fft.fft2(centered))) / float(density.size)
        band_powers += np.asarray(
            [np.mean(power[mask]) for mask in masks], dtype=np.float64
        )
    band_powers /= float(len(seeds))
    return band_powers / np.sum(band_powers, dtype=np.float64), finite_nonnegative


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        math.sqrt(
            np.mean(np.square(np.asarray(left) - np.asarray(right)), dtype=np.float64)
        )
    )


def _stable_id(core: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def evaluate_joint_dye_cloud_signature(contract: dict[str, Any]) -> dict[str, Any]:
    reference = contract["synthetic_reference"]
    split = contract["split"]
    candidate = contract["candidate"]
    truth_sigmas = [float(value) for value in reference["truth_sigma_pixels"]]
    layer_ids = [str(value) for value in reference["layer_ids"]]
    if len(truth_sigmas) != 3 or len(layer_ids) != 3:
        raise ValueError("U6.P4AS requires exactly three colour layers")
    truncate = float(reference["gaussian_truncate"])
    development_frequencies = np.asarray(
        split["development_mtf_frequencies_cycles_per_pixel"], dtype=np.float64
    )
    confirmation_frequencies = np.asarray(
        split["confirmation_mtf_frequencies_cycles_per_pixel"], dtype=np.float64
    )
    edges = np.asarray(
        split["confirmation_nps_radial_edges_cycles_per_pixel"], dtype=np.float64
    )
    shape = tuple(int(value) for value in reference["shape"])
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("reference shape must contain two positive dimensions")
    if np.any(np.diff(edges) <= 0.0) or edges[0] <= 0.0 or edges[-1] >= 0.5:
        raise ValueError("NPS edges must be strictly increasing inside Nyquist")
    grid = _sigma_grid(candidate["sigma_grid_pixels"])

    # Bundle freeze: this phase consumes development MTF only.
    fitted_sigmas: list[float] = []
    development_fit_rmse: list[float] = []
    for truth_sigma in truth_sigmas:
        measured = gaussian_mtf(development_frequencies, truth_sigma, truncate)
        fitted, fit_rmse = fit_sigma_from_mtf(
            measured,
            development_frequencies,
            sigma_grid=grid,
            truncate=truncate,
        )
        fitted_sigmas.append(fitted)
        development_fit_rmse.append(fit_rmse)
    bundle_core = {
        "family": candidate["family"],
        "layer_ids": layer_ids,
        "fitted_sigma_pixels": fitted_sigmas,
        "development_frequencies_cycles_per_pixel": development_frequencies.tolist(),
        "truncate": truncate,
    }
    bundle_id = _stable_id(bundle_core)

    controls = contract["controls"]
    wrong_mapping = [int(value) for value in controls["wrong_layer_mapping"]]
    if sorted(wrong_mapping) != [0, 1, 2]:
        raise ValueError("wrong-layer mapping must be a permutation")
    identity_sigma = float(controls["no_diffusion_sigma_pixels"])
    confirmation_seeds = [int(value) for value in reference["confirmation_seeds"]]
    event_rates = [float(value) for value in reference["poisson_event_rates"]]
    grid_nps_shapes = np.stack(
        [predicted_nps_shape(shape, edges, sigma, truncate) for sigma in grid]
    )

    layer_metrics: list[dict[str, Any]] = []
    all_disagreements: list[float] = []
    all_candidate_nps_rmse: list[float] = []
    all_identity_improvements: list[float] = []
    all_wrong_improvements: list[float] = []
    all_density_valid = True
    for layer_index, (layer_id, truth_sigma, fitted_sigma) in enumerate(
        zip(layer_ids, truth_sigmas, fitted_sigmas, strict=True)
    ):
        truth_mtf = gaussian_mtf(confirmation_frequencies, truth_sigma, truncate)
        candidate_mtf = gaussian_mtf(confirmation_frequencies, fitted_sigma, truncate)
        mtf_rmse = _rmse(candidate_mtf, truth_mtf)
        candidate_shape = predicted_nps_shape(shape, edges, fitted_sigma, truncate)
        identity_shape = predicted_nps_shape(shape, edges, identity_sigma, truncate)
        wrong_sigma = fitted_sigmas[wrong_mapping[layer_index]]
        wrong_shape = predicted_nps_shape(shape, edges, wrong_sigma, truncate)
        density_metrics: list[dict[str, Any]] = []
        for rate_index, event_rate in enumerate(event_rates):
            seeds = [
                seed + layer_index * 1000 + rate_index * 100
                for seed in confirmation_seeds
            ]
            observed_shape, density_valid = realized_poisson_nps_shape(
                shape=shape,
                edges=edges,
                sigma_pixels=truth_sigma,
                truncate=truncate,
                event_rate=event_rate,
                seeds=seeds,
            )
            all_density_valid = bool(all_density_valid and density_valid)
            candidate_error = _rmse(candidate_shape, observed_shape)
            identity_error = _rmse(identity_shape, observed_shape)
            wrong_error = _rmse(wrong_shape, observed_shape)
            oracle_losses = np.sqrt(
                np.mean(
                    np.square(grid_nps_shapes - observed_shape[None, :]),
                    axis=1,
                    dtype=np.float64,
                )
            )
            nps_sigma = float(grid[int(np.argmin(oracle_losses))])
            disagreement = abs(nps_sigma - fitted_sigma) / truth_sigma
            improvement_identity = 1.0 - candidate_error / identity_error
            improvement_wrong = 1.0 - candidate_error / wrong_error
            all_disagreements.append(disagreement)
            all_candidate_nps_rmse.append(candidate_error)
            all_identity_improvements.append(improvement_identity)
            all_wrong_improvements.append(improvement_wrong)
            density_metrics.append(
                {
                    "event_rate": event_rate,
                    "observed_nps_shape": observed_shape.tolist(),
                    "candidate_nps_rmse": candidate_error,
                    "identity_nps_rmse": identity_error,
                    "wrong_layer_nps_rmse": wrong_error,
                    "candidate_improvement_vs_identity": improvement_identity,
                    "candidate_improvement_vs_wrong_layer": improvement_wrong,
                    "nps_only_oracle_sigma_pixels": nps_sigma,
                    "nps_vs_mtf_sigma_relative_disagreement": disagreement,
                    "finite_nonnegative_density": density_valid,
                }
            )
        layer_metrics.append(
            {
                "layer_id": layer_id,
                "truth_sigma_pixels": truth_sigma,
                "fitted_sigma_pixels": fitted_sigma,
                "sigma_relative_error": abs(fitted_sigma - truth_sigma) / truth_sigma,
                "development_mtf_fit_rmse": development_fit_rmse[layer_index],
                "confirmation_mtf_rmse": mtf_rmse,
                "confirmation_truth_mtf": truth_mtf.tolist(),
                "confirmation_candidate_mtf": candidate_mtf.tolist(),
                "candidate_nps_shape": candidate_shape.tolist(),
                "wrong_layer_sigma_pixels": wrong_sigma,
                "densities": density_metrics,
            }
        )

    gates = contract["automatic_gates"]
    sigma_error_max = max(item["sigma_relative_error"] for item in layer_metrics)
    mtf_rmse_max = max(item["confirmation_mtf_rmse"] for item in layer_metrics)
    nps_rmse_max = max(all_candidate_nps_rmse)
    identity_improvement_median = float(np.median(all_identity_improvements))
    wrong_improvement_median = float(np.median(all_wrong_improvements))
    disagreement_median = float(np.median(all_disagreements))
    disagreement_p95 = float(np.quantile(all_disagreements, 0.95))
    decisions = {
        "sigma_recovery": sigma_error_max
        <= float(gates["maximum_sigma_relative_error"]),
        "held_mtf": mtf_rmse_max <= float(gates["maximum_confirmation_mtf_rmse"]),
        "held_nps": nps_rmse_max <= float(gates["maximum_confirmation_nps_shape_rmse"]),
        "nps_vs_no_diffusion": identity_improvement_median
        >= float(gates["minimum_median_nps_improvement_vs_no_diffusion"]),
        "nps_vs_wrong_layer": wrong_improvement_median
        >= float(gates["minimum_median_nps_improvement_vs_wrong_layer"]),
        "nps_sigma_median": disagreement_median
        <= float(gates["maximum_median_nps_sigma_relative_disagreement"]),
        "nps_sigma_tail": disagreement_p95
        <= float(gates["maximum_p95_nps_sigma_relative_disagreement"]),
        "all_layers_and_densities": len(all_candidate_nps_rmse)
        == len(layer_ids) * len(event_rates),
        "finite_nonnegative_density": all_density_valid,
        "confirmation_after_bundle_freeze": bool(bundle_id),
    }
    automatic_pass = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p4as_joint_dye_cloud_mtf_nps_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "bundle": {**bundle_core, "bundle_id": bundle_id},
        "layer_metrics": layer_metrics,
        "aggregate": {
            "maximum_sigma_relative_error": sigma_error_max,
            "maximum_confirmation_mtf_rmse": mtf_rmse_max,
            "maximum_confirmation_nps_shape_rmse": nps_rmse_max,
            "median_nps_improvement_vs_no_diffusion": identity_improvement_median,
            "median_nps_improvement_vs_wrong_layer": wrong_improvement_median,
            "median_nps_sigma_relative_disagreement": disagreement_median,
            "p95_nps_sigma_relative_disagreement": disagreement_p95,
        },
        "decisions": decisions,
        "automatic_pass": automatic_pass,
        "branch": contract["branch_rule"]["pass" if automatic_pass else "fail"],
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
