"""Frozen U6.P4AI nonstationary covariance-LOD audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from src.eval.physical_derivative_conditioned_structure import profile_from_contract
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    gaussian_block_mean_variance_scale,
    render_derivative_conditioned_structure,
    render_derivative_conditioned_structure_region,
)


SCHEMA = "neuro_film.u6_p4ai_covariance_lod_nonstationary_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ai_covariance_lod_nonstationary_report.v1"


class CovarianceLodNonstationaryError(RuntimeError):
    """Raised when P4AI contracts or parents drift."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CovarianceLodNonstationaryError("P4AI parent path must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluation, gates = value.get("evaluation", {}), value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or evaluation.get("shape") != [512, 768]
        or evaluation.get("lod_factors") != [2, 4, 8]
        or evaluation.get("row_partitions") != [17, 61]
        or evaluation.get("statistics_grid") != [4, 6]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_cell_mean_density_absolute_error") != 0.02
        or gates.get("minimum_cell_variance_ratio") != 0.5
        or gates.get("maximum_cell_variance_ratio") != 1.8
        or gates.get("maximum_median_absolute_log_variance_error") != 0.3
        or gates.get("minimum_median_variance_error_improvement_over_direct") != 0.3
        or gates.get("maximum_isolated_excursion_count") != 0
        or not gates.get("no_parameter_fit")
    ):
        raise CovarianceLodNonstationaryError("P4AI frozen contract drift")
    return value


def _parents(config: dict[str, Any], root: Path) -> tuple[dict, dict]:
    loaded = {}
    for key in ("p4ah_decision", "p4af_contract", "sensitometry_contract"):
        path = _relative(root, config["parents"][f"{key}_path"])
        if not path.is_file() or _hash(path) != config["parents"][f"{key}_sha256"]:
            raise CovarianceLodNonstationaryError(f"P4AI parent mismatch: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded["p4ah_decision"].get("passed"):
        raise CovarianceLodNonstationaryError("P4AH did not authorize P4AI")
    return loaded["p4af_contract"], loaded["sensitometry_contract"]


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    height, width, channels = values.shape
    return values.reshape(height // factor, factor, width // factor, factor, channels).mean(axis=(1, 3), dtype=np.float64)


def _profile(parent: DerivativeConditionedStructureProfile, factor: int, scale: float) -> DerivativeConditionedStructureProfile:
    return DerivativeConditionedStructureProfile(
        peak_density_variance=parent.peak_density_variance * scale,
        correlation_sigma_pixels=parent.correlation_sigma_pixels / factor,
        layer_seeds=parent.layer_seeds,
        exposure_minimum=parent.exposure_minimum,
        exposure_maximum=parent.exposure_maximum,
        normalization_samples=parent.normalization_samples,
    )


def _patterns(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = (int(item) for item in spec["shape"])
    fields = {}
    ramp = np.geomspace(*map(float, spec["patterns"]["ramp"][:2]), width)
    fields["ramp"] = np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy()
    lo, mid, hi = map(float, spec["patterns"]["step"])
    step = np.empty((height, width, 3), dtype=np.float64)
    step[:, : width // 3] = lo
    step[:, width // 3 : 2 * width // 3] = mid
    step[:, 2 * width // 3 :] = hi
    fields["step"] = step
    lo, hi, period = spec["patterns"]["checkerboard"]
    yy, xx = np.indices((height, width))
    mask = ((yy // int(period) + xx // int(period)) % 2).astype(bool)
    fields["checkerboard"] = np.repeat(np.where(mask, float(hi), float(lo))[..., None], 3, axis=2)
    lo, hi, spacing = spec["patterns"]["sparse"]
    sparse = np.full((height, width, 3), float(lo), dtype=np.float64)
    sparse[int(spacing) // 2 :: int(spacing), int(spacing) // 2 :: int(spacing)] = float(hi)
    fields["sparse"] = sparse
    return fields


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1])


def _isolated(z: np.ndarray, threshold: float, support: float, minimum: int) -> int:
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    count = 0
    for channel in range(3):
        magnitude = np.abs(z[..., channel])
        neighbors = convolve((magnitude >= support).astype(np.int16), kernel, mode="constant")
        count += int(np.count_nonzero((magnitude >= threshold) & (neighbors < minimum)))
    return count


def evaluate_covariance_lod_nonstationary(config: dict[str, Any], root: Path) -> dict[str, Any]:
    p4af_contract, sensitometry = _parents(config, root)
    operator = build_operator(sensitometry)
    parent = profile_from_contract(p4af_contract)
    evaluation, gates = config["evaluation"], config["gates"]
    grid_y, grid_x = (int(item) for item in evaluation["statistics_grid"])
    rows = []
    density_errors, transmittance_errors, ratios = [], [], []
    candidate_log_errors, direct_improvements, boundary_errors, acf_errors = [], [], [], []
    isolated_count = 0
    domain = repeat_exact = partition_exact = input_unchanged = True

    for pattern_name, exposure in _patterns(evaluation).items():
        preserved = exposure.copy()
        full = render_derivative_conditioned_structure(exposure, operator, parent)
        for factor_value in evaluation["lod_factors"]:
            factor = int(factor_value)
            low_exposure = _block_mean(exposure, factor)
            reference_density = _block_mean(full.density.astype(np.float64), factor)
            reference_transmittance = _block_mean(full.transmittance.astype(np.float64), factor)
            scale = gaussian_block_mean_variance_scale(parent.correlation_sigma_pixels, factor)
            candidate_profile = _profile(parent, factor, scale)
            candidate = render_derivative_conditioned_structure(low_exposure, operator, candidate_profile)
            repeat = render_derivative_conditioned_structure(low_exposure, operator, candidate_profile)
            repeat_exact &= np.array_equal(candidate.density, repeat.density)
            repeat_exact &= np.array_equal(candidate.transmittance, repeat.transmittance)
            direct = render_derivative_conditioned_structure(
                low_exposure, operator, _profile(parent, factor, 1.0 / float(factor * factor))
            )
            assembled_density = np.empty_like(candidate.density)
            assembled_transmittance = np.empty_like(candidate.transmittance)
            for partition in evaluation["row_partitions"]:
                for y0 in range(0, low_exposure.shape[0], int(partition)):
                    y1 = min(low_exposure.shape[0], y0 + int(partition))
                    region = render_derivative_conditioned_structure_region(
                        low_exposure,
                        operator,
                        candidate_profile,
                        origin_yx=(y0, 0),
                        shape=(y1 - y0, low_exposure.shape[1]),
                    )
                    assembled_density[y0:y1] = region.density
                    assembled_transmittance[y0:y1] = region.transmittance
                partition_exact &= np.array_equal(assembled_density, candidate.density)
                partition_exact &= np.array_equal(assembled_transmittance, candidate.transmittance)
            target, normalized = derivative_variance_shape(low_exposure, operator, parent)
            expected_variance = parent.peak_density_variance * scale * normalized
            z = np.zeros_like(target)
            active = expected_variance > 0.0
            z[active] = (candidate.density.astype(np.float64)[active] - target[active]) / np.sqrt(expected_variance[active])
            row_isolated = _isolated(
                z,
                float(gates["isolated_excursion_z_threshold"]),
                float(gates["isolated_neighbor_support_z_threshold"]),
                int(gates["minimum_isolated_neighbor_support"]),
            )
            isolated_count += row_isolated
            domain &= bool(
                np.min(candidate.density) >= 0.0
                and np.min(candidate.transmittance) > 0.0
                and np.max(candidate.transmittance) <= 1.0
            )
            input_unchanged &= np.array_equal(exposure, preserved)
            y_edges = np.linspace(0, low_exposure.shape[0], grid_y + 1, dtype=int)
            x_edges = np.linspace(0, low_exposure.shape[1], grid_x + 1, dtype=int)
            cell_rows = []
            reference_cell_means, candidate_cell_means = [], []
            for y_index in range(grid_y):
                for x_index in range(grid_x):
                    sl = np.s_[y_edges[y_index] : y_edges[y_index + 1], x_edges[x_index] : x_edges[x_index + 1], :]
                    if reference_density[sl].shape[0] * reference_density[sl].shape[1] < int(evaluation["minimum_cell_pixels"]):
                        raise RuntimeError("P4AI statistics cell is too small")
                    channel_rows = []
                    for channel in range(3):
                        layer_slice = np.s_[y_edges[y_index] : y_edges[y_index + 1], x_edges[x_index] : x_edges[x_index + 1], channel]
                        reference = reference_density[layer_slice]
                        candidate_values = candidate.density[layer_slice].astype(np.float64)
                        direct_values = direct.density[layer_slice].astype(np.float64)
                        reference_variance = float(np.var(reference))
                        candidate_ratio = float(np.var(candidate_values)) / reference_variance
                        direct_ratio = float(np.var(direct_values)) / reference_variance
                        candidate_error = abs(math.log(candidate_ratio))
                        direct_error = abs(math.log(direct_ratio))
                        improvement = 1.0 - candidate_error / direct_error if direct_error > 0.0 else 0.0
                        density_error = abs(float(np.mean(candidate_values) - np.mean(reference)))
                        trans_error = abs(float(np.mean(candidate.transmittance[layer_slice]) - np.mean(reference_transmittance[layer_slice])))
                        density_errors.append(density_error)
                        transmittance_errors.append(trans_error)
                        ratios.append(candidate_ratio)
                        candidate_log_errors.append(candidate_error)
                        direct_improvements.append(improvement)
                        reference_cell_means.append(float(np.mean(reference)))
                        candidate_cell_means.append(float(np.mean(candidate_values)))
                        channel_rows.append({
                            "channel": channel,
                            "density_mean_absolute_error": density_error,
                            "transmittance_mean_absolute_error": trans_error,
                            "variance_ratio": candidate_ratio,
                            "absolute_log_variance_error": candidate_error,
                            "variance_error_improvement_over_direct": improvement,
                        })
                    cell_rows.append({"y": y_index, "x": x_index, "channels": channel_rows})
            reference_means = np.asarray(reference_cell_means).reshape(grid_y * grid_x, 3)
            candidate_means = np.asarray(candidate_cell_means).reshape(grid_y * grid_x, 3)
            boundary_error = float(np.max(np.abs(np.ptp(candidate_means, axis=0) - np.ptp(reference_means, axis=0))))
            boundary_errors.append(boundary_error)
            reference_residual = reference_density - _block_mean(operator.apply(exposure), factor)
            candidate_residual = candidate.density.astype(np.float64) - operator.apply(low_exposure)
            acf_error = max(abs(_lag1(candidate_residual[..., channel]) - _lag1(reference_residual[..., channel])) for channel in range(3))
            acf_errors.append(acf_error)
            rows.append({
                "pattern": pattern_name,
                "factor": factor,
                "analytic_variance_scale": scale,
                "isolated_excursion_count": row_isolated,
                "boundary_contrast_absolute_error": boundary_error,
                "maximum_lag1_acf_absolute_error": acf_error,
                "cells": cell_rows,
            })

    metrics = {
        "rows": rows,
        "maximum_cell_mean_density_absolute_error": max(density_errors),
        "maximum_cell_mean_transmittance_absolute_error": max(transmittance_errors),
        "minimum_cell_variance_ratio": min(ratios),
        "maximum_cell_variance_ratio": max(ratios),
        "median_absolute_log_variance_error": float(np.median(candidate_log_errors)),
        "median_variance_error_improvement_over_direct": float(np.median(direct_improvements)),
        "maximum_boundary_contrast_absolute_error": max(boundary_errors),
        "maximum_lag1_acf_absolute_error": max(acf_errors),
        "total_isolated_excursion_count": isolated_count,
        "density_transmittance_domain": domain,
        "input_unchanged": input_unchanged,
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
    }
    checks = {
        "density_mean": metrics["maximum_cell_mean_density_absolute_error"] <= gates["maximum_cell_mean_density_absolute_error"],
        "transmittance_mean": metrics["maximum_cell_mean_transmittance_absolute_error"] <= gates["maximum_cell_mean_transmittance_absolute_error"],
        "variance_range": metrics["minimum_cell_variance_ratio"] >= gates["minimum_cell_variance_ratio"] and metrics["maximum_cell_variance_ratio"] <= gates["maximum_cell_variance_ratio"],
        "variance_median": metrics["median_absolute_log_variance_error"] <= gates["maximum_median_absolute_log_variance_error"],
        "beats_direct": metrics["median_variance_error_improvement_over_direct"] >= gates["minimum_median_variance_error_improvement_over_direct"],
        "boundary_contrast": metrics["maximum_boundary_contrast_absolute_error"] <= gates["maximum_boundary_contrast_absolute_error"],
        "acf": metrics["maximum_lag1_acf_absolute_error"] <= gates["maximum_lag1_acf_absolute_error"],
        "isolated": isolated_count <= gates["maximum_isolated_excursion_count"],
        "physical_domain": domain,
        "input_unchanged": input_unchanged,
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
        "no_parameter_fit": True,
    }
    passed = all(checks.values())
    stable = {"experiment_id": config["experiment_id"], "metrics": metrics, "checks": checks, "passed": passed}
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["CovarianceLodNonstationaryError", "REPORT_SCHEMA", "SCHEMA", "evaluate_covariance_lod_nonstationary", "load_contract"]
