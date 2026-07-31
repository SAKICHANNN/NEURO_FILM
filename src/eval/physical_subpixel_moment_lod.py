"""Frozen U6.P4AJ subpixel developed-density moment LOD audit."""

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
)
from src.film_physics.subpixel_moment_structure import (
    compile_subpixel_density_moments,
    render_density_moment_structure,
    render_density_moment_structure_region,
)


SCHEMA = "neuro_film.u6_p4aj_subpixel_moment_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4aj_subpixel_moment_lod_report.v1"


class SubpixelMomentLodError(RuntimeError):
    """Raised when P4AJ frozen evidence drifts."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SubpixelMomentLodError("P4AJ parent path must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model, evaluation, gates = value.get("model", {}), value.get("evaluation", {}), value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or model.get("maximum_factor") != 8
        or model.get("full_resolution_material_field_required")
        or model.get("per_image_fit_allowed")
        or model.get("post_result_retuning_allowed")
        or evaluation.get("shape") != [512, 768]
        or evaluation.get("lod_factors") != [2, 4, 8]
        or evaluation.get("row_partitions") != [17, 61]
        or evaluation.get("statistics_grid") != [4, 6]
        or evaluation.get("runs") != 2
        or gates.get("maximum_cell_mean_density_absolute_error") != 0.01
        or gates.get("minimum_cell_variance_ratio") != 0.7
        or gates.get("maximum_cell_variance_ratio") != 1.3
        or gates.get("maximum_median_absolute_log_variance_error") != 0.15
        or gates.get("minimum_median_variance_error_improvement_over_p4ai") != 0.5
        or not gates.get("no_parameter_fit")
    ):
        raise SubpixelMomentLodError("P4AJ frozen contract drift")
    return value


def _parents(config: dict[str, Any], root: Path) -> tuple[dict, dict]:
    loaded = {}
    for key in ("p4ai_decision", "p4af_contract", "sensitometry_contract"):
        path = _relative(root, config["parents"][f"{key}_path"])
        if not path.is_file() or _hash(path) != config["parents"][f"{key}_sha256"]:
            raise SubpixelMomentLodError(f"P4AJ parent mismatch: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    if loaded["p4ai_decision"].get("decision") != "close_scalar_exposure_covariance_lod_nonstationary_reuse":
        raise SubpixelMomentLodError("P4AI decision does not authorize P4AJ")
    return loaded["p4af_contract"], loaded["sensitometry_contract"]


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    h, w, c = values.shape
    return values.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3), dtype=np.float64)


def _scalar_profile(parent: DerivativeConditionedStructureProfile, factor: int) -> DerivativeConditionedStructureProfile:
    return DerivativeConditionedStructureProfile(
        peak_density_variance=parent.peak_density_variance * gaussian_block_mean_variance_scale(parent.correlation_sigma_pixels, factor),
        correlation_sigma_pixels=parent.correlation_sigma_pixels / factor,
        layer_seeds=parent.layer_seeds,
        exposure_minimum=parent.exposure_minimum,
        exposure_maximum=parent.exposure_maximum,
        normalization_samples=parent.normalization_samples,
    )


def _patterns(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    h, w = map(int, spec["shape"])
    ramp = np.geomspace(*map(float, spec["patterns"]["ramp"][:2]), w)
    result = {"ramp": np.broadcast_to(ramp[None, :, None], (h, w, 3)).copy()}
    lo, mid, hi = map(float, spec["patterns"]["step"])
    step = np.empty((h, w, 3)); step[:, :w // 3] = lo; step[:, w // 3:2 * w // 3] = mid; step[:, 2 * w // 3:] = hi
    result["step"] = step
    lo, hi, period = spec["patterns"]["checkerboard"]
    yy, xx = np.indices((h, w)); mask = ((yy // int(period) + xx // int(period)) % 2).astype(bool)
    result["checkerboard"] = np.repeat(np.where(mask, float(hi), float(lo))[..., None], 3, axis=2)
    lo, hi, spacing = spec["patterns"]["sparse"]
    sparse = np.full((h, w, 3), float(lo)); sparse[int(spacing)//2::int(spacing), int(spacing)//2::int(spacing)] = float(hi)
    result["sparse"] = sparse
    return result


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1])


def _isolated(z: np.ndarray, threshold: float, support: float, minimum: int) -> int:
    kernel = np.ones((3, 3), dtype=np.int16); kernel[1, 1] = 0
    return sum(int(np.count_nonzero((np.abs(z[..., c]) >= threshold) & (convolve((np.abs(z[..., c]) >= support).astype(np.int16), kernel, mode="constant") < minimum))) for c in range(3))


def evaluate_subpixel_moment_lod(config: dict[str, Any], root: Path) -> dict[str, Any]:
    p4af_contract, sensitometry = _parents(config, root)
    operator, parent = build_operator(sensitometry), profile_from_contract(p4af_contract)
    spec, gates = config["evaluation"], config["gates"]
    gy, gx = map(int, spec["statistics_grid"])
    rows, density_errors, trans_errors, ratios, log_errors, improvements, boundary_errors, acf_errors = [], [], [], [], [], [], [], []
    isolated_count = 0
    domain = input_unchanged = repeat_exact = partition_exact = True
    for pattern, exposure in _patterns(spec).items():
        preserved = exposure.copy()
        target, normalized = derivative_variance_shape(exposure, operator, parent)
        full = render_derivative_conditioned_structure(exposure, operator, parent)
        for factor_value in spec["lod_factors"]:
            factor = int(factor_value)
            moments = compile_subpixel_density_moments(
                target,
                parent.peak_density_variance * normalized,
                pixel_size_factor=factor,
                correlation_sigma_pixels=parent.correlation_sigma_pixels,
            )
            sigma = parent.correlation_sigma_pixels / factor
            candidate = render_density_moment_structure(moments, correlation_sigma_pixels=sigma, layer_seeds=parent.layer_seeds)
            repeat = render_density_moment_structure(moments, correlation_sigma_pixels=sigma, layer_seeds=parent.layer_seeds)
            repeat_exact &= np.array_equal(candidate.density, repeat.density) and np.array_equal(candidate.transmittance, repeat.transmittance)
            low_exposure = _block_mean(exposure, factor)
            p4ai = render_derivative_conditioned_structure(low_exposure, operator, _scalar_profile(parent, factor))
            reference_density = _block_mean(full.density.astype(np.float64), factor)
            reference_transmittance = _block_mean(full.transmittance.astype(np.float64), factor)
            assembled_density = np.empty_like(candidate.density); assembled_transmittance = np.empty_like(candidate.transmittance)
            for partition in spec["row_partitions"]:
                for y0 in range(0, moments.mean_density.shape[0], int(partition)):
                    y1 = min(moments.mean_density.shape[0], y0 + int(partition))
                    region = render_density_moment_structure_region(moments, correlation_sigma_pixels=sigma, layer_seeds=parent.layer_seeds, origin_yx=(y0, 0), shape=(y1-y0, moments.mean_density.shape[1]))
                    assembled_density[y0:y1] = region.density; assembled_transmittance[y0:y1] = region.transmittance
                partition_exact &= np.array_equal(assembled_density, candidate.density) and np.array_equal(assembled_transmittance, candidate.transmittance)
            z = np.zeros_like(moments.mean_density)
            active = moments.variance_density > 0.0
            z[active] = (candidate.density.astype(np.float64)[active] - moments.mean_density[active]) / np.sqrt(moments.variance_density[active])
            row_isolated = _isolated(z, float(gates["isolated_excursion_z_threshold"]), float(gates["isolated_neighbor_support_z_threshold"]), int(gates["minimum_isolated_neighbor_support"]))
            isolated_count += row_isolated
            domain &= bool(np.min(candidate.density) >= 0 and np.min(candidate.transmittance) > 0 and np.max(candidate.transmittance) <= 1)
            input_unchanged &= np.array_equal(exposure, preserved)
            y_edges = np.linspace(0, candidate.density.shape[0], gy+1, dtype=int); x_edges = np.linspace(0, candidate.density.shape[1], gx+1, dtype=int)
            reference_means, candidate_means, cells = [], [], []
            for yi in range(gy):
                for xi in range(gx):
                    cell_channels = []
                    for channel in range(3):
                        sl = np.s_[y_edges[yi]:y_edges[yi+1], x_edges[xi]:x_edges[xi+1], channel]
                        reference = reference_density[sl]; values = candidate.density[sl].astype(np.float64); control = p4ai.density[sl].astype(np.float64)
                        reference_variance = float(np.var(reference)); ratio = float(np.var(values)) / reference_variance; control_ratio = float(np.var(control)) / reference_variance
                        error, control_error = abs(math.log(ratio)), abs(math.log(control_ratio))
                        improvement = 1.0 - error / control_error if control_error > 0 else 0.0
                        density_error = abs(float(np.mean(values) - np.mean(reference)))
                        trans_error = abs(float(np.mean(candidate.transmittance[sl]) - np.mean(reference_transmittance[sl])))
                        density_errors.append(density_error); trans_errors.append(trans_error); ratios.append(ratio); log_errors.append(error); improvements.append(improvement)
                        reference_means.append(float(np.mean(reference))); candidate_means.append(float(np.mean(values)))
                        cell_channels.append({"channel": channel, "density_mean_absolute_error": density_error, "transmittance_mean_absolute_error": trans_error, "variance_ratio": ratio, "absolute_log_variance_error": error, "variance_error_improvement_over_p4ai": improvement})
                    cells.append({"y": yi, "x": xi, "channels": cell_channels})
            reference_means_array = np.asarray(reference_means).reshape(gy*gx, 3); candidate_means_array = np.asarray(candidate_means).reshape(gy*gx, 3)
            boundary_error = float(np.max(np.abs(np.ptp(candidate_means_array, axis=0) - np.ptp(reference_means_array, axis=0)))); boundary_errors.append(boundary_error)
            reference_residual = reference_density - _block_mean(target, factor); candidate_residual = candidate.density.astype(np.float64) - moments.mean_density
            acf_error = max(abs(_lag1(candidate_residual[..., c]) - _lag1(reference_residual[..., c])) for c in range(3)); acf_errors.append(acf_error)
            rows.append({"pattern": pattern, "factor": factor, "isolated_excursion_count": row_isolated, "boundary_contrast_absolute_error": boundary_error, "maximum_lag1_acf_absolute_error": acf_error, "cells": cells})
    metrics = {
        "rows": rows,
        "maximum_cell_mean_density_absolute_error": max(density_errors),
        "maximum_cell_mean_transmittance_absolute_error": max(trans_errors),
        "minimum_cell_variance_ratio": min(ratios),
        "maximum_cell_variance_ratio": max(ratios),
        "median_absolute_log_variance_error": float(np.median(log_errors)),
        "median_variance_error_improvement_over_p4ai": float(np.median(improvements)),
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
        "beats_p4ai": metrics["median_variance_error_improvement_over_p4ai"] >= gates["minimum_median_variance_error_improvement_over_p4ai"],
        "boundary": metrics["maximum_boundary_contrast_absolute_error"] <= gates["maximum_boundary_contrast_absolute_error"],
        "acf": metrics["maximum_lag1_acf_absolute_error"] <= gates["maximum_lag1_acf_absolute_error"],
        "isolated": isolated_count <= gates["maximum_isolated_excursion_count"],
        "physical_domain": domain, "input_unchanged": input_unchanged, "repeat_exact": repeat_exact, "row_partition_exact": partition_exact, "no_parameter_fit": True,
    }
    passed = all(checks.values())
    stable = {"experiment_id": config["experiment_id"], "metrics": metrics, "checks": checks, "passed": passed}
    return {"schema": REPORT_SCHEMA, **stable, "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(), "decision": config["branch_rule"]["pass" if passed else "fail"], "claim_ceiling": config["claim_ceiling"]}


__all__ = ["SubpixelMomentLodError", "REPORT_SCHEMA", "SCHEMA", "evaluate_subpixel_moment_lod", "load_contract"]
