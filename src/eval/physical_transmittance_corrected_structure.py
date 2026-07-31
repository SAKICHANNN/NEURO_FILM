"""Frozen P4AM synthetic audit of analytic mean-transmittance correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.physical_callier_source import hash_file
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    gamma_mean_transmittance_density_offset,
    render_derivative_conditioned_structure,
    render_transmittance_corrected_structure,
    render_transmittance_corrected_structure_region,
)


SCHEMA = "neuro_film.u6_p4am_transmittance_moment_corrected_structure_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4am_transmittance_moment_corrected_structure_report.v1"


class TransmittanceCorrectedStructureError(RuntimeError):
    """Raised when the P4AM contract or parent evidence drifts."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model, evaluation, gates = value.get("model", {}), value.get("evaluation", {}), value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or model.get("peak_density_variance") != 0.0001
        or model.get("correlation_sigma_pixels") != 0.65
        or model.get("layer_seeds") != [260831, 260837, 260851]
        or model.get("fitted_parameters") != 0
        or model.get("display_rgb_noise_allowed")
        or model.get("hard_clipping_allowed")
        or evaluation.get("shape") != [512, 768]
        or evaluation.get("row_partitions") != [31, 127]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_analytic_transmittance_mean_absolute_error") != 1e-12
        or gates.get("minimum_transmittance_bias_improvement_over_p4af") != 0.5
        or not gates.get("no_parameter_fit")
    ):
        raise TransmittanceCorrectedStructureError("P4AM frozen contract drift")
    return value


def _load_parents(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    identities = {}
    for stem in ("p4af_decision", "p4u_decision", "sensitometry_contract"):
        relative = Path(config["parents"][f"{stem}_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise TransmittanceCorrectedStructureError("P4AM parent path escaped root")
        path = root / relative
        actual = hash_file(path)
        if actual != config["parents"][f"{stem}_sha256"]:
            raise TransmittanceCorrectedStructureError(f"P4AM parent hash mismatch: {stem}")
        identities[stem] = actual
    sensitometry = json.loads((root / config["parents"]["sensitometry_contract_path"]).read_text(encoding="utf-8"))
    return sensitometry, identities


def _profile(config: dict[str, Any]) -> DerivativeConditionedStructureProfile:
    model = config["model"]
    return DerivativeConditionedStructureProfile(
        peak_density_variance=float(model["peak_density_variance"]),
        correlation_sigma_pixels=float(model["correlation_sigma_pixels"]),
        layer_seeds=tuple(model["layer_seeds"]),
        exposure_minimum=float(model["variance_normalization_domain"][0]),
        exposure_maximum=float(model["variance_normalization_domain"][1]),
        normalization_samples=int(model["variance_normalization_samples"]),
    )


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1])


def evaluate_structure(config: dict[str, Any], root: Path) -> dict[str, Any]:
    sensitometry, identities = _load_parents(config, root)
    operator, profile = build_operator(sensitometry), _profile(config)
    evaluation, gates = config["evaluation"], config["gates"]
    height, width = evaluation["shape"]
    border = int(evaluation["interior_border_pixels"])
    analytic_errors, flat_errors, flat_baseline_errors, variance_errors, lag1_values = [], [], [], [], []
    input_unchanged = True
    for level in evaluation["flat_exposure_levels"]:
        source = np.full((height, width, 3), level, dtype=np.float64)
        preserved = source.copy()
        mean, shape = derivative_variance_shape(source, operator, profile)
        variance = profile.peak_density_variance * shape
        offset = gamma_mean_transmittance_density_offset(mean, variance)
        active = variance > 0.0
        target_transmittance = np.power(10.0, -mean)
        analytic = target_transmittance.copy()
        if np.any(active):
            gamma_shape = np.square(mean[active]) / variance[active]
            gamma_scale = variance[active] / mean[active]
            log_ten = np.log(10.0)
            analytic[active] = np.exp(
                -log_ten * offset[active]
                - gamma_shape * np.log1p(log_ten * gamma_scale)
            )
        analytic_errors.append(float(np.max(np.abs(analytic - target_transmittance))))
        candidate = render_transmittance_corrected_structure(source, operator, profile)
        baseline = render_derivative_conditioned_structure(source, operator, profile)
        sl = np.s_[border:-border, border:-border, :]
        flat_errors.append(float(np.max(np.abs(np.mean(candidate.transmittance[sl], axis=(0, 1)) - target_transmittance[0, 0]))))
        flat_baseline_errors.append(float(np.max(np.abs(np.mean(baseline.transmittance[sl], axis=(0, 1)) - target_transmittance[0, 0]))))
        residual = candidate.density[sl].astype(np.float64) - (mean + offset)[sl]
        if np.any(active[0, 0]):
            observed = np.mean(np.square(residual), axis=(0, 1))
            variance_errors.extend((np.abs(observed[active[0, 0]] - variance[0, 0, active[0, 0]]) / variance[0, 0, active[0, 0]]).tolist())
            lag1_values.extend(_lag1(residual[..., channel]) for channel in range(3) if active[0, 0, channel])
        input_unchanged &= np.array_equal(source, preserved)

    ramp_values = np.geomspace(*evaluation["ramp_exposure_interval"], width, dtype=np.float64)
    ramp = np.broadcast_to(ramp_values[None, :, None], (height, width, 3)).copy()
    mean, shape = derivative_variance_shape(ramp, operator, profile)
    variance = profile.peak_density_variance * shape
    offset = gamma_mean_transmittance_density_offset(mean, variance)
    target_transmittance = np.power(10.0, -mean)
    candidate = render_transmittance_corrected_structure(ramp, operator, profile)
    baseline = render_derivative_conditioned_structure(ramp, operator, profile)
    repeated = render_transmittance_corrected_structure(ramp, operator, profile)
    repeat_exact = np.array_equal(candidate.density, repeated.density) and np.array_equal(candidate.transmittance, repeated.transmittance)
    partition_exact = True
    for rows in evaluation["row_partitions"]:
        assembled_density = np.empty_like(candidate.density)
        assembled_transmittance = np.empty_like(candidate.transmittance)
        for y0 in range(0, height, rows):
            region = render_transmittance_corrected_structure_region(ramp, operator, profile, origin_yx=(y0, 0), shape=(min(rows, height-y0), width))
            assembled_density[y0:y0+region.density.shape[0]] = region.density
            assembled_transmittance[y0:y0+region.transmittance.shape[0]] = region.transmittance
        partition_exact &= np.array_equal(candidate.density, assembled_density) and np.array_equal(candidate.transmittance, assembled_transmittance)
    candidate_errors, baseline_errors, observed_variance, expected_variance = [], [], [], []
    edges = np.linspace(0, width, evaluation["ramp_bins"] + 1, dtype=int)
    density_residual = candidate.density.astype(np.float64) - (mean + offset)
    for channel in range(3):
        for x0, x1 in zip(edges[:-1], edges[1:]):
            region = np.s_[border:-border, x0:x1, channel]
            target_mean = float(np.mean(target_transmittance[region]))
            candidate_errors.append(abs(float(np.mean(candidate.transmittance[region])) - target_mean))
            baseline_errors.append(abs(float(np.mean(baseline.transmittance[region])) - target_mean))
            observed_variance.append(float(np.mean(np.square(density_residual[region]))))
            expected_variance.append(float(np.mean(variance[region])))
    candidate_p90 = float(np.quantile(candidate_errors, 0.9))
    baseline_p90 = float(np.quantile(baseline_errors, 0.9))
    improvement = 1.0 - candidate_p90 / baseline_p90
    metrics = {
        "maximum_analytic_transmittance_mean_absolute_error": max(analytic_errors),
        "maximum_empirical_flat_transmittance_mean_absolute_error": max(flat_errors),
        "maximum_empirical_flat_p4af_error": max(flat_baseline_errors),
        "ramp_transmittance_mean_absolute_error_p90": candidate_p90,
        "ramp_p4af_transmittance_mean_absolute_error_p90": baseline_p90,
        "transmittance_bias_improvement_over_p4af": improvement,
        "maximum_density_variance_relative_error": max(variance_errors),
        "ramp_variance_shape_spearman": float(spearmanr(expected_variance, observed_variance).statistic),
        "minimum_lag1_autocorrelation": min(lag1_values),
        "density_minimum": float(np.min(candidate.density)),
        "transmittance_minimum": float(np.min(candidate.transmittance)),
        "transmittance_maximum": float(np.max(candidate.transmittance)),
        "repeat_exact": bool(repeat_exact),
        "row_partition_exact": bool(partition_exact),
        "input_unchanged": bool(input_unchanged),
    }
    checks = {
        "analytic_transmittance_mean": metrics["maximum_analytic_transmittance_mean_absolute_error"] <= gates["maximum_analytic_transmittance_mean_absolute_error"],
        "empirical_flat_transmittance_mean": metrics["maximum_empirical_flat_transmittance_mean_absolute_error"] <= gates["maximum_empirical_flat_transmittance_mean_absolute_error"],
        "ramp_transmittance_mean": candidate_p90 <= gates["maximum_ramp_transmittance_mean_absolute_error_p90"],
        "bias_improvement": improvement >= gates["minimum_transmittance_bias_improvement_over_p4af"],
        "density_variance": metrics["maximum_density_variance_relative_error"] <= gates["maximum_density_variance_relative_error"],
        "variance_shape": metrics["ramp_variance_shape_spearman"] >= gates["minimum_ramp_variance_shape_spearman"],
        "lag1": metrics["minimum_lag1_autocorrelation"] >= gates["minimum_lag1_autocorrelation"],
        "physical_domain": metrics["density_minimum"] >= 0.0 and metrics["transmittance_minimum"] > 0.0 and metrics["transmittance_maximum"] <= 1.0,
        "repeat_exact": bool(repeat_exact),
        "row_partition_exact": bool(partition_exact),
        "input_unchanged": bool(input_unchanged),
        "no_parameter_fit": True,
    }
    stable = {"experiment_id": config["experiment_id"], "parent_identities": identities, "metrics": metrics, "checks": checks, "passed": all(checks.values())}
    branch = "pass" if stable["passed"] else "fail"
    return {"schema": REPORT_SCHEMA, **stable, "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(), "decision": config["branch_rule"][branch], "claim_ceiling": config["claim_ceiling"]}


__all__ = ["REPORT_SCHEMA", "SCHEMA", "TransmittanceCorrectedStructureError", "evaluate_structure", "load_contract"]
