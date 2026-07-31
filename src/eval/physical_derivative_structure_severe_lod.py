"""Frozen U6.P4AG synthetic severe-artifact and direct-LOD audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve
from scipy.special import gammaincinv, ndtr
from scipy.stats import spearmanr

from src.eval.physical_density_conditioned_structure import profiles_from_contract
from src.eval.physical_derivative_conditioned_structure import profile_from_contract
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.density_conditioned_structure import (
    render_density_conditioned_structure,
)
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    render_derivative_conditioned_structure,
    render_derivative_conditioned_structure_region,
)
from src.film_physics.structure_compiler import correlated_normal_region


SCHEMA = "neuro_film.u6_p4ag_derivative_structure_severe_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ag_derivative_structure_severe_lod_report.v1"


class DerivativeStructureSevereLodError(RuntimeError):
    """Raised when the frozen P4AG evidence boundary drifts."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DerivativeStructureSevereLodError("P4AG parent path must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    populations = value.get("populations", {})
    gates = value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or populations.get("shape") != [256, 384]
        or populations.get("lod_factors") != [2, 4]
        or populations.get("row_partitions") != [29, 113]
        or populations.get("runs") != 2
        or gates.get("maximum_absolute_standardized_density_residual") != 8.0
        or gates.get("maximum_isolated_excursion_count") != 0
        or gates.get("minimum_candidate_variance_shape_spearman") != 0.9
        or gates.get("minimum_candidate_spearman_advantage_over_each_control") != 0.5
        or gates.get("maximum_lod_mean_density_absolute_error") != 0.01
        or gates.get("minimum_lod_variance_ratio") != 0.4
        or gates.get("maximum_lod_variance_ratio") != 1.6
        or gates.get("maximum_lod_lag1_acf_absolute_error") != 0.25
        or gates.get("minimum_lod_variance_error_improvement_over_unscaled") != 0.1
        or not gates.get("no_parameter_fit")
    ):
        raise DerivativeStructureSevereLodError("P4AG frozen contract drift")
    return value


def _load_parents(config: dict[str, Any], root: Path) -> tuple[dict, dict, dict, dict]:
    loaded: dict[str, dict] = {}
    for key in ("p4af_decision", "p4af_contract", "p4d_contract", "sensitometry_contract"):
        path = _relative(root, config["parents"][f"{key}_path"])
        if not path.is_file() or _hash(path) != config["parents"][f"{key}_sha256"]:
            raise DerivativeStructureSevereLodError(f"P4AG parent mismatch: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded["p4af_decision"].get("passed"):
        raise DerivativeStructureSevereLodError("P4AF did not authorize P4AG")
    return (
        loaded["p4af_decision"],
        loaded["p4af_contract"],
        loaded["p4d_contract"],
        loaded["sensitometry_contract"],
    )


def _constant_variance_density(
    target: np.ndarray,
    profile: DerivativeConditionedStructureProfile,
) -> np.ndarray:
    output = np.empty_like(target, dtype=np.float64)
    full_shape = target.shape[:2]
    for channel, seed in enumerate(profile.layer_seeds):
        normal = correlated_normal_region(
            full_shape,
            origin_yx=(0, 0),
            shape=full_shape,
            sigma=profile.correlation_sigma_pixels,
            seed=seed,
        )
        uniform = ndtr(normal)
        mean = target[..., channel]
        variance = profile.peak_density_variance
        gamma_shape = np.square(mean) / variance
        gamma_scale = variance / mean
        output[..., channel] = gammaincinv(gamma_shape, uniform) * gamma_scale
    return np.asarray(output, dtype=np.float32)


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1])


def _isolated_count(z: np.ndarray, threshold: float, support: float, minimum: int) -> int:
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    total = 0
    for channel in range(z.shape[-1]):
        magnitude = np.abs(z[..., channel])
        neighbors = convolve((magnitude >= support).astype(np.int16), kernel, mode="constant")
        total += int(np.count_nonzero((magnitude >= threshold) & (neighbors < minimum)))
    return total


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    height, width = values.shape[:2]
    return values.reshape(
        height // factor, factor, width // factor, factor, values.shape[2]
    ).mean(axis=(1, 3), dtype=np.float64)


def _profile_lod(
    profile: DerivativeConditionedStructureProfile,
    factor: int,
    *,
    scale_variance: bool,
) -> DerivativeConditionedStructureProfile:
    return DerivativeConditionedStructureProfile(
        peak_density_variance=(
            profile.peak_density_variance / (factor * factor)
            if scale_variance
            else profile.peak_density_variance
        ),
        correlation_sigma_pixels=profile.correlation_sigma_pixels / factor,
        layer_seeds=profile.layer_seeds,
        exposure_minimum=profile.exposure_minimum,
        exposure_maximum=profile.exposure_maximum,
        normalization_samples=profile.normalization_samples,
    )


def _population_fields(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = (int(item) for item in spec["shape"])
    lo, mid, hi = (float(item) for item in spec["step_exposure_levels"])
    ramp = np.geomspace(*map(float, spec["ramp_exposure_interval"]), width)
    fields = {
        "ramp": np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy(),
    }
    step = np.empty((height, width, 3), dtype=np.float64)
    step[:, : width // 3] = lo
    step[:, width // 3 : 2 * width // 3] = mid
    step[:, 2 * width // 3 :] = hi
    fields["step"] = step
    yy, xx = np.indices((height, width))
    checker = ((yy // int(spec["checkerboard_period_pixels"]) + xx // int(spec["checkerboard_period_pixels"])) % 2)
    checker_levels = np.asarray(spec["checkerboard_exposure_levels"], dtype=np.float64)
    fields["checkerboard"] = np.repeat(checker_levels[checker][..., None], 3, axis=2)
    sparse = np.full((height, width, 3), lo, dtype=np.float64)
    spacing = int(spec["sparse_highlight_spacing_pixels"])
    sparse[spacing // 2 :: spacing, spacing // 2 :: spacing] = float(spec["sparse_highlight_exposure"])
    fields["sparse"] = sparse
    return fields


def evaluate_severe_lod(config: dict[str, Any], root: Path) -> dict[str, Any]:
    _, p4af_contract, p4d_contract, sensitometry = _load_parents(config, root)
    operator = build_operator(sensitometry)
    profile = profile_from_contract(p4af_contract)
    p4d_profiles = profiles_from_contract(p4d_contract)
    spec, gates = config["populations"], config["gates"]
    population_rows = []
    maximum_z = 0.0
    isolated = 0
    maximum_banding = 0.0
    domain_pass = repeat_exact = partition_exact = input_unchanged = True
    ramp_shapes: dict[str, list[float]] = {name: [] for name in ("expected", "candidate", "constant", "p4d")}

    for name, exposure in _population_fields(spec).items():
        preserved = exposure.copy()
        target, normalized = derivative_variance_shape(exposure, operator, profile)
        candidate = render_derivative_conditioned_structure(exposure, operator, profile)
        repeat = render_derivative_conditioned_structure(exposure, operator, profile)
        repeat_exact &= np.array_equal(candidate.density, repeat.density)
        repeat_exact &= np.array_equal(candidate.transmittance, repeat.transmittance)
        assembled = np.empty_like(candidate.density)
        for rows in spec["row_partitions"]:
            for y0 in range(0, exposure.shape[0], int(rows)):
                y1 = min(exposure.shape[0], y0 + int(rows))
                region = render_derivative_conditioned_structure_region(
                    exposure, operator, profile, origin_yx=(y0, 0), shape=(y1 - y0, exposure.shape[1])
                )
                assembled[y0:y1] = region.density
            partition_exact &= np.array_equal(assembled, candidate.density)
        residual = candidate.density.astype(np.float64) - target
        expected_variance = profile.peak_density_variance * normalized
        z = np.zeros_like(residual)
        active = expected_variance > 0.0
        z[active] = residual[active] / np.sqrt(expected_variance[active])
        population_z = float(np.max(np.abs(z)))
        population_isolated = _isolated_count(
            z,
            float(gates["isolated_excursion_z_threshold"]),
            float(gates["isolated_neighbor_support_z_threshold"]),
            int(gates["minimum_isolated_neighbor_support"]),
        )
        maximum_z = max(maximum_z, population_z)
        isolated += population_isolated
        banding = 0.0
        if name == "ramp":
            column_mean = np.mean(residual, axis=0, dtype=np.float64)
            banding = float(np.max(np.abs(np.diff(column_mean, n=2, axis=0))))
            maximum_banding = max(maximum_banding, banding)
            constant = _constant_variance_density(target, profile).astype(np.float64)
            p4d = render_density_conditioned_structure(target, p4d_profiles).density.astype(np.float64)
            edges = np.linspace(0, exposure.shape[1], 25, dtype=int)
            for channel in range(3):
                for index in range(24):
                    sl = np.s_[:, edges[index] : edges[index + 1], channel]
                    ramp_shapes["expected"].append(float(np.mean(expected_variance[sl])))
                    ramp_shapes["candidate"].append(float(np.mean(np.square(residual[sl]))))
                    ramp_shapes["constant"].append(float(np.mean(np.square(constant[sl] - target[sl]))))
                    ramp_shapes["p4d"].append(float(np.mean(np.square(p4d[sl] - target[sl]))))
        domain_pass &= bool(
            np.min(candidate.density) >= 0.0
            and np.min(candidate.transmittance) > 0.0
            and np.max(candidate.transmittance) <= 1.0
        )
        input_unchanged &= np.array_equal(exposure, preserved)
        population_rows.append({
            "population": name,
            "maximum_absolute_standardized_density_residual": population_z,
            "isolated_excursion_count": population_isolated,
            "ramp_column_second_difference": banding,
        })

    expected = np.asarray(ramp_shapes["expected"])
    correlations = {
        name: float(spearmanr(expected, np.asarray(ramp_shapes[name])).statistic)
        for name in ("candidate", "constant", "p4d")
    }
    minimum_advantage = min(
        correlations["candidate"] - correlations["constant"],
        correlations["candidate"] - correlations["p4d"],
    )

    lod_rows = []
    max_lod_mean = max_lod_acf = 0.0
    min_lod_ratio, max_lod_ratio = float("inf"), 0.0
    min_lod_improvement = float("inf")
    height, width = (int(item) for item in spec["shape"])
    for level in spec["flat_exposure_levels"]:
        exposure = np.full((height, width, 3), float(level), dtype=np.float64)
        truth = render_derivative_conditioned_structure(exposure, operator, profile)
        for factor in spec["lod_factors"]:
            low_exposure = _block_mean(exposure, int(factor))
            truth_density = _block_mean(truth.density.astype(np.float64), int(factor))
            scaled = render_derivative_conditioned_structure(
                low_exposure, operator, _profile_lod(profile, int(factor), scale_variance=True)
            ).density.astype(np.float64)
            unscaled = render_derivative_conditioned_structure(
                low_exposure, operator, _profile_lod(profile, int(factor), scale_variance=False)
            ).density.astype(np.float64)
            channel_rows = []
            for channel in range(3):
                truth_layer, scaled_layer, unscaled_layer = (
                    truth_density[..., channel], scaled[..., channel], unscaled[..., channel]
                )
                truth_variance = float(np.var(truth_layer))
                scaled_variance = float(np.var(scaled_layer))
                unscaled_variance = float(np.var(unscaled_layer))
                ratio = scaled_variance / truth_variance
                mean_error = abs(float(np.mean(scaled_layer) - np.mean(truth_layer)))
                acf_error = abs(_lag1(scaled_layer) - _lag1(truth_layer))
                scaled_error = abs(math.log(ratio))
                unscaled_error = abs(math.log(unscaled_variance / truth_variance))
                improvement = 1.0 - scaled_error / unscaled_error if unscaled_error > 0.0 else 0.0
                max_lod_mean = max(max_lod_mean, mean_error)
                min_lod_ratio, max_lod_ratio = min(min_lod_ratio, ratio), max(max_lod_ratio, ratio)
                max_lod_acf = max(max_lod_acf, acf_error)
                min_lod_improvement = min(min_lod_improvement, improvement)
                channel_rows.append({
                    "channel": channel,
                    "mean_density_absolute_error": mean_error,
                    "variance_ratio": ratio,
                    "lag1_acf_absolute_error": acf_error,
                    "variance_error_improvement_over_unscaled": improvement,
                })
            lod_rows.append({"exposure": float(level), "factor": int(factor), "channels": channel_rows})

    metrics = {
        "populations": population_rows,
        "maximum_absolute_standardized_density_residual": maximum_z,
        "total_isolated_excursion_count": isolated,
        "maximum_ramp_column_second_difference": maximum_banding,
        "variance_shape_spearman": correlations,
        "minimum_candidate_spearman_advantage_over_controls": minimum_advantage,
        "lod_rows": lod_rows,
        "maximum_lod_mean_density_absolute_error": max_lod_mean,
        "minimum_lod_variance_ratio": min_lod_ratio,
        "maximum_lod_variance_ratio": max_lod_ratio,
        "maximum_lod_lag1_acf_absolute_error": max_lod_acf,
        "minimum_lod_variance_error_improvement_over_unscaled": min_lod_improvement,
        "density_transmittance_domain": domain_pass,
        "input_unchanged": input_unchanged,
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
    }
    severe = {
        "standardized_residual": maximum_z <= gates["maximum_absolute_standardized_density_residual"],
        "isolated_excursions": isolated <= gates["maximum_isolated_excursion_count"],
        "ramp_banding": maximum_banding <= gates["maximum_ramp_column_second_difference"],
        "variance_shape": correlations["candidate"] >= gates["minimum_candidate_variance_shape_spearman"],
        "negative_controls": minimum_advantage >= gates["minimum_candidate_spearman_advantage_over_each_control"],
        "physical_domain": domain_pass,
        "input_unchanged": input_unchanged,
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
    }
    lod = {
        "mean": max_lod_mean <= gates["maximum_lod_mean_density_absolute_error"],
        "variance": min_lod_ratio >= gates["minimum_lod_variance_ratio"] and max_lod_ratio <= gates["maximum_lod_variance_ratio"],
        "acf": max_lod_acf <= gates["maximum_lod_lag1_acf_absolute_error"],
        "beats_unscaled": min_lod_improvement >= gates["minimum_lod_variance_error_improvement_over_unscaled"],
    }
    severe_pass, lod_pass = all(severe.values()), all(lod.values())
    decision_key = "all_pass" if severe_pass and lod_pass else ("severe_pass_lod_fail" if severe_pass else "severe_fail")
    stable = {
        "experiment_id": config["experiment_id"],
        "metrics": metrics,
        "severe_gates": severe,
        "lod_gates": lod,
        "severe_pass": severe_pass,
        "lod_pass": lod_pass,
        "decision_key": decision_key,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"][decision_key],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["DerivativeStructureSevereLodError", "REPORT_SCHEMA", "SCHEMA", "evaluate_severe_lod", "load_contract"]
