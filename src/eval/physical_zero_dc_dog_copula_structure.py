"""Frozen P4AO audit of a zero-DC Gaussian-band-pass density structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.physical_balanced_gaussian_copula_structure import (
    _block_std,
    _canonical,
    _checker_ratio,
    _isolated,
    _lag_xy,
)
from src.eval.physical_callier_source import hash_file
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    DerivativeConditionedStructureProfile,
    derivative_variance_shape,
    render_derivative_conditioned_structure,
    render_zero_dc_dog_derivative_structure,
    render_zero_dc_dog_derivative_structure_region,
)
from src.film_physics.structure_compiler import (
    correlated_normal_region,
    zero_dc_dog_kernel_metrics,
    zero_dc_dog_normal_region,
)


SCHEMA = "neuro_film.u6_p4ao_zero_dc_dog_copula_structure_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ao_zero_dc_dog_copula_structure_report.v1"


class ZeroDcDogCopulaError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model = value.get("model", {})
    evaluation = value.get("evaluation", {})
    gates = value.get("gates", {})
    if (
        value.get("schema") != SCHEMA
        or model.get("narrow_sigma_pixels") != 0.65
        or model.get("broad_sigma_pixels") != 1.2
        or model.get("broad_weight") != 1.0
        or model.get("peak_density_variance") != 0.0001
        or model.get("layer_seeds") != [260831, 260837, 260851]
        or model.get("fitted_parameters") != 0
        or model.get("sample_or_image_centering_allowed")
        or evaluation.get("shape") != [512, 768]
        or evaluation.get("low_frequency_block_sizes") != [4, 8]
        or evaluation.get("row_partitions") != [31, 127]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("minimum_flat_transmittance_error_improvement_over_p4af")
        != 0.5
        or gates.get("maximum_low_frequency_block_mean_std_ratio_to_p4af")
        != 0.5
        or not gates.get("no_parameter_fit")
    ):
        raise ZeroDcDogCopulaError("P4AO frozen contract drift")
    return value


def _parents(
    contract: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    identities: dict[str, str] = {}
    for stem in ("p4an_decision", "p4af_decision", "sensitometry_contract"):
        relative = Path(contract["parents"][f"{stem}_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ZeroDcDogCopulaError("P4AO parent path escaped root")
        actual = hash_file(root / relative)
        if actual != contract["parents"][f"{stem}_sha256"]:
            raise ZeroDcDogCopulaError(f"P4AO parent mismatch: {stem}")
        identities[stem] = actual
    sensitometry = json.loads(
        (root / contract["parents"]["sensitometry_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    return sensitometry, identities


def _profile(contract: dict[str, Any]) -> DerivativeConditionedStructureProfile:
    model = contract["model"]
    return DerivativeConditionedStructureProfile(
        float(model["peak_density_variance"]),
        float(model["narrow_sigma_pixels"]),
        tuple(model["layer_seeds"]),
        float(model["variance_normalization_domain"][0]),
        float(model["variance_normalization_domain"][1]),
        int(model["variance_normalization_samples"]),
    )


def evaluate_structure(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    sensitometry, parent_ids = _parents(contract, root)
    operator = build_operator(sensitometry)
    profile = _profile(contract)
    model = contract["model"]
    evaluation = contract["evaluation"]
    gates = contract["gates"]
    narrow = float(model["narrow_sigma_pixels"])
    broad = float(model["broad_sigma_pixels"])
    broad_weight = float(model["broad_weight"])
    height, width = evaluation["shape"]
    border = int(evaluation["interior_border_pixels"])
    interior = np.s_[border:-border, border:-border, :]
    dc_sum, analytic_variance = zero_dc_dog_kernel_metrics(
        narrow, broad, broad_weight=broad_weight
    )

    flat_errors: list[float] = []
    baseline_errors: list[float] = []
    variance_errors: list[float] = []
    lag_values: list[tuple[float, float]] = []
    isolated = 0
    input_unchanged = True
    for level in evaluation["flat_exposure_levels"]:
        source = np.full((height, width, 3), level, dtype=np.float64)
        original = source.copy()
        mean, shape = derivative_variance_shape(source, operator, profile)
        target_variance = profile.peak_density_variance * shape
        candidate = render_zero_dc_dog_derivative_structure(
            source,
            operator,
            profile,
            narrow_sigma=narrow,
            broad_sigma=broad,
            broad_weight=broad_weight,
        )
        baseline = render_derivative_conditioned_structure(source, operator, profile)
        target_transmittance = 10.0 ** (-mean[0, 0])
        flat_errors.append(
            float(
                np.max(
                    np.abs(
                        np.mean(candidate.transmittance[interior], axis=(0, 1))
                        - target_transmittance
                    )
                )
            )
        )
        baseline_errors.append(
            float(
                np.max(
                    np.abs(
                        np.mean(baseline.transmittance[interior], axis=(0, 1))
                        - target_transmittance
                    )
                )
            )
        )
        residual = candidate.density[interior].astype(np.float64) - mean[interior]
        active = target_variance[0, 0] > 0.0
        observed = np.mean(np.square(residual), axis=(0, 1))
        variance_errors.extend(
            (
                np.abs(observed[active] - target_variance[0, 0, active])
                / target_variance[0, 0, active]
            ).tolist()
        )
        standardized = residual / np.sqrt(target_variance[0, 0])
        isolated += _isolated(standardized)
        lag_values.extend(_lag_xy(standardized[..., channel]) for channel in range(3))
        input_unchanged &= np.array_equal(source, original)

    maximum_flat_error = max(flat_errors)
    maximum_baseline_error = max(baseline_errors)
    flat_improvement = 1.0 - maximum_flat_error / maximum_baseline_error

    ramp_values = np.geomspace(*evaluation["ramp_exposure_interval"], width)
    ramp = np.broadcast_to(
        ramp_values[None, :, None], (height, width, 3)
    ).copy()
    mean, shape = derivative_variance_shape(ramp, operator, profile)
    candidate = render_zero_dc_dog_derivative_structure(
        ramp,
        operator,
        profile,
        narrow_sigma=narrow,
        broad_sigma=broad,
        broad_weight=broad_weight,
    )
    repeated = render_zero_dc_dog_derivative_structure(
        ramp,
        operator,
        profile,
        narrow_sigma=narrow,
        broad_sigma=broad,
        broad_weight=broad_weight,
    )
    repeat_exact = np.array_equal(candidate.density, repeated.density) and np.array_equal(
        candidate.transmittance, repeated.transmittance
    )
    assembled_density = np.empty_like(candidate.density)
    assembled_transmittance = np.empty_like(candidate.transmittance)
    partition_exact = True
    for rows in evaluation["row_partitions"]:
        for y0 in range(0, height, rows):
            region = render_zero_dc_dog_derivative_structure_region(
                ramp,
                operator,
                profile,
                origin_yx=(y0, 0),
                shape=(min(rows, height - y0), width),
                narrow_sigma=narrow,
                broad_sigma=broad,
                broad_weight=broad_weight,
            )
            y1 = y0 + region.density.shape[0]
            assembled_density[y0:y1] = region.density
            assembled_transmittance[y0:y1] = region.transmittance
        partition_exact &= np.array_equal(candidate.density, assembled_density)
        partition_exact &= np.array_equal(
            candidate.transmittance, assembled_transmittance
        )

    residual = candidate.density.astype(np.float64) - mean
    observed_bins: list[float] = []
    expected_bins: list[float] = []
    edges = np.linspace(0, width, evaluation["ramp_bins"] + 1, dtype=int)
    for channel in range(3):
        for x0, x1 in zip(edges[:-1], edges[1:]):
            observed_bins.append(
                float(np.mean(np.square(residual[border:-border, x0:x1, channel])))
            )
            expected_bins.append(
                float(
                    profile.peak_density_variance
                    * np.mean(shape[border:-border, x0:x1, channel])
                )
            )
    variance_spearman = float(
        spearmanr(expected_bins, observed_bins).statistic
    )

    band_pass = zero_dc_dog_normal_region(
        (height, width),
        origin_yx=(0, 0),
        shape=(height, width),
        narrow_sigma=narrow,
        broad_sigma=broad,
        broad_weight=broad_weight,
        seed=profile.layer_seeds[0],
    )[border:-border, border:-border]
    baseline = correlated_normal_region(
        (height, width),
        origin_yx=(0, 0),
        shape=(height, width),
        sigma=narrow,
        seed=profile.layer_seeds[0],
    )[border:-border, border:-border]
    low_frequency_ratios = [
        _block_std(band_pass, factor) / _block_std(baseline, factor)
        for factor in evaluation["low_frequency_block_sizes"]
    ]
    lag_x, lag_y = _lag_xy(band_pass)
    metrics = {
        "kernel_dc_absolute_sum": abs(dc_sum),
        "analytic_output_variance": analytic_variance,
        "maximum_empirical_flat_transmittance_mean_absolute_error": maximum_flat_error,
        "maximum_p4af_flat_transmittance_mean_absolute_error": maximum_baseline_error,
        "flat_transmittance_error_improvement_over_p4af": flat_improvement,
        "maximum_density_variance_relative_error": max(variance_errors),
        "ramp_variance_shape_spearman": variance_spearman,
        "lag1_x": lag_x,
        "lag1_y": lag_y,
        "minimum_lag1_autocorrelation": min(lag_x, lag_y),
        "maximum_lag1_autocorrelation": max(lag_x, lag_y),
        "lag1_anisotropy": abs(lag_x - lag_y),
        "low_frequency_block_mean_std_ratios_to_p4af": low_frequency_ratios,
        "maximum_low_frequency_block_mean_std_ratio_to_p4af": max(
            low_frequency_ratios
        ),
        "checker_frequency_power_ratio_to_neighborhood": _checker_ratio(band_pass),
        "isolated_standardized_excursions": isolated,
        "density_minimum": float(np.min(candidate.density)),
        "transmittance_minimum": float(np.min(candidate.transmittance)),
        "transmittance_maximum": float(np.max(candidate.transmittance)),
        "repeat_exact": bool(repeat_exact),
        "row_partition_exact": bool(partition_exact),
        "input_unchanged": bool(input_unchanged),
    }
    checks = {
        "kernel_dc": metrics["kernel_dc_absolute_sum"]
        <= gates["maximum_kernel_dc_absolute_sum"],
        "flat_transmittance_mean": maximum_flat_error
        <= gates["maximum_empirical_flat_transmittance_mean_absolute_error"],
        "flat_improvement": flat_improvement
        >= gates["minimum_flat_transmittance_error_improvement_over_p4af"],
        "density_variance": metrics["maximum_density_variance_relative_error"]
        <= gates["maximum_density_variance_relative_error"],
        "variance_shape": variance_spearman
        >= gates["minimum_ramp_variance_shape_spearman"],
        "lag1_range": min(lag_x, lag_y) >= gates["minimum_lag1_autocorrelation"]
        and max(lag_x, lag_y) <= gates["maximum_lag1_autocorrelation"],
        "lag1_anisotropy": abs(lag_x - lag_y)
        <= gates["maximum_lag1_anisotropy"],
        "low_frequency": max(low_frequency_ratios)
        <= gates["maximum_low_frequency_block_mean_std_ratio_to_p4af"],
        "checker_frequency": metrics["checker_frequency_power_ratio_to_neighborhood"]
        <= gates["maximum_checker_frequency_power_ratio_to_neighborhood"],
        "isolated": isolated <= gates["maximum_isolated_standardized_excursions"],
        "physical_domain": metrics["density_minimum"] >= 0.0
        and metrics["transmittance_minimum"] > 0.0
        and metrics["transmittance_maximum"] <= 1.0,
        "repeat_exact": bool(repeat_exact),
        "row_partition_exact": bool(partition_exact),
        "input_unchanged": bool(input_unchanged),
        "no_parameter_fit": True,
    }
    stable = {
        "experiment_id": contract["experiment_id"],
        "parent_identities": parent_ids,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    branch = "pass" if stable["passed"] else "fail"
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": contract["branch_rule"][branch],
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "ZeroDcDogCopulaError",
    "evaluate_structure",
    "load_contract",
]
