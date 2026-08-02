"""U6.P4BQ held-band test of bounded-lognormal Boolean disc geometry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from src.eval.fixed_disc_boolean_nps_compatibility import (
    _band_mask,
    _candidate_shape,
    _fit_gaussian_control,
    _fit_log_amplitude,
    _log10_rmse,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.variable_disc_boolean_nps import (
    variable_disc_boolean_radial_nps,
)

SCHEMA = "neuro_film.u6_p4bq_lognormal_disc_boolean_nps_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bq_lognormal_disc_boolean_nps_compatibility_report.v1"


class LognormalDiscBooleanNPSCompatibilityError(RuntimeError):
    """Raised when frozen P4BQ evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise LognormalDiscBooleanNPSCompatibilityError("P4BQ paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    fit = payload.get("fit", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("median_radius_micrometres_bounds") != [0.25, 12.0]
        or model.get("log_radius_sigma_bounds") != [0.05, 0.55]
        or model.get("radius_cdf_interval") != [0.00001, 0.99999]
        or model.get("radius_distribution_quadrature_nodes") != 32
        or model.get("radial_hankel_quadrature_samples") != 4097
        or model.get("coverage_probability_bounds") != [0.02, 0.98]
        or model.get("clean_room_equation_use_only") is not True
        or model.get("source_code_reuse_allowed") is not False
        or fit.get("build_bands_lines_per_mm") != [[5.0, 100.0], [255.0, 350.0]]
        or fit.get("confirmation_bands_lines_per_mm")
        != [[105.0, 250.0], [355.0, 500.0]]
        or fit.get("optimizer_seed") != 2608022601
        or fit.get("optimizer_max_iterations") != 120
        or fit.get("optimizer_population_size") != 12
        or fit.get(
            "post_score_distribution_bound_quadrature_frequency_split_model_or_gate_tuning_allowed"
        )
        is not False
        or fit.get("render_allowed") is not False
        or evaluation.get("required_material_count") != 2
        or evaluation.get("required_profile_count") != 5
        or evaluation.get(
            "minimum_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.1
        or evaluation.get(
            "minimum_confirmation_median_relative_improvement_over_fixed_disc"
        )
        != 0.15
        or evaluation.get(
            "minimum_each_material_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.05
        or evaluation.get(
            "maximum_candidate_to_gaussian_worst_confirmation_error_ratio"
        )
        != 1.0
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise LognormalDiscBooleanNPSCompatibilityError("P4BQ frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise LognormalDiscBooleanNPSCompatibilityError(
            f"P4BQ parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _variable_shape(
    profile: HistoricalBWNoiseSpectrumProfile,
    frequency: np.ndarray,
    median_radius_millimetres: float,
    log_radius_sigma: float,
    coverage_probability: float,
    model: Mapping[str, Any],
) -> np.ndarray:
    intrinsic = variable_disc_boolean_radial_nps(
        frequency,
        median_radius_millimetres,
        log_radius_sigma,
        coverage_probability,
        cdf_interval=tuple(float(value) for value in model["radius_cdf_interval"]),
        radius_nodes=int(model["radius_distribution_quadrature_nodes"]),
        radial_samples=int(model["radial_hankel_quadrature_samples"]),
    )
    return intrinsic * np.square(profile.circular_aperture_mtf(frequency))


def _fit_material_candidate(
    profiles: Sequence[HistoricalBWNoiseSpectrumProfile],
    build_frequency: np.ndarray,
    build_targets: Sequence[np.ndarray],
    model: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    radius_bounds = tuple(
        float(value) for value in model["median_radius_micrometres_bounds"]
    )
    sigma_bounds = tuple(float(value) for value in model["log_radius_sigma_bounds"])
    coverage_bounds = tuple(
        float(value) for value in model["coverage_probability_bounds"]
    )

    def objective(parameters: np.ndarray) -> float:
        radius_mm = float(parameters[0]) / 1000.0
        sigma = float(parameters[1])
        errors = []
        try:
            for profile, target, coverage in zip(
                profiles, build_targets, parameters[2:], strict=True
            ):
                shape = _variable_shape(
                    profile,
                    build_frequency,
                    radius_mm,
                    sigma,
                    float(coverage),
                    model,
                )
                amplitude = _fit_log_amplitude(shape, target)
                errors.append(_log10_rmse(shape * (10.0**amplitude), target) ** 2)
        except RuntimeError:
            return 1e6
        return float(np.mean(errors))

    result = differential_evolution(
        objective,
        bounds=[
            radius_bounds,
            sigma_bounds,
            *([coverage_bounds] * len(profiles)),
        ],
        seed=int(fit["optimizer_seed"]),
        maxiter=int(fit["optimizer_max_iterations"]),
        popsize=int(fit["optimizer_population_size"]),
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-9,
        atol=1e-12,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise LognormalDiscBooleanNPSCompatibilityError(
            f"P4BQ candidate optimization failed: {result.message}"
        )
    parameters = np.asarray(result.x, dtype=np.float64)
    amplitudes = []
    for profile, target, coverage in zip(
        profiles, build_targets, parameters[2:], strict=True
    ):
        shape = _variable_shape(
            profile,
            build_frequency,
            float(parameters[0]) / 1000.0,
            float(parameters[1]),
            float(coverage),
            model,
        )
        amplitudes.append(_fit_log_amplitude(shape, target))
    return (
        float(parameters[0]),
        float(parameters[1]),
        parameters[2:],
        np.asarray(amplitudes),
    )


def evaluate_lognormal_disc_boolean_nps_compatibility(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    bundle = _load_parent(parents, root, "p4bl_bundle")
    _load_parent(parents, root, "p4bp_contract")
    fixed_decision = _load_parent(parents, root, "p4bp_decision")
    fixed_report = _load_parent(parents, root, "p4bp_report")
    if (
        bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or fixed_decision.get("decision")
        != "close_fixed_disc_boolean_measured_nps_equation_family"
        or fixed_report.get("stable_evidence_id") != parents["p4bp_stable_evidence_id"]
    ):
        raise LognormalDiscBooleanNPSCompatibilityError("P4BQ parent decision mismatch")
    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    model = contract["model"]
    fit = contract["fit"]
    grid = fit["frequency_grid_lines_per_mm"]
    frequency = np.arange(
        float(grid["minimum"]),
        float(grid["maximum"]) + 0.5 * float(grid["step"]),
        float(grid["step"]),
    )
    build_mask = _band_mask(frequency, fit["build_bands_lines_per_mm"])
    confirmation_mask = _band_mask(frequency, fit["confirmation_bands_lines_per_mm"])
    target_by_id = {
        profile.identity(): profile.aperture_convolved_2d(frequency, 0.0)
        for profile in profiles
    }
    fixed_rows = {row["profile_id"]: row for row in fixed_report["rows"]}
    grouped: dict[str, list[HistoricalBWNoiseSpectrumProfile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.material_id, []).append(profile)
    for material_profiles in grouped.values():
        material_profiles.sort(key=lambda item: item.diffuse_density)

    rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    gaussian_errors: list[float] = []
    fixed_errors: list[float] = []
    material_gaussian_improvements = []
    all_coverage_monotone = True
    all_interior = True
    all_positive = True
    all_replay = True
    margin_fraction = float(
        contract["evaluation"]["interior_parameter_margin_fraction"]
    )
    radius_lower, radius_upper = (
        float(value) for value in model["median_radius_micrometres_bounds"]
    )
    sigma_lower, sigma_upper = (
        float(value) for value in model["log_radius_sigma_bounds"]
    )
    for material_id in sorted(grouped):
        material_profiles = grouped[material_id]
        build_targets = [
            target_by_id[p.identity()][build_mask] for p in material_profiles
        ]
        radius_um, sigma, coverages, amplitudes = _fit_material_candidate(
            material_profiles,
            frequency[build_mask],
            build_targets,
            model,
            fit,
        )
        all_coverage_monotone &= bool(np.all(np.diff(coverages) >= 0.0))
        radius_margin = (radius_upper - radius_lower) * margin_fraction
        sigma_margin = (sigma_upper - sigma_lower) * margin_fraction
        all_interior &= (
            radius_lower + radius_margin < radius_um < radius_upper - radius_margin
            and sigma_lower + sigma_margin < sigma < sigma_upper - sigma_margin
        )
        material_candidate = []
        material_gaussian = []
        for profile, coverage, amplitude in zip(
            material_profiles, coverages, amplitudes, strict=True
        ):
            target = target_by_id[profile.identity()]
            candidate_shape = _variable_shape(
                profile,
                frequency,
                radius_um / 1000.0,
                sigma,
                float(coverage),
                model,
            )
            candidate = candidate_shape * (10.0 ** float(amplitude))
            gaussian_scale, gaussian_amplitude = _fit_gaussian_control(
                profile,
                frequency[build_mask],
                target[build_mask],
                [5.0, 1000.0],
            )
            gaussian = np.exp(
                -np.square(frequency) / (2.0 * gaussian_scale * gaussian_scale)
            ) * np.square(profile.circular_aperture_mtf(frequency))
            gaussian *= 10.0**gaussian_amplitude
            fixed = fixed_rows[profile.identity()]
            fixed_prediction = _candidate_shape(
                profile,
                frequency,
                float(fixed["fixed_disc_radius_micrometres"]) / 1000.0,
                float(fixed["coverage_probability"]),
                4097,
            ) * (10.0 ** float(fixed["candidate_log10_amplitude"]))
            candidate_error = _log10_rmse(
                candidate[confirmation_mask], target[confirmation_mask]
            )
            gaussian_error = _log10_rmse(
                gaussian[confirmation_mask], target[confirmation_mask]
            )
            fixed_error = _log10_rmse(
                fixed_prediction[confirmation_mask], target[confirmation_mask]
            )
            replay = _variable_shape(
                profile,
                frequency,
                radius_um / 1000.0,
                sigma,
                float(coverage),
                model,
            ) * (10.0 ** float(amplitude))
            all_replay &= bool(np.array_equal(candidate, replay))
            all_positive &= bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(gaussian > 0.0)
                and np.all(fixed_prediction > 0.0)
            )
            candidate_errors.append(candidate_error)
            gaussian_errors.append(gaussian_error)
            fixed_errors.append(fixed_error)
            material_candidate.append(candidate_error)
            material_gaussian.append(gaussian_error)
            rows.append(
                {
                    "profile_id": profile.identity(),
                    "material_id": material_id,
                    "diffuse_density": profile.diffuse_density,
                    "median_radius_micrometres": radius_um,
                    "log_radius_sigma": sigma,
                    "coverage_probability": float(coverage),
                    "candidate_log10_amplitude": float(amplitude),
                    "candidate_confirmation_log10_rmse": candidate_error,
                    "gaussian_confirmation_log10_rmse": gaussian_error,
                    "fixed_disc_confirmation_log10_rmse": fixed_error,
                }
            )
        material_candidate_median = float(np.median(material_candidate))
        material_gaussian_median = float(np.median(material_gaussian))
        material_improvement = (
            material_gaussian_median - material_candidate_median
        ) / material_gaussian_median
        material_gaussian_improvements.append(material_improvement)
        material_rows.append(
            {
                "material_id": material_id,
                "profile_count": len(material_profiles),
                "shared_median_radius_micrometres": radius_um,
                "shared_log_radius_sigma": sigma,
                "coverage_probabilities_in_density_order": coverages.tolist(),
                "candidate_confirmation_median_log10_rmse": material_candidate_median,
                "gaussian_confirmation_median_log10_rmse": material_gaussian_median,
                "confirmation_median_relative_improvement_over_gaussian": material_improvement,
            }
        )

    candidate_median = float(np.median(candidate_errors))
    gaussian_median = float(np.median(gaussian_errors))
    fixed_median = float(np.median(fixed_errors))
    candidate_worst = max(candidate_errors)
    gaussian_worst = max(gaussian_errors)
    evaluation = contract["evaluation"]
    gates = {
        "material_count": len(grouped) == evaluation["required_material_count"],
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "confirmation_median_beats_gaussian": (
            (gaussian_median - candidate_median) / gaussian_median
            >= evaluation[
                "minimum_confirmation_median_relative_improvement_over_gaussian"
            ]
        ),
        "confirmation_median_beats_fixed_disc": (
            (fixed_median - candidate_median) / fixed_median
            >= evaluation[
                "minimum_confirmation_median_relative_improvement_over_fixed_disc"
            ]
        ),
        "each_material_confirmation_median_beats_gaussian": all(
            value
            >= evaluation[
                "minimum_each_material_confirmation_median_relative_improvement_over_gaussian"
            ]
            for value in material_gaussian_improvements
        ),
        "worst_confirmation_not_worse_than_gaussian": (
            candidate_worst / gaussian_worst
            <= evaluation[
                "maximum_candidate_to_gaussian_worst_confirmation_error_ratio"
            ]
        ),
        "density_monotone_coverage": all_coverage_monotone,
        "interior_distribution_parameters": all_interior,
        "finite_positive_predictions": all_positive,
        "parameter_prediction_replay_exact": all_replay,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "parent_fixed_disc_stable_evidence_id": parents["p4bp_stable_evidence_id"],
        "material_count": len(grouped),
        "profile_count": len(profiles),
        "candidate_confirmation_median_log10_rmse": candidate_median,
        "gaussian_confirmation_median_log10_rmse": gaussian_median,
        "fixed_disc_confirmation_median_log10_rmse": fixed_median,
        "confirmation_median_relative_improvement_over_gaussian": (
            (gaussian_median - candidate_median) / gaussian_median
        ),
        "confirmation_median_relative_improvement_over_fixed_disc": (
            (fixed_median - candidate_median) / fixed_median
        ),
        "candidate_confirmation_worst_log10_rmse": candidate_worst,
        "gaussian_confirmation_worst_log10_rmse": gaussian_worst,
        "candidate_to_gaussian_worst_error_ratio": candidate_worst / gaussian_worst,
        "materials": material_rows,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_lognormal_disc_boolean_measured_nps_equation_family"
            if automatic_pass
            else "close_independent_disc_boolean_measured_nps_equation_family"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    stable_payload.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "LognormalDiscBooleanNPSCompatibilityError",
    "evaluate_lognormal_disc_boolean_nps_compatibility",
    "load_contract",
    "write_report",
]
