"""U6.P4BP measured-spectrum compatibility for fixed-disc Boolean geometry."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar

from src.film_physics.fixed_disc_boolean_nps import (
    fixed_disc_boolean_radial_nps,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)

SCHEMA = "neuro_film.u6_p4bp_fixed_disc_boolean_nps_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bp_fixed_disc_boolean_nps_compatibility_report.v1"


class FixedDiscBooleanNPSCompatibilityError(RuntimeError):
    """Raised when frozen P4BP evidence or semantics drift."""


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
        raise FixedDiscBooleanNPSCompatibilityError("P4BP paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit", {})
    evaluation = payload.get("evaluation", {})
    source = payload.get("primary_model_source", {})
    if (
        payload.get("schema") != SCHEMA
        or fit.get("build_bands_lines_per_mm") != [[5.0, 100.0], [255.0, 350.0]]
        or fit.get("confirmation_bands_lines_per_mm")
        != [[105.0, 250.0], [355.0, 500.0]]
        or fit.get("candidate_radius_micrometres_bounds") != [0.25, 20.0]
        or fit.get("candidate_coverage_probability_bounds") != [0.02, 0.98]
        or fit.get("candidate_radial_quadrature_samples") != 4097
        or fit.get("control_scale_lines_per_mm_bounds") != [5.0, 1000.0]
        or fit.get("optimizer_seed") != 2608022501
        or fit.get("optimizer_max_iterations") != 120
        or fit.get("optimizer_population_size") != 12
        or fit.get("post_score_radius_coverage_frequency_split_model_or_gate_tuning_allowed")
        is not False
        or fit.get("render_allowed") is not False
        or source.get("source_code_reuse_allowed") is not False
        or source.get("clean_room_equation_use_only") is not True
        or evaluation.get("required_material_count") != 2
        or evaluation.get("required_profile_count") != 5
        or evaluation.get("minimum_confirmation_median_relative_improvement_over_gaussian")
        != 0.1
        or evaluation.get("minimum_each_material_confirmation_median_relative_improvement")
        != 0.05
        or evaluation.get("maximum_candidate_to_control_worst_confirmation_error_ratio")
        != 1.0
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise FixedDiscBooleanNPSCompatibilityError("P4BP frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise FixedDiscBooleanNPSCompatibilityError(
            f"P4BP parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _band_mask(frequency: np.ndarray, bands: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(frequency.shape, dtype=bool)
    for lower, upper in bands:
        mask |= (frequency >= float(lower)) & (frequency <= float(upper))
    return mask


def _fit_log_amplitude(shape: np.ndarray, target: np.ndarray) -> float:
    if np.any(shape <= 0.0) or np.any(target <= 0.0):
        raise FixedDiscBooleanNPSCompatibilityError(
            "P4BP fit requires positive spectra"
        )
    return float(np.mean(np.log10(target) - np.log10(shape)))


def _log10_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(np.log10(actual) - np.log10(expected)),
                dtype=np.float64,
            )
        )
    )


def _candidate_shape(
    profile: HistoricalBWNoiseSpectrumProfile,
    frequency: np.ndarray,
    radius_millimetres: float,
    coverage_probability: float,
    quadrature_samples: int,
) -> np.ndarray:
    intrinsic = fixed_disc_boolean_radial_nps(
        frequency,
        radius_millimetres,
        coverage_probability,
        quadrature_samples=quadrature_samples,
    )
    return intrinsic * np.square(profile.circular_aperture_mtf(frequency))


def _fit_material_candidate(
    profiles: Sequence[HistoricalBWNoiseSpectrumProfile],
    build_frequency: np.ndarray,
    build_targets: Sequence[np.ndarray],
    fit: Mapping[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    radius_bounds = tuple(float(value) for value in fit["candidate_radius_micrometres_bounds"])
    coverage_bounds = tuple(
        float(value) for value in fit["candidate_coverage_probability_bounds"]
    )
    quadrature_samples = int(fit["candidate_radial_quadrature_samples"])

    def objective(parameters: np.ndarray) -> float:
        radius_mm = float(parameters[0]) / 1000.0
        errors = []
        for profile, target, coverage in zip(
            profiles, build_targets, parameters[1:], strict=True
        ):
            shape = _candidate_shape(
                profile,
                build_frequency,
                radius_mm,
                float(coverage),
                quadrature_samples,
            )
            log_amplitude = _fit_log_amplitude(shape, target)
            prediction = shape * (10.0**log_amplitude)
            errors.append(_log10_rmse(prediction, target) ** 2)
        return float(np.mean(errors))

    result = differential_evolution(
        objective,
        bounds=[radius_bounds, *([coverage_bounds] * len(profiles))],
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
        raise FixedDiscBooleanNPSCompatibilityError(
            f"P4BP candidate optimization failed: {result.message}"
        )
    parameters = np.asarray(result.x, dtype=np.float64)
    radius_mm = float(parameters[0]) / 1000.0
    log_amplitudes = []
    for profile, target, coverage in zip(
        profiles, build_targets, parameters[1:], strict=True
    ):
        shape = _candidate_shape(
            profile,
            build_frequency,
            radius_mm,
            float(coverage),
            quadrature_samples,
        )
        log_amplitudes.append(_fit_log_amplitude(shape, target))
    return float(parameters[0]), parameters[1:], np.asarray(log_amplitudes)


def _fit_gaussian_control(
    profile: HistoricalBWNoiseSpectrumProfile,
    build_frequency: np.ndarray,
    build_target: np.ndarray,
    scale_bounds: Sequence[float],
) -> tuple[float, float]:
    lower, upper = (float(value) for value in scale_bounds)

    def objective(scale: float) -> float:
        shape = np.exp(-np.square(build_frequency) / (2.0 * scale * scale))
        shape *= np.square(profile.circular_aperture_mtf(build_frequency))
        log_amplitude = _fit_log_amplitude(shape, build_target)
        return _log10_rmse(shape * (10.0**log_amplitude), build_target) ** 2

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-11, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise FixedDiscBooleanNPSCompatibilityError(
            "P4BP Gaussian control optimization failed"
        )
    scale = float(result.x)
    shape = np.exp(-np.square(build_frequency) / (2.0 * scale * scale))
    shape *= np.square(profile.circular_aperture_mtf(build_frequency))
    return scale, _fit_log_amplitude(shape, build_target)


def evaluate_fixed_disc_boolean_nps_compatibility(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    _load_parent(parents, root, "p4bl_contract")
    decision = _load_parent(parents, root, "p4bl_decision")
    bundle = _load_parent(parents, root, "p4bl_bundle")
    parent_report = _load_parent(parents, root, "p4bl_report")
    if (
        decision.get("automatic_pass") is not True
        or decision.get("decision") != "retain_historical_bw_measured_nps_profiles"
        or bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or parent_report.get("stable_evidence_id")
        != parents["p4bl_stable_evidence_id"]
    ):
        raise FixedDiscBooleanNPSCompatibilityError("P4BP parent decision mismatch")
    source = contract["primary_model_source"]
    for stem in ("article_low_resolution_pdf", "official_source_archive"):
        path = _relative(root, str(source[f"{stem}_path"]))
        if not path.is_file() or _hash_file(path) != source[f"{stem}_sha256"]:
            raise FixedDiscBooleanNPSCompatibilityError(
                f"P4BP primary source mismatch: {stem}"
            )

    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    fit = contract["fit"]
    grid = fit["frequency_grid_lines_per_mm"]
    frequency = np.arange(
        float(grid["minimum"]),
        float(grid["maximum"]) + 0.5 * float(grid["step"]),
        float(grid["step"]),
        dtype=np.float64,
    )
    build_mask = _band_mask(frequency, fit["build_bands_lines_per_mm"])
    confirmation_mask = _band_mask(
        frequency, fit["confirmation_bands_lines_per_mm"]
    )
    if np.any(build_mask & confirmation_mask) or not np.all(
        build_mask | confirmation_mask
    ):
        raise FixedDiscBooleanNPSCompatibilityError("P4BP frequency split drift")
    target_by_id = {
        profile.identity(): profile.aperture_convolved_2d(frequency, 0.0)
        for profile in profiles
    }
    grouped: dict[str, list[HistoricalBWNoiseSpectrumProfile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.material_id, []).append(profile)
    for material_profiles in grouped.values():
        material_profiles.sort(key=lambda item: item.diffuse_density)

    rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    all_predictions_positive = True
    all_parameter_replays_exact = True
    all_coverage_monotone = True
    all_radius_interior = True
    candidate_confirmation_errors: list[float] = []
    control_confirmation_errors: list[float] = []
    material_improvements: list[float] = []
    radius_lower, radius_upper = (
        float(value) for value in fit["candidate_radius_micrometres_bounds"]
    )
    radius_margin = (radius_upper - radius_lower) * float(
        contract["evaluation"]["interior_radius_margin_fraction"]
    )
    for material_id in sorted(grouped):
        material_profiles = grouped[material_id]
        build_targets = [
            target_by_id[profile.identity()][build_mask]
            for profile in material_profiles
        ]
        radius_um, coverages, log_amplitudes = _fit_material_candidate(
            material_profiles,
            frequency[build_mask],
            build_targets,
            fit,
        )
        all_coverage_monotone &= bool(np.all(np.diff(coverages) >= 0.0))
        all_radius_interior &= (
            radius_lower + radius_margin
            < radius_um
            < radius_upper - radius_margin
        )
        material_candidate_errors = []
        material_control_errors = []
        for profile, coverage, candidate_log_amplitude in zip(
            material_profiles, coverages, log_amplitudes, strict=True
        ):
            target = target_by_id[profile.identity()]
            candidate_shape = _candidate_shape(
                profile,
                frequency,
                radius_um / 1000.0,
                float(coverage),
                int(fit["candidate_radial_quadrature_samples"]),
            )
            candidate = candidate_shape * (10.0 ** float(candidate_log_amplitude))
            control_scale, control_log_amplitude = _fit_gaussian_control(
                profile,
                frequency[build_mask],
                target[build_mask],
                fit["control_scale_lines_per_mm_bounds"],
            )
            control_shape = np.exp(
                -np.square(frequency) / (2.0 * control_scale * control_scale)
            ) * np.square(profile.circular_aperture_mtf(frequency))
            control = control_shape * (10.0**control_log_amplitude)
            candidate_build_error = _log10_rmse(
                candidate[build_mask], target[build_mask]
            )
            candidate_confirmation_error = _log10_rmse(
                candidate[confirmation_mask], target[confirmation_mask]
            )
            control_build_error = _log10_rmse(
                control[build_mask], target[build_mask]
            )
            control_confirmation_error = _log10_rmse(
                control[confirmation_mask], target[confirmation_mask]
            )
            replay = _candidate_shape(
                profile,
                frequency,
                radius_um / 1000.0,
                float(coverage),
                int(fit["candidate_radial_quadrature_samples"]),
            ) * (10.0 ** float(candidate_log_amplitude))
            all_parameter_replays_exact &= bool(np.array_equal(candidate, replay))
            all_predictions_positive &= bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(np.isfinite(control))
                and np.all(control > 0.0)
            )
            candidate_confirmation_errors.append(candidate_confirmation_error)
            control_confirmation_errors.append(control_confirmation_error)
            material_candidate_errors.append(candidate_confirmation_error)
            material_control_errors.append(control_confirmation_error)
            rows.append(
                {
                    "profile_id": profile.identity(),
                    "material_id": material_id,
                    "diffuse_density": profile.diffuse_density,
                    "fixed_disc_radius_micrometres": radius_um,
                    "coverage_probability": float(coverage),
                    "candidate_log10_amplitude": float(candidate_log_amplitude),
                    "gaussian_scale_lines_per_mm": control_scale,
                    "gaussian_log10_amplitude": control_log_amplitude,
                    "candidate_build_log10_rmse": candidate_build_error,
                    "candidate_confirmation_log10_rmse": candidate_confirmation_error,
                    "gaussian_build_log10_rmse": control_build_error,
                    "gaussian_confirmation_log10_rmse": control_confirmation_error,
                    "confirmation_relative_improvement": (
                        (control_confirmation_error - candidate_confirmation_error)
                        / control_confirmation_error
                    ),
                }
            )
        candidate_median = float(np.median(material_candidate_errors))
        control_median = float(np.median(material_control_errors))
        improvement = (control_median - candidate_median) / control_median
        material_improvements.append(improvement)
        material_rows.append(
            {
                "material_id": material_id,
                "profile_count": len(material_profiles),
                "shared_fixed_disc_radius_micrometres": radius_um,
                "coverage_probabilities_in_density_order": coverages.tolist(),
                "candidate_confirmation_median_log10_rmse": candidate_median,
                "gaussian_confirmation_median_log10_rmse": control_median,
                "confirmation_median_relative_improvement": improvement,
            }
        )

    candidate_median = float(np.median(candidate_confirmation_errors))
    control_median = float(np.median(control_confirmation_errors))
    candidate_worst = max(candidate_confirmation_errors)
    control_worst = max(control_confirmation_errors)
    evaluation = contract["evaluation"]
    gates = {
        "material_count": len(grouped) == evaluation["required_material_count"],
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "confirmation_median_improvement": (
            (control_median - candidate_median) / control_median
            >= evaluation[
                "minimum_confirmation_median_relative_improvement_over_gaussian"
            ]
        ),
        "each_material_confirmation_median_improvement": all(
            value
            >= evaluation[
                "minimum_each_material_confirmation_median_relative_improvement"
            ]
            for value in material_improvements
        ),
        "worst_confirmation_not_worse": (
            candidate_worst / control_worst
            <= evaluation[
                "maximum_candidate_to_control_worst_confirmation_error_ratio"
            ]
        ),
        "density_monotone_coverage": all_coverage_monotone,
        "interior_shared_radius": all_radius_interior,
        "finite_positive_predictions": all_predictions_positive,
        "parameter_prediction_replay_exact": all_parameter_replays_exact,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "primary_model_source_pdf_sha256": source[
            "article_low_resolution_pdf_sha256"
        ],
        "material_count": len(grouped),
        "profile_count": len(profiles),
        "frequency_count": int(frequency.size),
        "build_frequency_count": int(np.count_nonzero(build_mask)),
        "confirmation_frequency_count": int(np.count_nonzero(confirmation_mask)),
        "candidate_confirmation_median_log10_rmse": candidate_median,
        "gaussian_confirmation_median_log10_rmse": control_median,
        "confirmation_median_relative_improvement": (
            (control_median - candidate_median) / control_median
        ),
        "candidate_confirmation_worst_log10_rmse": candidate_worst,
        "gaussian_confirmation_worst_log10_rmse": control_worst,
        "candidate_to_gaussian_worst_error_ratio": candidate_worst / control_worst,
        "materials": material_rows,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_fixed_disc_boolean_measured_nps_equation_family"
            if automatic_pass
            else "close_fixed_disc_boolean_measured_nps_equation_family"
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
    payload = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FixedDiscBooleanNPSCompatibilityError",
    "evaluate_fixed_disc_boolean_nps_compatibility",
    "load_contract",
    "write_report",
]
