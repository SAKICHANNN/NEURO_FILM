"""U6.P4BR held-band Thomas-cluster NPS compatibility evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from src.eval.fixed_disc_boolean_nps_compatibility import (
    _band_mask,
    _fit_gaussian_control,
    _fit_log_amplitude,
    _log10_rmse,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.thomas_cluster_nps import (
    thomas_cluster_gaussian_mark_nps,
)

SCHEMA = "neuro_film.u6_p4br_thomas_cluster_nps_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4br_thomas_cluster_nps_compatibility_report.v1"


class ThomasClusterNPSCompatibilityError(RuntimeError):
    """Raised when frozen P4BR evidence or semantics drift."""


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
        raise ThomasClusterNPSCompatibilityError("P4BR paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    fit = payload.get("fit", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("particle_sigma_micrometres_shared_per_material_bounds")
        != [0.1, 12.0]
        or model.get("cluster_sigma_micrometres_per_state_bounds") != [0.1, 30.0]
        or model.get("mean_offspring_per_cluster_per_state_bounds") != [0.01, 100.0]
        or model.get("density_interpolation_allowed") is not False
        or model.get("render_allowed") is not False
        or fit.get("build_bands_lines_per_mm") != [[5.0, 100.0], [255.0, 350.0]]
        or fit.get("confirmation_bands_lines_per_mm")
        != [[105.0, 250.0], [355.0, 500.0]]
        or fit.get("optimizer_seed") != 2608022701
        or fit.get("optimizer_max_iterations") != 160
        or fit.get("optimizer_population_size") != 15
        or fit.get(
            "post_score_scale_multiplicity_frequency_split_model_or_gate_tuning_allowed"
        )
        is not False
        or evaluation.get("required_material_count") != 2
        or evaluation.get("required_profile_count") != 5
        or evaluation.get(
            "minimum_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.25
        or evaluation.get(
            "minimum_each_material_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.1
        or evaluation.get(
            "maximum_candidate_to_gaussian_worst_confirmation_error_ratio"
        )
        != 1.0
        or evaluation.get("minimum_profiles_beating_gaussian") != 4
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ThomasClusterNPSCompatibilityError("P4BR frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ThomasClusterNPSCompatibilityError(
            f"P4BR parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_shape(
    profile: HistoricalBWNoiseSpectrumProfile,
    frequency: np.ndarray,
    particle_sigma_micrometres: float,
    cluster_sigma_micrometres: float,
    mean_offspring: float,
) -> np.ndarray:
    intrinsic = thomas_cluster_gaussian_mark_nps(
        frequency,
        particle_sigma_micrometres / 1000.0,
        cluster_sigma_micrometres / 1000.0,
        mean_offspring,
    )
    return intrinsic * np.square(profile.circular_aperture_mtf(frequency))


def _fit_material_candidate(
    profiles: Sequence[HistoricalBWNoiseSpectrumProfile],
    build_frequency: np.ndarray,
    build_targets: Sequence[np.ndarray],
    model: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    particle_bounds = tuple(
        math.log10(float(value))
        for value in model["particle_sigma_micrometres_shared_per_material_bounds"]
    )
    cluster_bounds = tuple(
        math.log10(float(value))
        for value in model["cluster_sigma_micrometres_per_state_bounds"]
    )
    offspring_bounds = tuple(
        math.log10(float(value))
        for value in model["mean_offspring_per_cluster_per_state_bounds"]
    )

    def objective(parameters: np.ndarray) -> float:
        particle_sigma = 10.0 ** float(parameters[0])
        errors = []
        for index, (profile, target) in enumerate(
            zip(profiles, build_targets, strict=True)
        ):
            cluster_sigma = 10.0 ** float(parameters[1 + 2 * index])
            offspring = 10.0 ** float(parameters[2 + 2 * index])
            shape = _candidate_shape(
                profile,
                build_frequency,
                particle_sigma,
                cluster_sigma,
                offspring,
            )
            amplitude = _fit_log_amplitude(shape, target)
            errors.append(_log10_rmse(shape * (10.0**amplitude), target) ** 2)
        return float(np.mean(errors))

    result = differential_evolution(
        objective,
        bounds=[
            particle_bounds,
            *chain.from_iterable((cluster_bounds, offspring_bounds) for _ in profiles),
        ],
        seed=int(fit["optimizer_seed"]),
        maxiter=int(fit["optimizer_max_iterations"]),
        popsize=int(fit["optimizer_population_size"]),
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-10,
        atol=1e-13,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ThomasClusterNPSCompatibilityError(
            f"P4BR candidate optimization failed: {result.message}"
        )
    particle_sigma = 10.0 ** float(result.x[0])
    cluster_sigmas = []
    offspring_counts = []
    amplitudes = []
    for index, (profile, target) in enumerate(
        zip(profiles, build_targets, strict=True)
    ):
        cluster_sigma = 10.0 ** float(result.x[1 + 2 * index])
        offspring = 10.0 ** float(result.x[2 + 2 * index])
        shape = _candidate_shape(
            profile,
            build_frequency,
            particle_sigma,
            cluster_sigma,
            offspring,
        )
        cluster_sigmas.append(cluster_sigma)
        offspring_counts.append(offspring)
        amplitudes.append(_fit_log_amplitude(shape, target))
    return (
        particle_sigma,
        np.asarray(cluster_sigmas),
        np.asarray(offspring_counts),
        np.asarray(amplitudes),
    )


def evaluate_thomas_cluster_nps_compatibility(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    bundle = _load_parent(parents, root, "p4bl_bundle")
    _load_parent(parents, root, "p4bq_contract")
    disc_decision = _load_parent(parents, root, "p4bq_decision")
    disc_report = _load_parent(parents, root, "p4bq_report")
    if (
        bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or disc_decision.get("decision")
        != "close_independent_disc_boolean_measured_nps_equation_family"
        or disc_report.get("stable_evidence_id") != parents["p4bq_stable_evidence_id"]
    ):
        raise ThomasClusterNPSCompatibilityError("P4BR parent decision mismatch")
    source = contract["primary_model_source"]
    source_path = _relative(root, str(source["official_vor_pdf_path"]))
    if (
        not source_path.is_file()
        or _hash_file(source_path) != source["official_vor_pdf_sha256"]
    ):
        raise ThomasClusterNPSCompatibilityError("P4BR primary source mismatch")
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
    grouped: dict[str, list[HistoricalBWNoiseSpectrumProfile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.material_id, []).append(profile)
    for material_profiles in grouped.values():
        material_profiles.sort(key=lambda item: item.diffuse_density)

    rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    control_errors: list[float] = []
    material_improvements = []
    all_positive = True
    all_replay = True
    all_particle_interior = True
    all_state_interior = True
    profiles_beating_control = 0
    margin_fraction = float(
        contract["evaluation"]["interior_parameter_margin_fraction"]
    )
    particle_bounds = model["particle_sigma_micrometres_shared_per_material_bounds"]
    cluster_bounds = model["cluster_sigma_micrometres_per_state_bounds"]
    offspring_bounds = model["mean_offspring_per_cluster_per_state_bounds"]
    for material_id in sorted(grouped):
        material_profiles = grouped[material_id]
        build_targets = [
            target_by_id[p.identity()][build_mask] for p in material_profiles
        ]
        particle_sigma, cluster_sigmas, offspring_counts, amplitudes = (
            _fit_material_candidate(
                material_profiles,
                frequency[build_mask],
                build_targets,
                model,
                fit,
            )
        )
        particle_margin = (particle_bounds[1] - particle_bounds[0]) * margin_fraction
        all_particle_interior &= (
            particle_bounds[0] + particle_margin
            < particle_sigma
            < particle_bounds[1] - particle_margin
        )
        material_candidate = []
        material_control = []
        for profile, cluster_sigma, offspring, amplitude in zip(
            material_profiles,
            cluster_sigmas,
            offspring_counts,
            amplitudes,
            strict=True,
        ):
            cluster_margin = (cluster_bounds[1] - cluster_bounds[0]) * margin_fraction
            offspring_log_margin = (
                math.log10(offspring_bounds[1]) - math.log10(offspring_bounds[0])
            ) * margin_fraction
            all_state_interior &= (
                cluster_bounds[0] + cluster_margin
                < cluster_sigma
                < cluster_bounds[1] - cluster_margin
                and math.log10(offspring_bounds[0]) + offspring_log_margin
                < math.log10(offspring)
                < math.log10(offspring_bounds[1]) - offspring_log_margin
            )
            target = target_by_id[profile.identity()]
            candidate_shape = _candidate_shape(
                profile,
                frequency,
                particle_sigma,
                float(cluster_sigma),
                float(offspring),
            )
            candidate = candidate_shape * (10.0 ** float(amplitude))
            control_scale, control_amplitude = _fit_gaussian_control(
                profile,
                frequency[build_mask],
                target[build_mask],
                [5.0, 1000.0],
            )
            control = np.exp(
                -np.square(frequency) / (2.0 * control_scale * control_scale)
            ) * np.square(profile.circular_aperture_mtf(frequency))
            control *= 10.0**control_amplitude
            candidate_error = _log10_rmse(
                candidate[confirmation_mask], target[confirmation_mask]
            )
            control_error = _log10_rmse(
                control[confirmation_mask], target[confirmation_mask]
            )
            replay = _candidate_shape(
                profile,
                frequency,
                particle_sigma,
                float(cluster_sigma),
                float(offspring),
            ) * (10.0 ** float(amplitude))
            all_replay &= bool(np.array_equal(candidate, replay))
            all_positive &= bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(np.isfinite(control))
                and np.all(control > 0.0)
            )
            profiles_beating_control += int(candidate_error < control_error)
            candidate_errors.append(candidate_error)
            control_errors.append(control_error)
            material_candidate.append(candidate_error)
            material_control.append(control_error)
            rows.append(
                {
                    "profile_id": profile.identity(),
                    "material_id": material_id,
                    "diffuse_density": profile.diffuse_density,
                    "particle_sigma_micrometres": particle_sigma,
                    "cluster_sigma_micrometres": float(cluster_sigma),
                    "mean_offspring_per_cluster": float(offspring),
                    "candidate_log10_amplitude": float(amplitude),
                    "candidate_build_log10_rmse": _log10_rmse(
                        candidate[build_mask], target[build_mask]
                    ),
                    "candidate_confirmation_log10_rmse": candidate_error,
                    "gaussian_confirmation_log10_rmse": control_error,
                    "confirmation_relative_improvement": (
                        (control_error - candidate_error) / control_error
                    ),
                }
            )
        candidate_material_median = float(np.median(material_candidate))
        control_material_median = float(np.median(material_control))
        improvement = (
            control_material_median - candidate_material_median
        ) / control_material_median
        material_improvements.append(improvement)
        material_rows.append(
            {
                "material_id": material_id,
                "profile_count": len(material_profiles),
                "shared_particle_sigma_micrometres": particle_sigma,
                "cluster_sigmas_micrometres_in_density_order": cluster_sigmas.tolist(),
                "mean_offspring_counts_in_density_order": offspring_counts.tolist(),
                "candidate_confirmation_median_log10_rmse": candidate_material_median,
                "gaussian_confirmation_median_log10_rmse": control_material_median,
                "confirmation_median_relative_improvement": improvement,
            }
        )

    candidate_median = float(np.median(candidate_errors))
    control_median = float(np.median(control_errors))
    candidate_worst = max(candidate_errors)
    control_worst = max(control_errors)
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
                "minimum_each_material_confirmation_median_relative_improvement_over_gaussian"
            ]
            for value in material_improvements
        ),
        "worst_confirmation_not_worse": (
            candidate_worst / control_worst
            <= evaluation[
                "maximum_candidate_to_gaussian_worst_confirmation_error_ratio"
            ]
        ),
        "profiles_beating_gaussian": (
            profiles_beating_control >= evaluation["minimum_profiles_beating_gaussian"]
        ),
        "interior_particle_sigma": all_particle_interior,
        "interior_cluster_sigma_and_multiplicity": all_state_interior,
        "finite_positive_predictions": all_positive,
        "parameter_prediction_replay_exact": all_replay,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "parent_independent_disc_stable_evidence_id": parents[
            "p4bq_stable_evidence_id"
        ],
        "primary_model_source_pdf_sha256": source["official_vor_pdf_sha256"],
        "material_count": len(grouped),
        "profile_count": len(profiles),
        "candidate_confirmation_median_log10_rmse": candidate_median,
        "gaussian_confirmation_median_log10_rmse": control_median,
        "confirmation_median_relative_improvement": (
            (control_median - candidate_median) / control_median
        ),
        "candidate_confirmation_worst_log10_rmse": candidate_worst,
        "gaussian_confirmation_worst_log10_rmse": control_worst,
        "candidate_to_gaussian_worst_error_ratio": candidate_worst / control_worst,
        "profiles_beating_gaussian": profiles_beating_control,
        "materials": material_rows,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_thomas_cluster_historical_measured_nps_equation_family"
            if automatic_pass
            else "close_one_level_thomas_cluster_measured_nps_equation_family"
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
    "ThomasClusterNPSCompatibilityError",
    "evaluate_thomas_cluster_nps_compatibility",
    "load_contract",
    "write_report",
]
