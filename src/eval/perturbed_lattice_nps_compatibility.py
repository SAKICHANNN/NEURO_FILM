"""U6.P4CE held-band perturbed-lattice NPS compatibility evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
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
from src.eval.thomas_cluster_nps_compatibility import _optimizer_result_is_usable
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.perturbed_lattice_nps import (
    perturbed_lattice_gaussian_mark_nps,
)
from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps

SCHEMA = "neuro_film.u6_p4ce_perturbed_lattice_nps_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ce_perturbed_lattice_nps_compatibility_report.v1"


class PerturbedLatticeNPSCompatibilityError(RuntimeError):
    """Raised when frozen P4CE evidence or semantics drift."""


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
        raise PerturbedLatticeNPSCompatibilityError("P4CE paths must be relative")
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
        or model.get("jitter_sigma_micrometres_per_state_bounds") != [0.05, 50.0]
        or model.get("bragg_atoms_modeled") is not False
        or model.get("density_interpolation_allowed") is not False
        or model.get("render_allowed") is not False
        or fit.get("build_bands_lines_per_mm") != [[5.0, 100.0], [255.0, 350.0]]
        or fit.get("confirmation_bands_lines_per_mm")
        != [[105.0, 250.0], [355.0, 500.0]]
        or fit.get("optimizer_seed") != 2608023101
        or fit.get("optimizer_max_iterations") != 160
        or fit.get("optimizer_population_size") != 15
        or fit.get(
            "post_score_scale_jitter_frequency_split_model_or_gate_tuning_allowed"
        )
        is not False
        or evaluation.get("required_material_count") != 2
        or evaluation.get("required_profile_count") != 5
        or evaluation.get(
            "minimum_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.1
        or evaluation.get(
            "minimum_confirmation_median_relative_improvement_over_thomas"
        )
        != 0.1
        or evaluation.get(
            "minimum_each_material_confirmation_median_relative_improvement_over_gaussian"
        )
        != 0.0
        or evaluation.get(
            "maximum_candidate_to_gaussian_worst_confirmation_error_ratio"
        )
        != 1.0
        or evaluation.get("minimum_profiles_beating_gaussian") != 4
        or evaluation.get("minimum_profiles_beating_thomas") != 4
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise PerturbedLatticeNPSCompatibilityError("P4CE frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise PerturbedLatticeNPSCompatibilityError(
            f"P4CE parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_shape(
    profile: HistoricalBWNoiseSpectrumProfile,
    frequency: np.ndarray,
    particle_sigma_micrometres: float,
    jitter_sigma_micrometres: float,
) -> np.ndarray:
    intrinsic = perturbed_lattice_gaussian_mark_nps(
        frequency,
        particle_sigma_micrometres / 1000.0,
        jitter_sigma_micrometres / 1000.0,
    )
    return intrinsic * np.square(profile.circular_aperture_mtf(frequency))


def _fit_material_candidate(
    profiles: Sequence[HistoricalBWNoiseSpectrumProfile],
    build_frequency: np.ndarray,
    build_targets: Sequence[np.ndarray],
    model: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    particle_bounds = tuple(
        math.log10(float(value))
        for value in model["particle_sigma_micrometres_shared_per_material_bounds"]
    )
    jitter_bounds = tuple(
        math.log10(float(value))
        for value in model["jitter_sigma_micrometres_per_state_bounds"]
    )

    def objective(parameters: np.ndarray) -> float:
        particle_sigma = 10.0 ** float(parameters[0])
        errors = []
        for index, (profile, target) in enumerate(
            zip(profiles, build_targets, strict=True)
        ):
            jitter_sigma = 10.0 ** float(parameters[1 + index])
            shape = _candidate_shape(
                profile, build_frequency, particle_sigma, jitter_sigma
            )
            amplitude = _fit_log_amplitude(shape, target)
            errors.append(_log10_rmse(shape * (10.0**amplitude), target) ** 2)
        return float(np.mean(errors))

    result = differential_evolution(
        objective,
        bounds=[particle_bounds, *([jitter_bounds] * len(profiles))],
        seed=int(fit["optimizer_seed"]),
        maxiter=int(fit["optimizer_max_iterations"]),
        popsize=int(fit["optimizer_population_size"]),
        polish=True,
        workers=1,
        updating="immediate",
        tol=1e-10,
        atol=1e-13,
    )
    if not _optimizer_result_is_usable(result, int(fit["optimizer_max_iterations"])):
        raise PerturbedLatticeNPSCompatibilityError(
            f"P4CE candidate optimization failed: {result.message}"
        )
    particle_sigma = 10.0 ** float(result.x[0])
    jitter_sigmas = 10.0 ** np.asarray(result.x[1:], dtype=np.float64)
    amplitudes = []
    for profile, target, jitter_sigma in zip(
        profiles, build_targets, jitter_sigmas, strict=True
    ):
        amplitudes.append(
            _fit_log_amplitude(
                _candidate_shape(
                    profile, build_frequency, particle_sigma, float(jitter_sigma)
                ),
                target,
            )
        )
    return particle_sigma, jitter_sigmas, np.asarray(amplitudes)


def _thomas_prediction(
    profile: HistoricalBWNoiseSpectrumProfile,
    frequency: np.ndarray,
    row: Mapping[str, Any],
) -> np.ndarray:
    intrinsic = thomas_cluster_gaussian_mark_nps(
        frequency,
        float(row["particle_sigma_micrometres"]) / 1000.0,
        float(row["cluster_sigma_micrometres"]) / 1000.0,
        float(row["mean_offspring_per_cluster"]),
    )
    return (
        intrinsic
        * np.square(profile.circular_aperture_mtf(frequency))
        * (10.0 ** float(row["candidate_log10_amplitude"]))
    )


def evaluate_perturbed_lattice_nps_compatibility(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    bundle = _load_parent(parents, root, "p4bl_bundle")
    thomas_report = _load_parent(parents, root, "p4br_report")
    if (
        bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or thomas_report.get("stable_evidence_id")
        != parents["p4br_stable_evidence_id"]
    ):
        raise PerturbedLatticeNPSCompatibilityError("P4CE parent identity mismatch")
    source = contract["primary_model_source"]
    source_path = _relative(root, str(source["paper_path"]))
    if not source_path.is_file() or _hash_file(source_path) != source["paper_sha256"]:
        raise PerturbedLatticeNPSCompatibilityError("P4CE primary source mismatch")

    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    thomas_by_profile = {row["profile_id"]: row for row in thomas_report["rows"]}
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
    gaussian_errors: list[float] = []
    thomas_errors: list[float] = []
    material_gaussian_improvements: list[float] = []
    profiles_beating_gaussian = 0
    profiles_beating_thomas = 0
    all_positive = True
    all_replay = True
    all_interior = True
    margin_fraction = float(contract["evaluation"]["interior_parameter_margin_fraction"])
    particle_bounds = model["particle_sigma_micrometres_shared_per_material_bounds"]
    jitter_bounds = model["jitter_sigma_micrometres_per_state_bounds"]
    particle_margin = (particle_bounds[1] - particle_bounds[0]) * margin_fraction
    jitter_log_margin = (
        math.log10(jitter_bounds[1]) - math.log10(jitter_bounds[0])
    ) * margin_fraction

    for material_id in sorted(grouped):
        material_profiles = grouped[material_id]
        build_targets = [
            target_by_id[profile.identity()][build_mask]
            for profile in material_profiles
        ]
        particle_sigma, jitter_sigmas, amplitudes = _fit_material_candidate(
            material_profiles,
            frequency[build_mask],
            build_targets,
            model,
            fit,
        )
        all_interior &= (
            particle_bounds[0] + particle_margin
            < particle_sigma
            < particle_bounds[1] - particle_margin
        )
        material_candidate: list[float] = []
        material_gaussian: list[float] = []
        for profile, jitter_sigma, amplitude in zip(
            material_profiles, jitter_sigmas, amplitudes, strict=True
        ):
            all_interior &= (
                math.log10(jitter_bounds[0]) + jitter_log_margin
                < math.log10(float(jitter_sigma))
                < math.log10(jitter_bounds[1]) - jitter_log_margin
            )
            target = target_by_id[profile.identity()]
            candidate = _candidate_shape(
                profile, frequency, particle_sigma, float(jitter_sigma)
            ) * (10.0 ** float(amplitude))
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
            thomas = _thomas_prediction(
                profile, frequency, thomas_by_profile[profile.identity()]
            )
            candidate_error = _log10_rmse(
                candidate[confirmation_mask], target[confirmation_mask]
            )
            gaussian_error = _log10_rmse(
                gaussian[confirmation_mask], target[confirmation_mask]
            )
            thomas_error = _log10_rmse(
                thomas[confirmation_mask], target[confirmation_mask]
            )
            replay = _candidate_shape(
                profile, frequency, particle_sigma, float(jitter_sigma)
            ) * (10.0 ** float(amplitude))
            all_replay &= bool(np.array_equal(candidate, replay))
            all_positive &= bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and np.all(np.isfinite(gaussian))
                and np.all(gaussian > 0.0)
                and np.all(np.isfinite(thomas))
                and np.all(thomas > 0.0)
            )
            profiles_beating_gaussian += int(candidate_error < gaussian_error)
            profiles_beating_thomas += int(candidate_error < thomas_error)
            candidate_errors.append(candidate_error)
            gaussian_errors.append(gaussian_error)
            thomas_errors.append(thomas_error)
            material_candidate.append(candidate_error)
            material_gaussian.append(gaussian_error)
            rows.append(
                {
                    "profile_id": profile.identity(),
                    "material_id": material_id,
                    "diffuse_density": profile.diffuse_density,
                    "particle_sigma_micrometres": particle_sigma,
                    "jitter_sigma_micrometres": float(jitter_sigma),
                    "candidate_log10_amplitude": float(amplitude),
                    "candidate_build_log10_rmse": _log10_rmse(
                        candidate[build_mask], target[build_mask]
                    ),
                    "candidate_confirmation_log10_rmse": candidate_error,
                    "gaussian_confirmation_log10_rmse": gaussian_error,
                    "thomas_confirmation_log10_rmse": thomas_error,
                    "relative_improvement_over_gaussian": (
                        (gaussian_error - candidate_error) / gaussian_error
                    ),
                    "relative_improvement_over_thomas": (
                        (thomas_error - candidate_error) / thomas_error
                    ),
                }
            )
        candidate_median = float(np.median(material_candidate))
        gaussian_median = float(np.median(material_gaussian))
        improvement = (gaussian_median - candidate_median) / gaussian_median
        material_gaussian_improvements.append(improvement)
        material_rows.append(
            {
                "material_id": material_id,
                "profile_count": len(material_profiles),
                "shared_particle_sigma_micrometres": particle_sigma,
                "jitter_sigmas_micrometres_in_density_order": jitter_sigmas.tolist(),
                "candidate_confirmation_median_log10_rmse": candidate_median,
                "gaussian_confirmation_median_log10_rmse": gaussian_median,
                "relative_improvement_over_gaussian": improvement,
            }
        )

    candidate_median = float(np.median(candidate_errors))
    gaussian_median = float(np.median(gaussian_errors))
    thomas_median = float(np.median(thomas_errors))
    candidate_worst = max(candidate_errors)
    gaussian_worst = max(gaussian_errors)
    improvement_gaussian = (gaussian_median - candidate_median) / gaussian_median
    improvement_thomas = (thomas_median - candidate_median) / thomas_median
    evaluation = contract["evaluation"]
    gates = {
        "material_count": len(grouped) == evaluation["required_material_count"],
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "confirmation_median_improvement_over_gaussian": (
            improvement_gaussian
            >= evaluation[
                "minimum_confirmation_median_relative_improvement_over_gaussian"
            ]
        ),
        "confirmation_median_improvement_over_thomas": (
            improvement_thomas
            >= evaluation[
                "minimum_confirmation_median_relative_improvement_over_thomas"
            ]
        ),
        "each_material_not_worse_than_gaussian": all(
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
        "profiles_beating_gaussian": (
            profiles_beating_gaussian >= evaluation["minimum_profiles_beating_gaussian"]
        ),
        "profiles_beating_thomas": (
            profiles_beating_thomas >= evaluation["minimum_profiles_beating_thomas"]
        ),
        "interior_particle_and_jitter_sigmas": all_interior,
        "finite_positive_predictions": all_positive,
        "parameter_prediction_replay_exact": all_replay,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "parent_thomas_stable_evidence_id": parents["p4br_stable_evidence_id"],
        "primary_model_source_pdf_sha256": source["paper_sha256"],
        "bragg_atoms_modeled": False,
        "material_count": len(grouped),
        "profile_count": len(profiles),
        "candidate_confirmation_median_log10_rmse": candidate_median,
        "gaussian_confirmation_median_log10_rmse": gaussian_median,
        "thomas_confirmation_median_log10_rmse": thomas_median,
        "confirmation_median_relative_improvement_over_gaussian": improvement_gaussian,
        "confirmation_median_relative_improvement_over_thomas": improvement_thomas,
        "candidate_confirmation_worst_log10_rmse": candidate_worst,
        "gaussian_confirmation_worst_log10_rmse": gaussian_worst,
        "candidate_to_gaussian_worst_error_ratio": candidate_worst / gaussian_worst,
        "profiles_beating_gaussian": profiles_beating_gaussian,
        "profiles_beating_thomas": profiles_beating_thomas,
        "materials": material_rows,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_perturbed_lattice_diffuse_historical_nps_baseline"
            if automatic_pass
            else "close_independent_perturbed_lattice_diffuse_nps_family"
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
    "PerturbedLatticeNPSCompatibilityError",
    "evaluate_perturbed_lattice_nps_compatibility",
    "load_contract",
    "write_report",
]
