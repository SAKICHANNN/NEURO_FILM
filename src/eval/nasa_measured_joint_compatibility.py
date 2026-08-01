"""U6.P4AU measured compatibility for one shared cloud MTF/noise scale."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j1

from src.eval.physical_callier_source import hash_file

SCHEMA = "neuro_film.u6_p4au_nasa_measured_joint_compatibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4au_nasa_measured_joint_compatibility_report.v1"


class MeasuredJointCompatibilityError(RuntimeError):
    """Raised when the measured compatibility contract or source drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredJointCompatibilityError("P4AU paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source_report", {})
    fit = payload.get("fit", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("evidence_role")
        != "development-only measured compatibility because every source table was observed during P4AT"
        or source.get("expected_sha256")
        != "53748b3eeb92fa3a9b103db50ea32f1bb2edba4278ae72e2a9b74f8aa2fd447b"
        or source.get("expected_stable_evidence_id")
        != "656897fd4816a46ee184736dd626aaf618e21d53acb8b2397a25d2db6c90d0c8"
        or sorted(payload.get("film_type_mapping", {})) != ["1", "2", "3"]
        or fit.get("integration_samples") != 50_001
        or fit.get("frequency_grid") != "zero plus geometric 1e-5..5000 cycles/mm"
        or fit.get("sigma_bounds_um") != [0.05, 50.0]
        or fit.get("optimizer") != "bounded scalar minimization with xatol 1e-10"
        or fit.get("objective")
        != "mean squared log RMS residual across all available aperture cells"
        or fit.get("fit_reads_mtf") is not False
        or evaluation.get("frequencies_cycles_per_mm") != [2, 4, 6, 8, 10, 12, 14, 16]
        or evaluation.get("mtf_values_are_never_fit") is not True
        or gates.get("maximum_fit_log_rmse") != 0.15
        or gates.get("maximum_candidate_residual") != 1.05
        or gates.get("minimum_dispersion_improvement_vs_no_diffusion") != 0.05
        or gates.get("minimum_dispersion_improvement_vs_wrong_film") != 0.05
        or not gates.get("all_scales_positive_finite")
        or not gates.get("all_predictions_finite")
        or not gates.get("source_count_contradiction_retained")
    ):
        raise MeasuredJointCompatibilityError("P4AU frozen contract drift")
    _relative_path(source.get("path", ""))
    return payload


def _load_source_report(config: dict[str, Any], root: Path) -> dict[str, Any]:
    locked = config["source_report"]
    path = root / _relative_path(locked["path"])
    if not path.is_file() or hash_file(path) != locked["expected_sha256"]:
        raise MeasuredJointCompatibilityError("P4AU source report drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema")
        != "neuro_film.u6_p4at_nasa_joint_mtf_granularity_source_report.v1"
        or report.get("stable_evidence_id") != locked["expected_stable_evidence_id"]
        or not report.get("source_pass")
        or not report.get("printed_prose_series_count_contradiction")
    ):
        raise MeasuredJointCompatibilityError("P4AU source facts drift")
    return report


def _frequency_grid(samples: int) -> np.ndarray:
    return np.concatenate(
        (np.zeros(1, dtype=np.float64), np.geomspace(1e-5, 5000.0, samples - 1))
    )


def _aperture_power(grid: np.ndarray, diameter_um: float) -> np.ndarray:
    argument = math.pi * (diameter_um / 1000.0) * grid
    response = np.ones_like(grid)
    nonzero = argument != 0.0
    response[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
    return response * response


def _variance_shapes(
    sigma_um: float,
    grid: np.ndarray,
    aperture_power: dict[int, np.ndarray],
) -> dict[int, float]:
    sigma_mm = sigma_um / 1000.0
    gaussian_power = np.exp(-4.0 * math.pi**2 * sigma_mm**2 * grid**2)
    return {
        diameter: float(
            2.0 * math.pi * np.trapezoid(grid * gaussian_power * power, x=grid)
        )
        for diameter, power in aperture_power.items()
    }


def _fit_film_scale(
    rows: list[dict[str, Any]],
    grid: np.ndarray,
    aperture_power: dict[int, np.ndarray],
    bounds: list[float],
) -> dict[str, Any]:
    def score(sigma_um: float) -> float:
        shapes = _variance_shapes(sigma_um, grid, aperture_power)
        squared: list[float] = []
        for row in rows:
            observations = {
                int(diameter): float(value)
                for diameter, value in row["rms_granularity_by_aperture_um"].items()
            }
            log_offsets = [
                math.log(value) - 0.5 * math.log(shapes[diameter])
                for diameter, value in observations.items()
            ]
            log_amplitude = float(np.mean(log_offsets))
            squared.extend(
                (math.log(value) - (log_amplitude + 0.5 * math.log(shapes[diameter])))
                ** 2
                for diameter, value in observations.items()
            )
        return float(np.mean(squared))

    result = minimize_scalar(
        score,
        bounds=(float(bounds[0]), float(bounds[1])),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise MeasuredJointCompatibilityError("P4AU bounded scale fit failed")
    sigma_um = float(result.x)
    shapes = _variance_shapes(sigma_um, grid, aperture_power)
    amplitudes: list[dict[str, float]] = []
    residuals: list[float] = []
    for row in rows:
        observations = {
            int(diameter): float(value)
            for diameter, value in row["rms_granularity_by_aperture_um"].items()
        }
        log_amplitude = float(
            np.mean(
                [
                    math.log(value) - 0.5 * math.log(shapes[diameter])
                    for diameter, value in observations.items()
                ]
            )
        )
        amplitudes.append(
            {"density": float(row["density"]), "log_amplitude": log_amplitude}
        )
        residuals.extend(
            math.log(value) - (log_amplitude + 0.5 * math.log(shapes[diameter]))
            for diameter, value in observations.items()
        )
    return {
        "sigma_um": sigma_um,
        "fit_log_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "density_amplitudes": amplitudes,
        "aperture_variance_shapes": {str(key): value for key, value in shapes.items()},
    }


def _film_code(exposure_id: str) -> str:
    parts = exposure_id.split("-")
    if len(parts) != 3 or parts[1] not in {"1", "2", "3"}:
        raise MeasuredJointCompatibilityError("P4AU exposure identity drift")
    return parts[1]


def _gaussian_mtf(sigma_um: float, frequency: int) -> float:
    sigma_mm = sigma_um / 1000.0
    return math.exp(-2.0 * math.pi**2 * sigma_mm**2 * frequency**2)


def _dispersion(values_by_frequency: dict[int, list[float]]) -> float:
    frequency_terms: list[float] = []
    for frequency in sorted(values_by_frequency):
        values = np.asarray(values_by_frequency[frequency], dtype=np.float64)
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise MeasuredJointCompatibilityError("P4AU residual is non-positive")
        logs = np.log(values)
        frequency_terms.append(float(np.median(np.abs(logs - np.median(logs)))))
    return float(np.mean(frequency_terms))


def evaluate_measured_joint_compatibility(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    source = _load_source_report(config, root)
    grid = _frequency_grid(config["fit"]["integration_samples"])
    aperture_power = {
        diameter: _aperture_power(grid, diameter) for diameter in (13, 27, 57)
    }
    fits: dict[str, dict[str, Any]] = {}
    mapping = config["film_type_mapping"]
    granularity_rows = source["numeric_density_granularity_rows"]
    for code, facts in mapping.items():
        rows = [
            row
            for row in granularity_rows
            if row["table"] == facts["granularity_table"]
        ]
        fits[code] = {
            "film_name": facts["name"],
            "granularity_table": facts["granularity_table"],
            **_fit_film_scale(
                rows,
                grid,
                aperture_power,
                config["fit"]["sigma_bounds_um"],
            ),
        }

    wrong_code = {"1": "2", "2": "3", "3": "1"}
    frequencies = set(config["evaluation"]["frequencies_cycles_per_mm"])
    candidate: dict[int, list[float]] = {frequency: [] for frequency in frequencies}
    no_diffusion: dict[int, list[float]] = {frequency: [] for frequency in frequencies}
    wrong_film: dict[int, list[float]] = {frequency: [] for frequency in frequencies}
    for block in source["mtf_blocks"]:
        code = _film_code(block["exposure_id"])
        sigma = fits[code]["sigma_um"]
        wrong_sigma = fits[wrong_code[code]]["sigma_um"]
        for row in block["rows"]:
            frequency = row["frequency_cycles_per_mm"]
            if frequency not in frequencies:
                continue
            film_mtf = _gaussian_mtf(sigma, frequency)
            wrong_mtf = _gaussian_mtf(wrong_sigma, frequency)
            for value in row["mtf_by_edge"].values():
                measured = float(value)
                candidate[frequency].append(measured / film_mtf)
                no_diffusion[frequency].append(measured)
                wrong_film[frequency].append(measured / wrong_mtf)

    candidate_dispersion = _dispersion(candidate)
    no_diffusion_dispersion = _dispersion(no_diffusion)
    wrong_film_dispersion = _dispersion(wrong_film)
    improvement_no = (
        no_diffusion_dispersion - candidate_dispersion
    ) / no_diffusion_dispersion
    improvement_wrong = (
        wrong_film_dispersion - candidate_dispersion
    ) / wrong_film_dispersion
    candidate_values = [value for values in candidate.values() for value in values]
    metrics = {
        "maximum_fit_log_rmse": max(fit["fit_log_rmse"] for fit in fits.values()),
        "candidate_residual_dispersion": candidate_dispersion,
        "no_diffusion_residual_dispersion": no_diffusion_dispersion,
        "wrong_film_residual_dispersion": wrong_film_dispersion,
        "dispersion_improvement_vs_no_diffusion": improvement_no,
        "dispersion_improvement_vs_wrong_film": improvement_wrong,
        "maximum_candidate_residual": max(candidate_values),
    }
    gates = config["gates"]
    gate_results = {
        "fit_log_rmse": metrics["maximum_fit_log_rmse"]
        <= gates["maximum_fit_log_rmse"],
        "candidate_residual": metrics["maximum_candidate_residual"]
        <= gates["maximum_candidate_residual"],
        "improvement_vs_no_diffusion": improvement_no
        >= gates["minimum_dispersion_improvement_vs_no_diffusion"],
        "improvement_vs_wrong_film": improvement_wrong
        >= gates["minimum_dispersion_improvement_vs_wrong_film"],
        "positive_finite_scales": all(
            math.isfinite(fit["sigma_um"]) and fit["sigma_um"] > 0.0
            for fit in fits.values()
        ),
        "finite_predictions": all(
            math.isfinite(value)
            for values in (candidate, no_diffusion, wrong_film)
            for series in values.values()
            for value in series
        ),
        "source_count_contradiction_retained": source[
            "printed_prose_series_count_contradiction"
        ],
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source_report_sha256": config["source_report"]["expected_sha256"],
        "source_stable_evidence_id": source["stable_evidence_id"],
        "evidence_role": config["evidence_role"],
        "fits": fits,
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_measured_development_compatibility"
            if automatic_pass
            else "close_exact_gaussian_shared_scale_measured_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredJointCompatibilityError",
    "evaluate_measured_joint_compatibility",
    "load_contract",
]
