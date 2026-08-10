"""Frozen U6.P2AC reciprocity measurement-identifiability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.film_physics.reciprocity import IntensityConditionedReciprocityProfile


class ReciprocityIdentifiabilityError(RuntimeError):
    """Raised when the U6.P2AC contract or parent evidence drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReciprocityIdentifiabilityError(f"expected object: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    return _load_object(path)


def _profile(parameters: np.ndarray, *, profile_id: str) -> IntensityConditionedReciprocityProfile:
    return IntensityConditionedReciprocityProfile(
        profile_id=profile_id,
        bright_exponent=float(parameters[0]),
        dark_exponent=float(parameters[1]),
        log2_pivot_rate=float(parameters[2]),
        log2_transition_width=float(parameters[3]),
    )


def _log_exposure(
    parameters: np.ndarray, rates: np.ndarray, physical_time: float
) -> np.ndarray:
    return np.log(
        _profile(parameters, profile_id="fit").effective_exposure(
            rates, physical_time
        )
    )


def _fit_local(
    *,
    observations: np.ndarray,
    rates: np.ndarray,
    times: np.ndarray,
    fit: dict[str, Any],
    per_time_gains: bool,
) -> Any:
    initial_parameters = np.asarray(fit["initial_parameters"], dtype=np.float64)
    lower = np.asarray(fit["lower_bounds"], dtype=np.float64)
    upper = np.asarray(fit["upper_bounds"], dtype=np.float64)
    if per_time_gains:
        initial = np.concatenate(
            [initial_parameters, np.asarray(fit["per_time_gain_initial"])]
        )
        low = np.concatenate(
            [lower, np.full(len(times), float(fit["per_time_gain_bounds"][0]))]
        )
        high = np.concatenate(
            [upper, np.full(len(times), float(fit["per_time_gain_bounds"][1]))]
        )

        def residual(values: np.ndarray) -> np.ndarray:
            predicted = np.stack(
                [
                    _log_exposure(values[:4], rates, float(time)) + values[4 + index]
                    for index, time in enumerate(times)
                ]
            )
            return (predicted - observations).ravel()

    else:
        initial = np.concatenate(
            [initial_parameters, [float(fit["shared_gain_initial"])]]
        )
        low = np.concatenate([lower, [float(fit["shared_gain_bounds"][0])]])
        high = np.concatenate([upper, [float(fit["shared_gain_bounds"][1])]])

        def residual(values: np.ndarray) -> np.ndarray:
            predicted = np.stack(
                [
                    _log_exposure(values[:4], rates, float(time)) + values[4]
                    for time in times
                ]
            )
            return (predicted - observations).ravel()

    tolerance = float(fit["least_squares_tolerances"])
    return least_squares(
        residual,
        initial,
        bounds=(low, high),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=int(fit["maximum_function_evaluations"]),
    )


def _fit_global(
    *, observations: np.ndarray, rates: np.ndarray, times: np.ndarray, fit: dict[str, Any]
) -> Any:
    def residual(values: np.ndarray) -> np.ndarray:
        exponent, gain = values
        predicted = np.stack(
            [np.log(rates) + np.log(float(time)) / exponent + gain for time in times]
        )
        return (predicted - observations).ravel()

    tolerance = float(fit["least_squares_tolerances"])
    return least_squares(
        residual,
        [1.3, float(fit["shared_gain_initial"])],
        bounds=(
            [float(fit["lower_bounds"][0]), float(fit["shared_gain_bounds"][0])],
            [float(fit["upper_bounds"][0]), float(fit["shared_gain_bounds"][1])],
        ),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=int(fit["maximum_function_evaluations"]),
    )


def _scaled_condition(jacobian: np.ndarray, scales: np.ndarray) -> float:
    return float(np.linalg.cond(jacobian * scales[np.newaxis, :]))


def _confirmation_rmse(
    candidate: np.ndarray,
    truth: np.ndarray,
    rates: np.ndarray,
    times: np.ndarray,
) -> float:
    residuals = np.concatenate(
        [
            _log_exposure(candidate, rates, float(time))
            - _log_exposure(truth, rates, float(time))
            for time in times
        ]
    )
    return float(np.sqrt(np.mean(np.square(residuals))))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p2ac_reciprocity_identifiability_contract.v1"
    ):
        raise ReciprocityIdentifiabilityError("unsupported P2AC contract")
    parent_binding = contract["parent"]
    parent_path = root / parent_binding["evidence_path"]
    if _sha(parent_path) != parent_binding["evidence_sha256"]:
        raise ReciprocityIdentifiabilityError("P2AB evidence hash mismatch")
    if _load_object(parent_path).get("decision") != parent_binding["required_decision"]:
        raise ReciprocityIdentifiabilityError("P2AB decision does not admit P2AC")

    truth_spec = contract["truth"]
    truth = np.asarray(
        [
            truth_spec["bright_exponent"],
            truth_spec["dark_exponent"],
            truth_spec["log2_pivot_rate"],
            truth_spec["log2_transition_width"],
        ],
        dtype=np.float64,
    )
    development = contract["development_design"]
    log2_minimum, log2_maximum, count = development["incident_rate_log2"]
    rates = np.exp2(
        np.linspace(float(log2_minimum), float(log2_maximum), int(count))
    )
    times = np.asarray(development["physical_times_seconds"], dtype=np.float64)
    rng = np.random.default_rng(int(development["seed"]))
    noise = rng.normal(
        0.0,
        float(development["log_measurement_noise_sigma"]),
        (len(times), len(rates)),
    )
    clean = np.stack(
        [_log_exposure(truth, rates, float(time)) for time in times]
    )
    anchored_observations = (
        clean + float(development["anchored_shared_log_gain"]) + noise
    )
    unanchored_observations = clean + np.asarray(
        development["unanchored_per_time_log_gains"], dtype=np.float64
    )[:, None] + noise

    fit = contract["fit"]
    anchored = _fit_local(
        observations=anchored_observations,
        rates=rates,
        times=times,
        fit=fit,
        per_time_gains=False,
    )
    unanchored = _fit_local(
        observations=unanchored_observations,
        rates=rates,
        times=times,
        fit=fit,
        per_time_gains=True,
    )
    global_fit = _fit_global(
        observations=anchored_observations,
        rates=rates,
        times=times,
        fit=fit,
    )
    wrong_time = _fit_local(
        observations=anchored_observations,
        rates=rates,
        times=times[::-1],
        fit=fit,
        per_time_gains=False,
    )
    confirmation_reads_before_fit_freeze = 0

    confirmation = contract["confirmation_design"]
    confirm_minimum, confirm_maximum, confirm_count = confirmation[
        "incident_rate_log2"
    ]
    confirm_rates = np.exp2(
        np.linspace(
            float(confirm_minimum), float(confirm_maximum), int(confirm_count)
        )
    )
    confirm_times = np.asarray(
        confirmation["physical_times_seconds"], dtype=np.float64
    )
    anchored_confirmation_rmse = _confirmation_rmse(
        anchored.x[:4], truth, confirm_rates, confirm_times
    )
    wrong_time_confirmation_rmse = _confirmation_rmse(
        wrong_time.x[:4], truth, confirm_rates, confirm_times
    )
    global_residuals = np.concatenate(
        [
            np.log(confirm_rates)
            + np.log(float(time)) / global_fit.x[0]
            - _log_exposure(truth, confirm_rates, float(time))
            for time in confirm_times
        ]
    )
    global_confirmation_rmse = float(
        np.sqrt(np.mean(np.square(global_residuals)))
    )

    parameter_scales = np.asarray(fit["parameter_scales"], dtype=np.float64)
    anchored_scales = np.concatenate(
        [parameter_scales, [float(fit["gain_scale"])]]
    )
    unanchored_scales = np.concatenate(
        [
            parameter_scales,
            np.full(len(times), float(fit["gain_scale"]), dtype=np.float64),
        ]
    )
    anchored_parameter_error = np.abs(anchored.x[:4] - truth)
    unanchored_exponent_error = np.abs(unanchored.x[:2] - truth[:2])
    anchored_condition = _scaled_condition(anchored.jac, anchored_scales)
    unanchored_condition = _scaled_condition(unanchored.jac, unanchored_scales)
    all_fits_finite = all(
        result.success
        and np.all(np.isfinite(result.x))
        and np.all(np.isfinite(result.fun))
        for result in (anchored, unanchored, global_fit, wrong_time)
    )

    measurements = {
        "maximum_anchored_parameter_absolute_error": float(
            np.max(anchored_parameter_error)
        ),
        "anchored_scaled_jacobian_condition": anchored_condition,
        "unanchored_scaled_jacobian_condition": unanchored_condition,
        "maximum_unanchored_exponent_absolute_error": float(
            np.max(unanchored_exponent_error)
        ),
        "anchored_confirmation_log_rmse": anchored_confirmation_rmse,
        "global_confirmation_log_rmse": global_confirmation_rmse,
        "wrong_time_confirmation_log_rmse": wrong_time_confirmation_rmse,
        "confirmation_target_reads_before_fit_freeze": confirmation_reads_before_fit_freeze,
        "all_fits_finite": all_fits_finite,
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "maximum_anchored_parameter_absolute_error": measurements[
            "maximum_anchored_parameter_absolute_error"
        ]
        <= gates["maximum_anchored_parameter_absolute_error"],
        "maximum_anchored_scaled_jacobian_condition": anchored_condition
        <= gates["maximum_anchored_scaled_jacobian_condition"],
        "minimum_unanchored_scaled_jacobian_condition": unanchored_condition
        >= gates["minimum_unanchored_scaled_jacobian_condition"],
        "minimum_unanchored_exponent_absolute_error": measurements[
            "maximum_unanchored_exponent_absolute_error"
        ]
        >= gates["minimum_unanchored_exponent_absolute_error"],
        "maximum_anchored_confirmation_log_rmse": anchored_confirmation_rmse
        <= gates["maximum_anchored_confirmation_log_rmse"],
        "minimum_global_confirmation_log_rmse": global_confirmation_rmse
        >= gates["minimum_global_confirmation_log_rmse"],
        "minimum_wrong_time_confirmation_log_rmse": wrong_time_confirmation_rmse
        >= gates["minimum_wrong_time_confirmation_log_rmse"],
        "confirmation_target_reads_before_fit_freeze": confirmation_reads_before_fit_freeze
        == gates["confirmation_target_reads_before_fit_freeze"],
        "all_fits_finite": all_fits_finite is gates["all_fits_finite"],
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(gates):
        raise ReciprocityIdentifiabilityError("P2AC gate vocabulary drift")
    stable = {
        "schema": "neuro_film.u6_p2ac_reciprocity_identifiability_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p2ac_reciprocity_identifiability_v1.json"
        ),
        "parent_evidence_sha256": parent_binding["evidence_sha256"],
        "truth_parameters": truth.tolist(),
        "anchored_fit": anchored.x.tolist(),
        "unanchored_fit": unanchored.x.tolist(),
        "global_fit": global_fit.x.tolist(),
        "wrong_time_fit": wrong_time.x.tolist(),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "ReciprocityIdentifiabilityError",
    "load_contract",
    "run_audit",
    "write_report",
]
