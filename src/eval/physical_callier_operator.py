"""Synthetic evaluation for the U6.P6N generic Callier primitive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.callier import (
    CallierProfile,
    apply_callier_components,
    callier_q_factor,
    invert_callier_components,
)


SCHEMA = "neuro_film.u6_p6n_generic_callier_operator_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6n_generic_callier_operator_report.v1"


class CallierOperatorError(RuntimeError):
    """Raised when the frozen P6N contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload.get("operator", {}).get("profile", {})
    evaluation = payload.get("synthetic_evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or profile
        != {
            "profile_id": "generic-callier-synthetic-v1",
            "peak_diffuse_density": 1.4,
            "peak_q_increment": 0.35,
            "wavelength_weights_rgb": [0.55, 0.75, 1.0],
            "maximum_component_density": 4.05,
        }
        or evaluation.get("density_grid_samples") != 4097
        or evaluation.get("density_grid_range") != [0.0, 4.05]
        or evaluation.get("collimation_values") != [0.0, 0.5, 1.0]
        or evaluation.get("inverse_iterations") != 64
        or evaluation.get("partition_rows") != [1, 7, 31]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_q_must_not_exceed") != 1.35
        or gates.get("maximum_peak_density_error") != 0.002
        or gates.get("inverse_maximum_absolute_error") != 1e-12
        or gates.get("full_vs_partition_maximum_absolute_error") != 0.0
        or not gates.get("invalid_domain_must_fail_before_output")
    ):
        raise CallierOperatorError("P6N frozen contract drift")
    return payload


def profile_from_contract(config: dict[str, Any]) -> CallierProfile:
    values = config["operator"]["profile"]
    return CallierProfile(
        profile_id=values["profile_id"],
        peak_diffuse_density=values["peak_diffuse_density"],
        peak_q_increment=values["peak_q_increment"],
        wavelength_weights_rgb=tuple(values["wavelength_weights_rgb"]),
        maximum_component_density=values["maximum_component_density"],
    )


def evaluate_operator(config: dict[str, Any]) -> dict[str, Any]:
    profile = profile_from_contract(config)
    evaluation = config["synthetic_evaluation"]
    gates = config["gates"]
    density = np.linspace(
        *evaluation["density_grid_range"],
        evaluation["density_grid_samples"],
        dtype=np.float64,
    )[:, None, None]
    density_rgb = np.broadcast_to(density, (density.shape[0], 1, 3)).copy()
    q_zero = callier_q_factor(density_rgb, profile, collimation=0.0)
    q_half = callier_q_factor(density_rgb, profile, collimation=0.5)
    q_full = callier_q_factor(density_rgb, profile, collimation=1.0)
    peak_indices = np.argmax(q_full[:, 0, :], axis=0)
    peak_densities = density[peak_indices, 0, 0]
    peak_q = q_full[peak_indices, 0, np.arange(3)]
    half_error = float(
        np.max(np.abs((q_half - 1.0) - 0.5 * (q_full - 1.0)))
    )
    directed_silver = density_rgb * q_full
    minimum_step = float(np.min(np.diff(directed_silver[:, 0, :], axis=0)))
    recovered = invert_callier_components(
        directed_silver,
        np.zeros_like(density_rgb),
        profile,
        collimation=1.0,
        iterations=evaluation["inverse_iterations"],
    )
    inverse_error = float(np.max(np.abs(recovered - density_rgb)))

    rng = np.random.default_rng(619)
    dye = rng.uniform(0.0, 1.8, size=(67, 29, 3)).astype(np.float64)
    silver = rng.uniform(0.0, 1.4, size=(67, 29, 3)).astype(np.float64)
    zeros = np.zeros_like(silver)
    dye_identity = apply_callier_components(
        dye, zeros, profile, collimation=1.0
    )
    zero_collimation = apply_callier_components(
        dye, silver, profile, collimation=0.0
    )
    full = apply_callier_components(dye, silver, profile, collimation=1.0)
    direct_formula = dye + silver * callier_q_factor(
        silver, profile, collimation=1.0
    )
    partition_errors = []
    for rows in evaluation["partition_rows"]:
        pieces = []
        for start in range(0, dye.shape[0], rows):
            stop = min(dye.shape[0], start + rows)
            pieces.append(
                apply_callier_components(
                    dye[start:stop], silver[start:stop], profile, collimation=1.0
                )
            )
        partition_errors.append(float(np.max(np.abs(np.concatenate(pieces) - full))))

    invalid_rejections = 0
    invalid_cases = [
        lambda: apply_callier_components(
            np.full((1, 1, 3), -0.1), zeros[:1, :1], profile, collimation=1.0
        ),
        lambda: apply_callier_components(
            dye[:2], silver[:1], profile, collimation=1.0
        ),
        lambda: apply_callier_components(
            dye[:1], silver[:1], profile, collimation=1.1
        ),
        lambda: invert_callier_components(
            np.zeros((1, 1, 3)), np.ones((1, 1, 3)), profile, collimation=1.0
        ),
    ]
    for case in invalid_cases:
        try:
            case()
        except (TypeError, ValueError):
            invalid_rejections += 1

    metrics = {
        "q_at_zero_exact": bool(np.array_equal(q_full[0], np.ones_like(q_full[0]))),
        "minimum_q": float(np.min(q_full)),
        "maximum_q": float(np.max(q_full)),
        "peak_density_rgb": [float(value) for value in peak_densities],
        "peak_q_rgb": [float(value) for value in peak_q],
        "half_collimation_increment_max_error": half_error,
        "minimum_directed_density_step": minimum_step,
        "inverse_maximum_absolute_error": inverse_error,
        "dye_only_identity_exact": bool(np.array_equal(dye_identity, dye)),
        "zero_collimation_identity_exact": bool(
            np.array_equal(zero_collimation, dye + silver)
        ),
        "mixed_component_formula_max_error": float(
            np.max(np.abs(full - direct_formula))
        ),
        "partition_maximum_absolute_errors": partition_errors,
        "invalid_rejections": invalid_rejections,
        "invalid_cases": len(invalid_cases),
        "q_zero_collimation_identity_exact": bool(
            np.array_equal(q_zero, np.ones_like(q_zero))
        ),
    }
    peak = np.asarray(metrics["peak_density_rgb"])
    q_peak = np.asarray(metrics["peak_q_rgb"])
    gate_results = {
        "q_at_zero": metrics["q_at_zero_exact"],
        "q_finite_and_at_least_one": bool(
            np.all(np.isfinite(q_full)) and metrics["minimum_q"] >= 1.0
        ),
        "maximum_q": metrics["maximum_q"] <= gates["maximum_q_must_not_exceed"],
        "peak_density": bool(
            np.max(np.abs(peak - profile.peak_diffuse_density))
            <= gates["maximum_peak_density_error"]
        ),
        "spectral_peak_order": bool(q_peak[2] > q_peak[1] > q_peak[0] > 1.0),
        "half_collimation": half_error
        <= gates["half_collimation_increment_relative_error_maximum"],
        "strictly_increasing_directed_density": minimum_step > 0.0,
        "inverse": inverse_error <= gates["inverse_maximum_absolute_error"],
        "dye_identity": metrics["dye_only_identity_exact"],
        "zero_collimation_identity": metrics["zero_collimation_identity_exact"],
        "mixed_formula": metrics["mixed_component_formula_max_error"]
        <= gates["mixed_component_formula_error_maximum"],
        "partition": max(partition_errors)
        <= gates["full_vs_partition_maximum_absolute_error"],
        "invalid_domain": invalid_rejections == len(invalid_cases),
        "q_zero_collimation": metrics["q_zero_collimation_identity_exact"],
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "profile": config["operator"]["profile"],
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_generic_callier_primitive"
            if passed
            else "close_generic_callier_q_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CallierOperatorError",
    "evaluate_operator",
    "load_contract",
    "profile_from_contract",
]
