"""Frozen U6.P4AD analytic characteristic-slope grain evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.film_physics.characteristic_slope_grain import (
    HillCharacteristicProfile,
    developed_density_variance,
)


SCHEMA = (
    "neuro_film.u6_p4ad_characteristic_slope_grain_reference_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u6_p4ad_characteristic_slope_grain_reference_report.v1"
)


class CharacteristicSlopeGrainError(RuntimeError):
    """Raised when P4AD input identities or frozen gates drift."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("normalized_photon_scale") != 1.0
        or model.get("density_min") != 0.1
        or model.get("density_span") != 2.0
        or model.get("half_exposure") != 0.18
        or model.get("gamma") != 1.6
        or model.get("display_rgb_noise_allowed")
        or model.get("post_development_additive_noise_allowed")
        or evaluation.get("exposure_minimum") != 0.0001
        or evaluation.get("exposure_maximum") != 16.0
        or evaluation.get("sample_count") != 32769
        or evaluation.get("photon_scales") != [128.0, 1024.0, 8192.0]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_characteristic_derivative_relative_error")
        != 0.00001
        or gates.get("maximum_analytic_peak_log_exposure_error") != 0.002
        or gates.get("maximum_peak_index_distance") != 1
        or gates.get("maximum_low_tail_to_peak_variance_ratio") != 0.05
        or gates.get("maximum_high_tail_to_peak_variance_ratio") != 0.05
        or gates.get("maximum_normalized_shape_difference_across_photon_scales")
        != 1e-12
        or not gates.get("no_parameter_fit")
    ):
        raise CharacteristicSlopeGrainError("P4AD frozen contract drift")
    return payload


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CharacteristicSlopeGrainError("P4AD parent path must be relative")
    return root / path


def _validate_parents(config: dict[str, Any], root: Path) -> dict[str, str]:
    parents = config["parents"]
    identities: dict[str, str] = {}
    for stem in ("p4ac_decision", "sensitometry_contract", "sensitometry_decision"):
        path = _relative(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise CharacteristicSlopeGrainError(f"missing P4AD parent: {stem}")
        actual = hash_file(path)
        if actual != parents[f"{stem}_sha256"]:
            raise CharacteristicSlopeGrainError(
                f"P4AD parent hash mismatch: {stem}"
            )
        identities[stem] = actual
    p4ac = json.loads(
        _relative(root, parents["p4ac_decision_path"]).read_text(encoding="utf-8")
    )
    sensitometry = json.loads(
        _relative(root, parents["sensitometry_decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        p4ac.get("decision") != "close_peaked_density_shape_without_rescue"
        or p4ac.get("passed") is not False
        or sensitometry.get("decision") != "sensitometry_primitive_pass"
    ):
        raise CharacteristicSlopeGrainError("P4AD parent decision mismatch")
    return identities


def evaluate_reference(config: dict[str, Any], root: Path) -> dict[str, Any]:
    identities = _validate_parents(config, root)
    model = config["model"]
    evaluation = config["evaluation"]
    profile = HillCharacteristicProfile(
        density_min=float(model["density_min"]),
        density_span=float(model["density_span"]),
        half_exposure=float(model["half_exposure"]),
        gamma=float(model["gamma"]),
    )
    exposure = np.geomspace(
        float(evaluation["exposure_minimum"]),
        float(evaluation["exposure_maximum"]),
        int(evaluation["sample_count"]),
        dtype=np.float64,
    )
    relative_step = float(evaluation["finite_difference_relative_step"])
    density = profile.density(exposure)
    derivative = profile.derivative(exposure)
    finite_difference = (
        profile.density(exposure * (1.0 + relative_step))
        - profile.density(exposure * (1.0 - relative_step))
    ) / (2.0 * relative_step * exposure)
    derivative_error = np.abs(finite_difference - derivative) / derivative

    normalized_shapes = []
    variance_summaries = []
    peak_indices = []
    for scale in evaluation["photon_scales"]:
        variance = developed_density_variance(
            exposure, profile, photon_scale=float(scale)
        )
        peak_index = int(np.argmax(variance))
        peak = float(variance[peak_index])
        normalized = variance / peak
        normalized_shapes.append(normalized)
        peak_indices.append(peak_index)
        variance_summaries.append(
            {
                "photon_scale": float(scale),
                "peak_exposure": float(exposure[peak_index]),
                "peak_variance": peak,
                "low_tail_to_peak": float(normalized[0]),
                "high_tail_to_peak": float(normalized[-1]),
            }
        )

    reference_shape = normalized_shapes[0]
    shape_difference = max(
        float(np.max(np.abs(candidate - reference_shape)))
        for candidate in normalized_shapes[1:]
    )
    analytic_peak = profile.variance_peak_exposure
    nearest_peak_index = int(np.argmin(np.abs(np.log(exposure / analytic_peak))))
    observed_peak_index = peak_indices[0]
    peak_log_error = abs(
        math.log(float(exposure[observed_peak_index]) / analytic_peak)
    )
    low_ratio = max(item["low_tail_to_peak"] for item in variance_summaries)
    high_ratio = max(item["high_tail_to_peak"] for item in variance_summaries)
    metrics = {
        "density_minimum": float(np.min(density)),
        "density_maximum": float(np.max(density)),
        "minimum_density_step": float(np.min(np.diff(density))),
        "maximum_characteristic_derivative_relative_error": float(
            np.max(derivative_error)
        ),
        "analytic_variance_peak_exposure": analytic_peak,
        "sampled_variance_peak_exposure": float(exposure[observed_peak_index]),
        "analytic_peak_log_exposure_error": peak_log_error,
        "peak_index_distance": abs(observed_peak_index - nearest_peak_index),
        "maximum_low_tail_to_peak_variance_ratio": low_ratio,
        "maximum_high_tail_to_peak_variance_ratio": high_ratio,
        "maximum_normalized_shape_difference_across_photon_scales": (
            shape_difference
        ),
        "variance_summaries": variance_summaries,
        "finite_nonnegative": bool(
            np.all(np.isfinite(density))
            and np.all(np.isfinite(derivative))
            and all(np.all(np.isfinite(item)) for item in normalized_shapes)
            and all(np.all(item >= 0.0) for item in normalized_shapes)
        ),
    }
    gates = config["gates"]
    gate_results = {
        "characteristic_derivative": metrics[
            "maximum_characteristic_derivative_relative_error"
        ]
        <= gates["maximum_characteristic_derivative_relative_error"],
        "analytic_peak": metrics["analytic_peak_log_exposure_error"]
        <= gates["maximum_analytic_peak_log_exposure_error"],
        "sampled_peak_index": metrics["peak_index_distance"]
        <= gates["maximum_peak_index_distance"],
        "low_tail": low_ratio
        <= gates["maximum_low_tail_to_peak_variance_ratio"],
        "high_tail": high_ratio
        <= gates["maximum_high_tail_to_peak_variance_ratio"],
        "photon_scale_shape": shape_difference
        <= gates["maximum_normalized_shape_difference_across_photon_scales"],
        "density_strictly_increasing": metrics["minimum_density_step"] > 0.0,
        "finite_nonnegative_interior_peak": bool(
            metrics["finite_nonnegative"]
            and 0 < observed_peak_index < exposure.size - 1
        ),
        "no_parameter_fit": True,
        "parent_identity": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "model": model,
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_existing_sensitometry_derivative_propagation_audit"
            if passed
            else "close_characteristic_slope_grain_reference_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CharacteristicSlopeGrainError",
    "evaluate_reference",
    "load_contract",
]
