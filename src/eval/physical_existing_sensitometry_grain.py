"""Frozen U6.P4AE compatibility audit for current U2.2 sensitometry."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.characteristic_slope_grain import (
    propagate_poisson_exposure_variance,
)


SCHEMA = (
    "neuro_film.u6_p4ae_existing_sensitometry_grain_variance_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u6_p4ae_existing_sensitometry_grain_variance_report.v1"
)


class ExistingSensitometryGrainError(RuntimeError):
    """Raised when P4AE contract or frozen parent identities drift."""


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
        or model.get("photon_scale") != 1024.0
        or model.get("curve_or_encoder_changes_allowed")
        or model.get("amplitude_fit_allowed")
        or model.get("display_rgb_noise_allowed")
        or evaluation.get("linear_exposure_minimum") != 0.0001
        or evaluation.get("linear_exposure_maximum") != 16.0
        or evaluation.get("sample_count") != 32769
        or evaluation.get("minimum_peak_index_margin") != 16
        or evaluation.get("peak_density_interval") != [0.2, 1.8]
        or evaluation.get("runs") != 2
        or evaluation.get("post_result_retuning_allowed")
        or gates.get("maximum_chain_derivative_relative_error") != 0.0001
        or gates.get("required_interior_peaked_channels") != 3
        or gates.get("maximum_low_tail_to_peak_variance_ratio") != 0.75
        or gates.get("maximum_high_tail_to_peak_variance_ratio") != 0.25
        or gates.get("minimum_channels_with_peak_density_in_interval") != 2
        or gates.get("minimum_pairwise_normalized_shape_linf_distance") != 0.01
        or not gates.get("no_parameter_fit")
    ):
        raise ExistingSensitometryGrainError("P4AE frozen contract drift")
    return payload


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ExistingSensitometryGrainError("P4AE parent path must be relative")
    return root / path


def _load_parents(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    parents = config["parents"]
    identities: dict[str, str] = {}
    for stem in ("p4ad_decision", "sensitometry_contract", "sensitometry_decision"):
        path = _relative(root, parents[f"{stem}_path"])
        if not path.is_file():
            raise ExistingSensitometryGrainError(f"missing P4AE parent: {stem}")
        actual = hash_file(path)
        if actual != parents[f"{stem}_sha256"]:
            raise ExistingSensitometryGrainError(
                f"P4AE parent hash mismatch: {stem}"
            )
        identities[stem] = actual
    p4ad = json.loads(
        _relative(root, parents["p4ad_decision_path"]).read_text(encoding="utf-8")
    )
    sensitometry_decision = json.loads(
        _relative(root, parents["sensitometry_decision_path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        p4ad.get("decision")
        != "open_existing_sensitometry_derivative_propagation_audit"
        or not p4ad.get("passed")
        or sensitometry_decision.get("decision") != "sensitometry_primitive_pass"
    ):
        raise ExistingSensitometryGrainError("P4AE parent decision mismatch")
    sensitometry = json.loads(
        _relative(root, parents["sensitometry_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    return sensitometry, identities


def evaluate_existing(config: dict[str, Any], root: Path) -> dict[str, Any]:
    sensitometry, identities = _load_parents(config, root)
    operator = build_operator(sensitometry)
    evaluation = config["evaluation"]
    exposure = np.geomspace(
        float(evaluation["linear_exposure_minimum"]),
        float(evaluation["linear_exposure_maximum"]),
        int(evaluation["sample_count"]),
        dtype=np.float64,
    )
    step = float(evaluation["finite_difference_relative_step"])
    log_exposure = operator.encoder.apply(exposure)
    encoder_derivative = operator.encoder.derivative(exposure)
    channel_reports = []
    normalized_shapes: list[np.ndarray] = []
    derivative_errors = []
    interior_count = 0
    density_interval_count = 0
    low_ratios = []
    high_ratios = []
    peak_margin = int(evaluation["minimum_peak_index_margin"])
    density_interval = evaluation["peak_density_interval"]

    for curve in operator.curves:
        density = curve.apply(log_exposure)
        derivative = curve.derivative(log_exposure) * encoder_derivative
        density_high = curve.apply(
            operator.encoder.apply(exposure * (1.0 + step))
        )
        density_low = curve.apply(
            operator.encoder.apply(exposure * (1.0 - step))
        )
        finite_difference = (density_high - density_low) / (2.0 * step * exposure)
        derivative_error = np.abs(finite_difference - derivative) / derivative
        variance = propagate_poisson_exposure_variance(
            exposure,
            derivative,
            photon_scale=float(config["model"]["photon_scale"]),
        )
        peak_index = int(np.argmax(variance))
        peak_variance = float(variance[peak_index])
        normalized = variance / peak_variance
        peak_density = float(density[peak_index])
        interior = peak_margin <= peak_index < exposure.size - peak_margin
        density_in_interval = bool(
            density_interval[0] <= peak_density <= density_interval[1]
        )
        interior_count += int(interior)
        density_interval_count += int(density_in_interval)
        derivative_errors.append(float(np.max(derivative_error)))
        normalized_shapes.append(normalized)
        low_ratios.append(float(normalized[0]))
        high_ratios.append(float(normalized[-1]))
        channel_reports.append(
            {
                "layer": curve.layer,
                "peak_index": peak_index,
                "peak_exposure": float(exposure[peak_index]),
                "peak_density": peak_density,
                "peak_variance": peak_variance,
                "low_tail_to_peak": float(normalized[0]),
                "high_tail_to_peak": float(normalized[-1]),
                "maximum_chain_derivative_relative_error": float(
                    np.max(derivative_error)
                ),
                "minimum_density_step": float(np.min(np.diff(density))),
                "interior_peak": interior,
                "peak_density_in_interval": density_in_interval,
            }
        )

    pairwise_distances = {
        f"{first}-{second}": float(
            np.max(np.abs(normalized_shapes[first_index] - normalized_shapes[second_index]))
        )
        for (first_index, first), (second_index, second) in combinations(
            enumerate(("red", "green", "blue")), 2
        )
    }
    minimum_pairwise_distance = min(pairwise_distances.values())
    metrics = {
        "channel_reports": channel_reports,
        "maximum_chain_derivative_relative_error": max(derivative_errors),
        "interior_peaked_channels": interior_count,
        "maximum_low_tail_to_peak_variance_ratio": max(low_ratios),
        "maximum_high_tail_to_peak_variance_ratio": max(high_ratios),
        "channels_with_peak_density_in_interval": density_interval_count,
        "pairwise_normalized_shape_linf_distance": pairwise_distances,
        "minimum_pairwise_normalized_shape_linf_distance": minimum_pairwise_distance,
        "density_strictly_increasing": all(
            item["minimum_density_step"] > 0.0 for item in channel_reports
        ),
        "variance_finite_nonnegative": bool(
            all(np.all(np.isfinite(shape)) for shape in normalized_shapes)
            and all(np.all(shape >= 0.0) for shape in normalized_shapes)
        ),
    }
    gates = config["gates"]
    gate_results = {
        "chain_derivative": metrics["maximum_chain_derivative_relative_error"]
        <= gates["maximum_chain_derivative_relative_error"],
        "interior_peaks": interior_count
        == gates["required_interior_peaked_channels"],
        "low_tail": metrics["maximum_low_tail_to_peak_variance_ratio"]
        <= gates["maximum_low_tail_to_peak_variance_ratio"],
        "high_tail": metrics["maximum_high_tail_to_peak_variance_ratio"]
        <= gates["maximum_high_tail_to_peak_variance_ratio"],
        "peak_density_interval": density_interval_count
        >= gates["minimum_channels_with_peak_density_in_interval"],
        "layer_shape_diversity": minimum_pairwise_distance
        >= gates["minimum_pairwise_normalized_shape_linf_distance"],
        "density_strictly_increasing": metrics["density_strictly_increasing"],
        "variance_finite_nonnegative": metrics["variance_finite_nonnegative"],
        "no_parameter_fit": True,
        "parent_identity": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_sha256": identities,
        "model": config["model"],
        "metrics": metrics,
        "gate_results": gate_results,
        "passed": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "open_synthetic_derivative_conditioned_structure_compiler"
            if passed
            else "close_existing_sensitometry_grain_variance_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "ExistingSensitometryGrainError",
    "evaluate_existing",
    "load_contract",
]
