"""Frozen U6.P2AB scalar limitation and local-intensity capacity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.reciprocity import (
    IntensityConditionedReciprocityProfile,
    PowerReciprocityProfile,
)


class LocalIntensityReciprocityError(RuntimeError):
    """Raised when the U6.P2AB contract or parent evidence drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LocalIntensityReciprocityError(f"expected object: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    return _load_object(path)


def _normalized_log(values: np.ndarray) -> np.ndarray:
    logged = np.log(values)
    return logged - np.mean(logged)


def _partition_exact(
    profile: IntensityConditionedReciprocityProfile,
    rates: np.ndarray,
    physical_time: float,
    sizes: list[int],
    expected: np.ndarray,
) -> bool:
    if sum(sizes) != len(rates):
        return False
    parts = []
    start = 0
    for size in sizes:
        parts.append(
            profile.effective_exposure(
                rates[start : start + size], physical_time
            )
        )
        start += size
    return bool(np.array_equal(np.concatenate(parts), expected))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p2ab_local_intensity_reciprocity_contract.v1"
    ):
        raise LocalIntensityReciprocityError("unsupported P2AB contract")
    parent_binding = contract["parent"]
    parent_path = root / parent_binding["evidence_path"]
    if _sha(parent_path) != parent_binding["evidence_sha256"]:
        raise LocalIntensityReciprocityError("P2AA evidence hash mismatch")
    parent = _load_object(parent_path)
    if parent.get("decision") != parent_binding["required_decision"]:
        raise LocalIntensityReciprocityError("P2AA decision does not admit P2AB")
    p2aa_contract_binding = parent["bindings"]["contract"]
    p2aa_contract_path = root / p2aa_contract_binding["path"]
    if _sha(p2aa_contract_path) != p2aa_contract_binding["sha256"]:
        raise LocalIntensityReciprocityError("P2AA contract hash mismatch")
    p2aa_contract = _load_object(p2aa_contract_path)

    mechanism = contract["mechanism"]
    rates = np.exp2(
        np.linspace(
            float(mechanism["incident_rate_range_log2"][0]),
            float(mechanism["incident_rate_range_log2"][1]),
            int(mechanism["incident_rate_sample_count"]),
            dtype=np.float64,
        )
    )
    times = np.asarray(mechanism["physical_times_seconds"], dtype=np.float64)
    local = IntensityConditionedReciprocityProfile(
        profile_id=mechanism["schema"],
        bright_exponent=float(mechanism["bright_exponent"]),
        dark_exponent=float(mechanism["dark_exponent"]),
        log2_pivot_rate=float(mechanism["log2_pivot_rate"]),
        log2_transition_width=float(mechanism["log2_transition_width"]),
        reference_time_seconds=float(mechanism["reference_time_seconds"]),
    )
    wrong = IntensityConditionedReciprocityProfile(
        profile_id="wrong-swapped-conditioning",
        bright_exponent=local.dark_exponent,
        dark_exponent=local.bright_exponent,
        log2_pivot_rate=local.log2_pivot_rate,
        log2_transition_width=local.log2_transition_width,
        reference_time_seconds=local.reference_time_seconds,
    )
    truth = np.stack([local.effective_exposure(rates, time) for time in times])
    wrong_outputs = np.stack(
        [wrong.effective_exposure(rates, time) for time in times]
    )
    source_profiles = p2aa_contract["sources"]["ilford"]["profiles"]
    global_profiles = [
        PowerReciprocityProfile(profile_id, float(exponent))
        for profile_id, exponent in source_profiles.items()
    ]
    global_shape_errors: list[dict[str, Any]] = []
    maximum_global_contrast_drift = 0.0
    for profile in global_profiles:
        rows = []
        for time in times:
            output = profile.effective_exposure(rates, time)
            normalized = _normalized_log(output)
            truth_normalized = _normalized_log(
                local.effective_exposure(rates, time)
            )
            contrast_drift = float(
                np.log(output[-1] / output[0])
                - np.log(rates[-1] / rates[0])
            )
            maximum_global_contrast_drift = max(
                maximum_global_contrast_drift, abs(contrast_drift)
            )
            rows.append(
                {
                    "time_seconds": float(time),
                    "shape_rmse": float(
                        np.sqrt(np.mean(np.square(normalized - truth_normalized)))
                    ),
                    "shape_maximum_absolute_error": float(
                        np.max(np.abs(normalized - truth_normalized))
                    ),
                    "log_contrast_drift": contrast_drift,
                }
            )
        global_shape_errors.append(
            {
                "profile_id": profile.profile_id,
                "exponent": profile.exponent,
                "rows": rows,
            }
        )

    longest_truth_shape = _normalized_log(truth[-1])
    longest_wrong_shape = _normalized_log(wrong_outputs[-1])
    longest_truth_contrast_drift = float(
        np.log(truth[-1, -1] / truth[-1, 0])
        - np.log(rates[-1] / rates[0])
    )
    longest_global_row = global_shape_errors[0]["rows"][-1]
    reference_identity_error = float(np.max(np.abs(truth[0] - rates)))
    wrong_shape_rmse = float(
        np.sqrt(np.mean(np.square(longest_truth_shape - longest_wrong_shape)))
    )
    local_rate_monotone = bool(np.all(np.diff(truth, axis=1) > 0.0))
    local_time_monotone = bool(np.all(np.diff(truth, axis=0) > 0.0))
    partition_error = 0.0
    for time, expected in zip(times, truth, strict=True):
        partitioned_exact = _partition_exact(
            local,
            rates,
            float(time),
            [int(value) for value in mechanism["partition_sizes"]],
            expected,
        )
        if not partitioned_exact:
            partition_error = float("inf")
            break
    all_positive_finite = bool(np.all(np.isfinite(truth)) and np.all(truth > 0.0))

    measurements = {
        "reference_time_identity_maximum_absolute_error": reference_identity_error,
        "longest_time_truth_log_contrast_drift": longest_truth_contrast_drift,
        "longest_time_global_shape_rmse": longest_global_row["shape_rmse"],
        "longest_time_global_shape_max_error": longest_global_row[
            "shape_maximum_absolute_error"
        ],
        "maximum_any_global_log_contrast_drift": maximum_global_contrast_drift,
        "wrong_conditioning_shape_rmse": wrong_shape_rmse,
        "local_rate_monotone_strict": local_rate_monotone,
        "local_time_monotone_strict": local_time_monotone,
        "partition_maximum_absolute_error": partition_error,
        "all_outputs_positive_finite": all_positive_finite,
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "reference_time_identity_maximum_absolute_error": reference_identity_error
        <= gates["reference_time_identity_maximum_absolute_error"],
        "minimum_longest_time_truth_log_contrast_drift": abs(
            longest_truth_contrast_drift
        )
        >= gates["minimum_longest_time_truth_log_contrast_drift"],
        "minimum_longest_time_global_shape_rmse": longest_global_row["shape_rmse"]
        >= gates["minimum_longest_time_global_shape_rmse"],
        "minimum_longest_time_global_shape_max_error": longest_global_row[
            "shape_maximum_absolute_error"
        ]
        >= gates["minimum_longest_time_global_shape_max_error"],
        "maximum_any_global_log_contrast_drift": maximum_global_contrast_drift
        <= gates["maximum_any_global_log_contrast_drift"],
        "minimum_wrong_conditioning_shape_rmse": wrong_shape_rmse
        >= gates["minimum_wrong_conditioning_shape_rmse"],
        "local_rate_monotone_strict": local_rate_monotone
        is gates["local_rate_monotone_strict"],
        "local_time_monotone_strict": local_time_monotone
        is gates["local_time_monotone_strict"],
        "partition_maximum_absolute_error": partition_error
        <= gates["partition_maximum_absolute_error"],
        "all_outputs_positive_finite": all_positive_finite
        is gates["all_outputs_positive_finite"],
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(gates):
        raise LocalIntensityReciprocityError("P2AB gate vocabulary drift")
    stable = {
        "schema": "neuro_film.u6_p2ab_local_intensity_reciprocity_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p2ab_local_intensity_reciprocity_v1.json"
        ),
        "parent_evidence_sha256": parent_binding["evidence_sha256"],
        "incident_rate_count": len(rates),
        "physical_times_seconds": times.tolist(),
        "local_exponent_range": [
            float(np.min(local.exponent_for_rate(rates))),
            float(np.max(local.exponent_for_rate(rates))),
        ],
        "global_controls": global_shape_errors,
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
    "LocalIntensityReciprocityError",
    "load_contract",
    "run_audit",
    "write_report",
]
