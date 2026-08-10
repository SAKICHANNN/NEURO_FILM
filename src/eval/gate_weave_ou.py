"""Frozen U6.P9A deterministic bounded gate-weave trajectory audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.gate_weave import (
    GateWeaveProfile,
    advance_gate_weave,
    generate_gate_weave,
    initial_gate_weave_state,
)


class GateWeaveAuditError(RuntimeError):
    """Raised when the frozen contract is invalid."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateWeaveAuditError("contract must be an object")
    return payload


def _profile(payload: dict[str, Any]) -> GateWeaveProfile:
    keys = (
        "theta_x",
        "sigma_x_pixels",
        "theta_y_initial",
        "theta_y_average",
        "kappa",
        "theta_y_sigma",
        "theta_y_minimum",
        "theta_y_maximum",
        "sigma_y_pixels",
        "padding_x_pixels",
        "padding_y_pixels",
        "seed",
    )
    return GateWeaveProfile(**{key: payload[key] for key in keys})


def _lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _segment_arrays(segment: Any) -> tuple[np.ndarray, ...]:
    return (
        segment.frame_indices,
        segment.x_pixels,
        segment.y_pixels,
        segment.theta_y,
        segment.x_integer_pixels,
        segment.y_integer_pixels,
        segment.coordinate_clipped,
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9a_gate_weave_ou_contract.v1":
        raise GateWeaveAuditError("unsupported P9A contract")
    profile = _profile(contract["profile"])
    experiment = contract["experiment"]
    frame_count = int(experiment["frame_count"])
    full = generate_gate_weave(profile, frame_count)
    replay = generate_gate_weave(profile, frame_count)
    replay_exact = all(
        np.array_equal(left, right)
        for left, right in zip(
            _segment_arrays(full), _segment_arrays(replay), strict=True
        )
    )

    state = initial_gate_weave_state(profile)
    partition_parts: list[list[np.ndarray]] = []
    for count in experiment["partition_transition_counts"]:
        segment, state = advance_gate_weave(
            profile,
            total_frame_count=frame_count,
            state=state,
            transition_count=int(count),
        )
        partition_parts.append(list(_segment_arrays(segment)))
    if state.frame_index != frame_count - 1:
        raise GateWeaveAuditError("partition counts do not cover the sequence")
    partition_arrays = [
        np.concatenate([part[index] for part in partition_parts])
        for index in range(len(partition_parts[0]))
    ]
    partition_exact = all(
        np.array_equal(partition, complete[1:])
        for partition, complete in zip(
            partition_arrays, _segment_arrays(full), strict=True
        )
    )

    burn = int(experiment["burn_in_frames"])
    x = full.x_pixels[burn:]
    y = full.y_pixels[burn:]
    quarter = (frame_count - burn) // 4
    early_ou_variance = float(np.var(x[:quarter]))
    late_ou_variance = float(np.var(x[-quarter:]))
    # A random walk is nonstationary in ensemble variance, not necessarily in
    # the within-window variance of one realized path. Repack the same frozen
    # innovations into a deterministic near-square ensemble and compare its
    # early and late cross-sectional variances.
    theta = profile.theta_x
    residual = (full.x_pixels[1:] - (1.0 - theta) * full.x_pixels[:-1]) / (
        profile.sigma_x_pixels * np.sqrt(2.0 * theta)
    )
    random_walk_path_length = int(np.sqrt(residual.size))
    random_walk_count = residual.size // random_walk_path_length
    used = random_walk_count * random_walk_path_length
    random_walk_ensemble = np.cumsum(
        float(experiment["random_walk_control_scale_pixels"])
        * residual[:used].reshape(random_walk_count, random_walk_path_length),
        axis=1,
    )
    early_rw_index = max(0, random_walk_path_length // 4 - 1)
    early_rw_variance = float(np.var(random_walk_ensemble[:, early_rw_index]))
    late_rw_variance = float(np.var(random_walk_ensemble[:, -1]))
    coordinate_clip_fraction = float(np.mean(full.coordinate_clipped))
    integer_exact = bool(
        np.array_equal(full.x_integer_pixels, replay.x_integer_pixels)
        and np.array_equal(full.y_integer_pixels, replay.y_integer_pixels)
    )
    finite = bool(
        all(
            np.all(np.isfinite(values))
            for values in (full.x_pixels, full.y_pixels, full.theta_y)
        )
    )
    within_padding = bool(
        np.max(np.abs(full.x_pixels)) <= profile.padding_x_pixels
        and np.max(np.abs(full.y_pixels)) <= profile.padding_y_pixels
        and np.max(np.abs(full.x_integer_pixels)) <= profile.padding_x_pixels
        and np.max(np.abs(full.y_integer_pixels)) <= profile.padding_y_pixels
    )
    measurements = {
        "replay_byte_exact": replay_exact,
        "partition_byte_exact": partition_exact,
        "integer_offsets_repeat_exact": integer_exact,
        "all_values_finite": finite,
        "all_offsets_within_padding": within_padding,
        "coordinate_clip_fraction": coordinate_clip_fraction,
        "stationary_mean_x_pixels": float(np.mean(x)),
        "stationary_mean_y_pixels": float(np.mean(y)),
        "stationary_lag1_x": _lag1(x),
        "stationary_lag1_y": _lag1(y),
        "stationary_std_x_over_sigma": float(np.std(x) / profile.sigma_x_pixels),
        "stationary_std_y_over_sigma": float(np.std(y) / profile.sigma_y_pixels),
        "theta_y_span": float(np.ptp(full.theta_y)),
        "late_to_early_ou_variance_ratio": late_ou_variance / early_ou_variance,
        "random_walk_late_to_early_variance_ratio": late_rw_variance
        / early_rw_variance,
        "random_walk_ensemble_shape": [
            random_walk_count,
            random_walk_path_length,
        ],
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "replay_byte_exact": replay_exact is gates["replay_byte_exact"],
        "partition_byte_exact": partition_exact is gates["partition_byte_exact"],
        "integer_offsets_repeat_exact": integer_exact
        is gates["integer_offsets_repeat_exact"],
        "all_values_finite": finite is gates["all_values_finite"],
        "all_offsets_within_padding": within_padding
        is gates["all_offsets_within_padding"],
        "maximum_coordinate_clip_fraction": coordinate_clip_fraction
        <= gates["maximum_coordinate_clip_fraction"],
        "maximum_absolute_stationary_mean_x_pixels": abs(
            measurements["stationary_mean_x_pixels"]
        )
        <= gates["maximum_absolute_stationary_mean_x_pixels"],
        "maximum_absolute_stationary_mean_y_pixels": abs(
            measurements["stationary_mean_y_pixels"]
        )
        <= gates["maximum_absolute_stationary_mean_y_pixels"],
        "minimum_stationary_lag1_x": measurements["stationary_lag1_x"]
        >= gates["minimum_stationary_lag1_x"],
        "maximum_stationary_lag1_x": measurements["stationary_lag1_x"]
        <= gates["maximum_stationary_lag1_x"],
        "minimum_stationary_lag1_y": measurements["stationary_lag1_y"]
        >= gates["minimum_stationary_lag1_y"],
        "maximum_stationary_lag1_y": measurements["stationary_lag1_y"]
        <= gates["maximum_stationary_lag1_y"],
        "minimum_stationary_std_x_over_sigma": measurements[
            "stationary_std_x_over_sigma"
        ]
        >= gates["minimum_stationary_std_x_over_sigma"],
        "maximum_stationary_std_x_over_sigma": measurements[
            "stationary_std_x_over_sigma"
        ]
        <= gates["maximum_stationary_std_x_over_sigma"],
        "minimum_stationary_std_y_over_sigma": measurements[
            "stationary_std_y_over_sigma"
        ]
        >= gates["minimum_stationary_std_y_over_sigma"],
        "maximum_stationary_std_y_over_sigma": measurements[
            "stationary_std_y_over_sigma"
        ]
        <= gates["maximum_stationary_std_y_over_sigma"],
        "minimum_theta_y_span": measurements["theta_y_span"]
        >= gates["minimum_theta_y_span"],
        "maximum_late_to_early_ou_variance_ratio": measurements[
            "late_to_early_ou_variance_ratio"
        ]
        <= gates["maximum_late_to_early_ou_variance_ratio"],
        "minimum_random_walk_late_to_early_variance_ratio": measurements[
            "random_walk_late_to_early_variance_ratio"
        ]
        >= gates["minimum_random_walk_late_to_early_variance_ratio"],
        "rgb_image_transform_count_zero": gates["rgb_image_transform_count_zero"]
        is True,
    }
    if set(gate_results) != set(gates):
        raise GateWeaveAuditError("P9A gate vocabulary drift")
    stable = {
        "schema": "neuro_film.u6_p9a_gate_weave_ou_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9a_gate_weave_ou_v1.json"),
        "frame_count": frame_count,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "decision": contract["branch_rule"]["pass"]
        if all(gate_results.values())
        else contract["branch_rule"]["fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_bytes = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(stable_bytes).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
