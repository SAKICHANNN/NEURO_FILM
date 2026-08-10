"""Frozen U6.P9E temporal exposure-flicker mechanism audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.structure_compiler import counter_normal_region
from src.film_physics.temporal_exposure import (
    TemporalExposureProfile,
    advance_temporal_exposure,
    apply_temporal_exposure,
    generate_temporal_exposure,
    initial_temporal_exposure_state,
)


class TemporalExposureAuditError(RuntimeError):
    """Raised when the frozen P9E contract or result is invalid."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TemporalExposureAuditError("contract must be an object")
    return payload


def _profile(payload: dict[str, Any]) -> TemporalExposureProfile:
    return TemporalExposureProfile(
        theta=float(payload["theta"]),
        stationary_sigma_stops=float(payload["stationary_sigma_stops"]),
        maximum_absolute_stops=float(payload["maximum_absolute_stops"]),
        seed=int(payload["seed"]),
    )


def _arrays(segment: Any) -> tuple[np.ndarray, ...]:
    return (
        segment.frame_indices,
        segment.offset_stops,
        segment.multiplier,
        segment.clipped,
    )


def _low_frequency_power_fraction(values: np.ndarray, cutoff: float) -> float:
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(centered.size)
    eligible = frequencies > 0.0
    denominator = float(np.sum(power[eligible]))
    if denominator <= 0.0:
        raise TemporalExposureAuditError("temporal control has zero spectral power")
    return float(np.sum(power[(frequencies > 0.0) & (frequencies <= cutoff)])) / denominator


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9e_temporal_exposure_flicker_contract.v1":
        raise TemporalExposureAuditError("unsupported P9E contract")
    profile = _profile(contract["profile"])
    experiment = contract["experiment"]
    frame_count = int(experiment["frame_count"])
    candidate = generate_temporal_exposure(profile, frame_count)
    replay = generate_temporal_exposure(profile, frame_count)
    replay_exact = all(
        np.array_equal(left, right)
        for left, right in zip(_arrays(candidate), _arrays(replay), strict=True)
    )

    state = initial_temporal_exposure_state()
    parts: list[tuple[np.ndarray, ...]] = []
    for count in experiment["partition_transition_counts"]:
        segment, state = advance_temporal_exposure(
            profile,
            total_frame_count=frame_count,
            state=state,
            transition_count=int(count),
        )
        parts.append(_arrays(segment))
    if state.frame_index != frame_count - 1:
        raise TemporalExposureAuditError("partition counts do not cover sequence")
    partition_exact = all(
        np.array_equal(
            np.concatenate([part[index] for part in parts]), complete[1:]
        )
        for index, complete in enumerate(_arrays(candidate))
    )

    normals = counter_normal_region(
        (frame_count, 1),
        origin_yx=(0, 0),
        shape=(frame_count, 1),
        seed=profile.seed ^ 0x6A09E667F3BCC909,
    )[:, 0]
    gamma = np.clip(
        np.exp(float(experiment["iid_gamma_log_standard_deviation"]) * normals),
        float(experiment["iid_gamma_minimum"]),
        float(experiment["iid_gamma_maximum"]),
    )
    probe = float(experiment["iid_gamma_probe_linear"])
    iid_gamma_stops = np.log2(np.power(probe, gamma) / probe)

    burn = int(experiment["burn_in_frames"])
    values = candidate.offset_stops[burn:]
    iid_values = iid_gamma_stops[burn:]
    candidate_jumps = np.abs(np.diff(values))
    iid_jumps = np.abs(np.diff(iid_values))
    cutoff = float(experiment["low_frequency_cutoff_cycles_per_frame"])
    candidate_lf = _low_frequency_power_fraction(values, cutoff)
    iid_lf = _low_frequency_power_fraction(iid_values, cutoff)

    quarter = values.size // 4
    candidate_variance_ratio = float(np.var(values[-quarter:]) / np.var(values[:quarter]))
    residual = (
        candidate.offset_stops[1:]
        - (1.0 - profile.theta) * candidate.offset_stops[:-1]
    ) / (profile.stationary_sigma_stops * np.sqrt(2.0 * profile.theta))
    path_length = int(np.sqrt(residual.size))
    ensemble_count = residual.size // path_length
    used = path_length * ensemble_count
    walks = np.cumsum(
        float(experiment["random_walk_scale_stops"])
        * residual[:used].reshape(ensemble_count, path_length),
        axis=1,
    )
    early_index = max(0, path_length // 4 - 1)
    random_walk_variance_ratio = float(
        np.var(walks[:, -1]) / np.var(walks[:, early_index])
    )
    zero_probe = np.linspace(0.0, 4.0, 257, dtype=np.float64)
    zero_identity = np.array_equal(apply_temporal_exposure(zero_probe, 0.0), zero_probe)
    finite = bool(
        np.all(np.isfinite(candidate.offset_stops))
        and np.all(np.isfinite(candidate.multiplier))
    )
    within_bound = bool(
        np.max(np.abs(candidate.offset_stops)) <= profile.maximum_absolute_stops
    )
    clip_fraction = float(np.mean(candidate.clipped))
    measurements = {
        "replay_byte_exact": replay_exact,
        "partition_byte_exact": partition_exact,
        "all_values_finite": finite,
        "all_offsets_within_bound": within_bound,
        "clip_fraction": clip_fraction,
        "stationary_mean_stops": float(np.mean(values)),
        "stationary_lag1": float(np.corrcoef(values[:-1], values[1:])[0, 1]),
        "stationary_std_over_profile_sigma": float(
            np.std(values) / profile.stationary_sigma_stops
        ),
        "candidate_p95_absolute_jump_stops": float(np.quantile(candidate_jumps, 0.95)),
        "iid_gamma_p95_absolute_jump_stops": float(np.quantile(iid_jumps, 0.95)),
        "candidate_to_iid_p95_jump_ratio": float(
            np.quantile(candidate_jumps, 0.95) / np.quantile(iid_jumps, 0.95)
        ),
        "candidate_low_frequency_power_fraction": candidate_lf,
        "iid_gamma_low_frequency_power_fraction": iid_lf,
        "candidate_to_iid_low_frequency_power_ratio": candidate_lf / iid_lf,
        "late_to_early_candidate_variance_ratio": candidate_variance_ratio,
        "random_walk_late_to_early_ensemble_variance_ratio": random_walk_variance_ratio,
        "random_walk_ensemble_shape": [ensemble_count, path_length],
        "zero_offset_layer_exposure_identity_exact": zero_identity,
        "display_rgb_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "replay_byte_exact": replay_exact is gates["replay_byte_exact"],
        "partition_byte_exact": partition_exact is gates["partition_byte_exact"],
        "all_values_finite": finite is gates["all_values_finite"],
        "all_offsets_within_bound": within_bound is gates["all_offsets_within_bound"],
        "maximum_clip_fraction": clip_fraction <= gates["maximum_clip_fraction"],
        "maximum_absolute_stationary_mean_stops": abs(measurements["stationary_mean_stops"])
        <= gates["maximum_absolute_stationary_mean_stops"],
        "minimum_stationary_lag1": measurements["stationary_lag1"]
        >= gates["minimum_stationary_lag1"],
        "maximum_stationary_lag1": measurements["stationary_lag1"]
        <= gates["maximum_stationary_lag1"],
        "minimum_stationary_std_over_profile_sigma": measurements[
            "stationary_std_over_profile_sigma"
        ]
        >= gates["minimum_stationary_std_over_profile_sigma"],
        "maximum_stationary_std_over_profile_sigma": measurements[
            "stationary_std_over_profile_sigma"
        ]
        <= gates["maximum_stationary_std_over_profile_sigma"],
        "maximum_candidate_to_iid_p95_jump_ratio": measurements[
            "candidate_to_iid_p95_jump_ratio"
        ]
        <= gates["maximum_candidate_to_iid_p95_jump_ratio"],
        "minimum_candidate_to_iid_low_frequency_power_ratio": measurements[
            "candidate_to_iid_low_frequency_power_ratio"
        ]
        >= gates["minimum_candidate_to_iid_low_frequency_power_ratio"],
        "maximum_late_to_early_candidate_variance_ratio": measurements[
            "late_to_early_candidate_variance_ratio"
        ]
        <= gates["maximum_late_to_early_candidate_variance_ratio"],
        "minimum_random_walk_late_to_early_ensemble_variance_ratio": measurements[
            "random_walk_late_to_early_ensemble_variance_ratio"
        ]
        >= gates["minimum_random_walk_late_to_early_ensemble_variance_ratio"],
        "zero_offset_layer_exposure_identity_exact": zero_identity
        is gates["zero_offset_layer_exposure_identity_exact"],
        "display_rgb_transform_count_zero": gates["display_rgb_transform_count_zero"]
        is True,
    }
    if set(gate_results) != set(gates):
        raise TemporalExposureAuditError("P9E gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9e_temporal_exposure_flicker_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9e_temporal_exposure_flicker_v1.json"),
        "frame_count": frame_count,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_bytes = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(stable_bytes).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
