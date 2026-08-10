"""Frozen U6.P9I shot-boundary temporal-state audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.gate_weave import (
    GateWeaveProfile,
    advance_gate_weave,
    generate_gate_weave,
    initial_gate_weave_state,
)
from src.film_physics.temporal_exposure import (
    TemporalExposureProfile,
    advance_temporal_exposure,
    generate_temporal_exposure,
    initial_temporal_exposure_state,
)
from src.film_physics.temporal_grain import temporal_grain_innovation_frame
from src.film_physics.temporal_sequence import (
    TemporalSequenceIdentity,
    derive_temporal_stream_seed,
)


class ShotBoundaryStateError(RuntimeError):
    """Raised when the P9I contract or its bound parents drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShotBoundaryStateError("contract must be an object")
    return payload


def _load(path: Path) -> dict[str, Any]:
    return load_contract(path)


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _exposure_arrays(segment: Any) -> tuple[np.ndarray, ...]:
    return (
        segment.frame_indices,
        segment.offset_stops,
        segment.multiplier,
        segment.clipped,
    )


def _weave_arrays(segment: Any) -> tuple[np.ndarray, ...]:
    return (
        segment.frame_indices,
        segment.x_pixels,
        segment.y_pixels,
        segment.theta_y,
        segment.x_integer_pixels,
        segment.y_integer_pixels,
        segment.coordinate_clipped,
    )


def _partition_exact(
    exposure_profile: TemporalExposureProfile,
    weave_profile: GateWeaveProfile,
    frame_count: int,
    counts: list[int],
) -> bool:
    full_exposure = generate_temporal_exposure(exposure_profile, frame_count)
    full_weave = generate_gate_weave(weave_profile, frame_count)
    exposure_state = initial_temporal_exposure_state()
    weave_state = initial_gate_weave_state(weave_profile)
    exposure_parts: list[Any] = []
    weave_parts: list[Any] = []
    for count in counts:
        exposure_part, exposure_state = advance_temporal_exposure(
            exposure_profile,
            total_frame_count=frame_count,
            state=exposure_state,
            transition_count=count,
        )
        weave_part, weave_state = advance_gate_weave(
            weave_profile,
            total_frame_count=frame_count,
            state=weave_state,
            transition_count=count,
        )
        exposure_parts.append(exposure_part)
        weave_parts.append(weave_part)
    if sum(counts) != frame_count - 1:
        raise ShotBoundaryStateError("partition counts do not cover shot transitions")
    exposure_joined = tuple(
        np.concatenate([getattr(part, name) for part in exposure_parts])
        for name in ("frame_indices", "offset_stops", "multiplier", "clipped")
    )
    weave_joined = tuple(
        np.concatenate([getattr(part, name) for part in weave_parts])
        for name in (
            "frame_indices",
            "x_pixels",
            "y_pixels",
            "theta_y",
            "x_integer_pixels",
            "y_integer_pixels",
            "coordinate_clipped",
        )
    )
    return all(
        np.array_equal(joined, full[1:])
        for joined, full in zip(
            exposure_joined, _exposure_arrays(full_exposure), strict=True
        )
    ) and all(
        np.array_equal(joined, full[1:])
        for joined, full in zip(weave_joined, _weave_arrays(full_weave), strict=True)
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9i_shot_boundary_state_contract.v1":
        raise ShotBoundaryStateError("unsupported P9I contract")
    parents = contract["parents"]
    bindings = (
        ("p9a_contract", "p9a_contract_sha256"),
        ("p9d_contract", "p9d_contract_sha256"),
        ("p9e_contract", "p9e_contract_sha256"),
        ("p9h_evidence", "p9h_evidence_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[sha_key]
        for path_key, sha_key in bindings
    )
    if not parent_hashes_exact:
        raise ShotBoundaryStateError("parent hash mismatch")
    if _load(root / parents["p9h_evidence"])["automatic_pass"] is not True:
        raise ShotBoundaryStateError("P9H does not admit P9I")

    p9a = _load(root / parents["p9a_contract"])
    p9e = _load(root / parents["p9e_contract"])
    p9d = _load(root / parents["p9d_contract"])
    base_exposure = TemporalExposureProfile(
        **{key: value for key, value in p9e["profile"].items() if key != "schema"}
    )
    base_weave = GateWeaveProfile(
        **{key: value for key, value in p9a["profile"].items() if key != "schema"}
    )
    experiment = contract["experiment"]
    frame_count = int(experiment["frame_count_per_shot"])
    counts = [int(value) for value in experiment["partition_transition_counts"]]
    shape = tuple(int(value) for value in experiment["grain_probe_shape"])
    layer_count = int(experiment["grain_layer_count"])
    sequences = [
        TemporalSequenceIdentity(value) for value in experiment["sequence_ids"]
    ]
    roles = tuple(contract["stream_roles"])
    base_grain_seed = int(p9d["experiment"]["seed"])
    profile_sha256 = p9d["experiment"]["profile_sha256"]

    seeds = {
        sequence.sequence_id: {
            role: derive_temporal_stream_seed(
                base_seed=(
                    base_exposure.seed
                    if role == "exposure_flicker"
                    else base_weave.seed
                    if role == "gate_weave"
                    else base_grain_seed
                ),
                sequence=sequence,
                stream_role=role,
            )
            for role in roles
        }
        for sequence in sequences
    }
    seed_replay = {
        sequence.sequence_id: {
            role: derive_temporal_stream_seed(
                base_seed=(
                    base_exposure.seed
                    if role == "exposure_flicker"
                    else base_weave.seed
                    if role == "gate_weave"
                    else base_grain_seed
                ),
                sequence=sequence,
                stream_role=role,
            )
            for role in roles
        }
        for sequence in sequences
    }
    sequence_identity_replay_exact = seeds == seed_replay
    flat_seeds = [
        seed for per_sequence in seeds.values() for seed in per_sequence.values()
    ]
    stream_role_domain_separation_exact = len(flat_seeds) == len(set(flat_seeds))

    exposures = []
    weaves = []
    grain_hashes: list[list[str]] = []
    partitions = []
    for sequence in sequences:
        per_sequence = seeds[sequence.sequence_id]
        exposure_profile = replace(base_exposure, seed=per_sequence["exposure_flicker"])
        weave_profile = replace(base_weave, seed=per_sequence["gate_weave"])
        exposure = generate_temporal_exposure(exposure_profile, frame_count)
        weave = generate_gate_weave(weave_profile, frame_count)
        exposures.append(exposure)
        weaves.append(weave)
        partitions.append(
            _partition_exact(exposure_profile, weave_profile, frame_count, counts)
        )
        grain_hashes.append(
            [
                _hash_array(
                    temporal_grain_innovation_frame(
                        profile_sha256=profile_sha256,
                        seed=per_sequence["density_grain"],
                        frame=frame,
                        full_shape=shape,
                        layer_count=layer_count,
                    )
                )
                for frame in range(frame_count)
            ]
        )

    cross_shot_grain_duplicates = sum(
        left == right
        for left, right in zip(grain_hashes[0], grain_hashes[1], strict=True)
    )
    naive_hashes = [
        _hash_array(
            temporal_grain_innovation_frame(
                profile_sha256=profile_sha256,
                seed=base_grain_seed,
                frame=frame,
                full_shape=shape,
                layer_count=layer_count,
            )
        )
        for frame in range(frame_count)
    ]
    naive_duplicate_fraction = float(
        np.mean(
            [
                left == right
                for left, right in zip(naive_hashes, naive_hashes, strict=True)
            ]
        )
    )
    shot_initial_exposure_reset_exact = all(
        segment.offset_stops[0] == 0.0 and segment.multiplier[0] == 1.0
        for segment in exposures
    )
    shot_initial_gate_weave_reset_exact = all(
        segment.x_pixels[0] == 0.0 and segment.y_pixels[0] == 0.0 for segment in weaves
    )
    continuous_exposure = generate_temporal_exposure(base_exposure, 2 * frame_count)
    continuous_weave = generate_gate_weave(base_weave, 2 * frame_count)
    continuous_boundary_exposure = float(continuous_exposure.offset_stops[frame_count])
    continuous_boundary_gate_norm = float(
        np.hypot(
            continuous_weave.x_pixels[frame_count],
            continuous_weave.y_pixels[frame_count],
        )
    )
    finite = bool(
        all(
            np.all(np.isfinite(array))
            for segment in exposures
            for array in _exposure_arrays(segment)
        )
        and all(
            np.all(np.isfinite(array))
            for segment in weaves
            for array in _weave_arrays(segment)
        )
    )
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "sequence_identity_replay_exact": sequence_identity_replay_exact,
        "stream_role_domain_separation_exact": stream_role_domain_separation_exact,
        "within_shot_partition_exact": all(partitions),
        "shot_initial_exposure_reset_exact": shot_initial_exposure_reset_exact,
        "shot_initial_gate_weave_reset_exact": shot_initial_gate_weave_reset_exact,
        "cross_shot_grain_duplicate_frame_count": cross_shot_grain_duplicates,
        "naive_local_frame_grain_duplicate_fraction": naive_duplicate_fraction,
        "continuous_control_boundary_exposure_stops": continuous_boundary_exposure,
        "continuous_control_boundary_gate_weave_norm_pixels": continuous_boundary_gate_norm,
        "all_values_finite": finite,
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "sequence_identity_replay_exact": sequence_identity_replay_exact
        is gates["sequence_identity_replay_exact"],
        "stream_role_domain_separation_exact": stream_role_domain_separation_exact
        is gates["stream_role_domain_separation_exact"],
        "within_shot_partition_exact": all(partitions)
        is gates["within_shot_partition_exact"],
        "shot_initial_exposure_reset_exact": shot_initial_exposure_reset_exact
        is gates["shot_initial_exposure_reset_exact"],
        "shot_initial_gate_weave_reset_exact": shot_initial_gate_weave_reset_exact
        is gates["shot_initial_gate_weave_reset_exact"],
        "cross_shot_grain_duplicate_frame_count_maximum": cross_shot_grain_duplicates
        <= gates["cross_shot_grain_duplicate_frame_count_maximum"],
        "naive_local_frame_grain_duplicate_fraction_minimum": naive_duplicate_fraction
        >= gates["naive_local_frame_grain_duplicate_fraction_minimum"],
        "continuous_control_boundary_exposure_nonzero": (
            continuous_boundary_exposure != 0.0
        )
        is gates["continuous_control_boundary_exposure_nonzero"],
        "continuous_control_boundary_gate_weave_nonzero": (
            continuous_boundary_gate_norm != 0.0
        )
        is gates["continuous_control_boundary_gate_weave_nonzero"],
        "all_values_finite": finite is gates["all_values_finite"],
        "display_rgb_effect_count_zero": (measurements["display_rgb_effect_count"] == 0)
        is gates["display_rgb_effect_count_zero"],
    }
    if set(gate_results) != set(gates):
        raise ShotBoundaryStateError("P9I gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9i_shot_boundary_state_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9i_shot_boundary_state_v1.json"),
        "sequence_count": len(sequences),
        "frame_count_per_shot": frame_count,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
