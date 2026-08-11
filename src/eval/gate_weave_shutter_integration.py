"""Frozen U6.P9L within-frame gate-weave integration audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.gate_weave import GateWeaveProfile, generate_gate_weave
from src.film_physics.gate_weave_sampling import (
    integrate_padded_translation,
    sample_padded_translation,
)


class GateWeaveShutterAuditError(RuntimeError):
    """Raised when the P9L contract or bound implementation drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateWeaveShutterAuditError("contract must be an object")
    return payload


def _chart(
    y: np.ndarray, x: np.ndarray, frequencies: np.ndarray, amplitude: float
) -> np.ndarray:
    channels = []
    for index, frequency in enumerate(frequencies):
        slope_y = (0.19 + 0.11 * index) * frequency
        channels.append(
            0.5
            + amplitude
            * np.sin(2.0 * np.pi * (frequency * x + slope_y * y))
        )
    return np.stack(channels, axis=-1)


def _analytic_average(
    y: np.ndarray,
    x: np.ndarray,
    frequencies: np.ndarray,
    amplitude: float,
    start_yx: tuple[float, float],
    end_yx: tuple[float, float],
) -> np.ndarray:
    channels = []
    for index, frequency in enumerate(frequencies):
        slope_y = (0.19 + 0.11 * index) * frequency
        phase0 = 2.0 * np.pi * (
            frequency * (x + start_yx[1]) + slope_y * (y + start_yx[0])
        )
        phase1 = 2.0 * np.pi * (
            frequency * (x + end_yx[1]) + slope_y * (y + end_yx[0])
        )
        midpoint = 0.5 * (phase0 + phase1)
        half_span = 0.5 * (phase1 - phase0)
        channels.append(
            0.5 + amplitude * np.sin(midpoint) * np.sinc(half_span / np.pi)
        )
    return np.stack(channels, axis=-1)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p9l_gate_weave_shutter_integration_contract.v1"
    ):
        raise GateWeaveShutterAuditError("unsupported P9L contract")
    parents = contract["parents"]
    bindings = (
        ("p9a_contract", "p9a_contract_sha256"),
        ("p9a_evidence", "p9a_evidence_sha256"),
        ("p9b_evidence", "p9b_evidence_sha256"),
        ("sampling_core", "sampling_core_sha256"),
    )
    hashes_exact = all(
        _sha(root / parents[path_key]) == parents[hash_key]
        for path_key, hash_key in bindings
    )
    if not hashes_exact:
        raise GateWeaveShutterAuditError("parent or sampling-core hash mismatch")
    p9a_evidence = load_contract(root / parents["p9a_evidence"])
    p9a_pass = p9a_evidence.get("automatic_pass")
    if p9a_pass is None:
        p9a_pass = p9a_evidence.get("results", {}).get("automatic_pass")
    if p9a_pass is not True:
        raise GateWeaveShutterAuditError("P9A does not admit P9L")
    p9b_evidence = load_contract(root / parents["p9b_evidence"])
    p9b_pass = p9b_evidence.get("automatic_pass")
    if p9b_pass is None:
        p9b_pass = p9b_evidence.get("results", {}).get("automatic_pass")
    if p9b_pass is not True:
        raise GateWeaveShutterAuditError("P9B does not admit P9L")

    p9a = load_contract(root / parents["p9a_contract"])
    profile = GateWeaveProfile(
        **{key: value for key, value in p9a["profile"].items() if key != "schema"}
    )
    trajectory = generate_gate_weave(profile, int(p9a["experiment"]["frame_count"]))
    experiment = contract["experiment"]
    frame_count = int(experiment["frame_count"])
    start_index = int(experiment["trajectory_frame_start"])
    indices = list(range(start_index, start_index + frame_count))
    if start_index < 1 or indices[-1] >= len(trajectory.frame_indices):
        raise GateWeaveShutterAuditError("trajectory window is outside P9A")
    output_shape = (
        int(experiment["output_height"]),
        int(experiment["output_width"]),
    )
    padding = (profile.padding_y_pixels, profile.padding_x_pixels)
    frequencies = np.asarray(
        experiment["chart_frequencies_cycles_per_pixel"], dtype=np.float64
    )
    amplitude = float(experiment["chart_amplitude"])
    candidate_samples = int(experiment["candidate_midpoint_sample_count"])
    reference_samples = int(experiment["reference_midpoint_sample_count"])
    padded_y = np.arange(
        -padding[0], output_shape[0] + padding[0], dtype=np.float64
    )[:, None]
    padded_x = np.arange(
        -padding[1], output_shape[1] + padding[1], dtype=np.float64
    )[None, :]
    output_y = np.arange(output_shape[0], dtype=np.float64)[:, None]
    output_x = np.arange(output_shape[1], dtype=np.float64)[None, :]
    source = _chart(padded_y, padded_x, frequencies, amplitude)
    source_minimum = float(np.min(source))
    source_maximum = float(np.max(source))

    def offsets(index: int) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (
                float(trajectory.y_pixels[index - 1]),
                float(trajectory.x_pixels[index - 1]),
            ),
            (
                float(trajectory.y_pixels[index]),
                float(trajectory.x_pixels[index]),
            ),
        )

    def render(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start_yx, end_yx = offsets(index)
        candidate = integrate_padded_translation(
            source,
            output_shape=output_shape,
            padding_yx=padding,
            start_offset_yx=start_yx,
            end_offset_yx=end_yx,
            sample_count=candidate_samples,
        )
        dense = integrate_padded_translation(
            source,
            output_shape=output_shape,
            padding_yx=padding,
            start_offset_yx=start_yx,
            end_offset_yx=end_yx,
            sample_count=reference_samples,
        )
        instantaneous = sample_padded_translation(
            source,
            output_shape=output_shape,
            padding_yx=padding,
            offset_yx=end_yx,
            interpolation="bilinear",
        )
        truth = _analytic_average(
            output_y, output_x, frequencies, amplitude, start_yx, end_yx
        )
        return candidate, dense, instantaneous, truth

    def render_candidate(index: int) -> np.ndarray:
        start_yx, end_yx = offsets(index)
        return integrate_padded_translation(
            source,
            output_shape=output_shape,
            padding_yx=padding,
            start_offset_yx=start_yx,
            end_offset_yx=end_yx,
            sample_count=candidate_samples,
        )

    full = {index: render(index) for index in indices}
    replay = {index: render_candidate(index) for index in indices}
    reverse = {index: render_candidate(index) for index in reversed(indices)}
    replay_exact = all(
        np.array_equal(full[index][0], replay[index]) for index in indices
    )
    reverse_exact = all(
        np.array_equal(full[index][0], reverse[index]) for index in indices
    )
    partitioned: dict[int, np.ndarray] = {}
    cursor = 0
    for count in experiment["frame_partition_counts"]:
        selected = indices[cursor : cursor + int(count)]
        partitioned.update({index: render_candidate(index) for index in selected})
        cursor += int(count)
    if cursor != frame_count:
        raise GateWeaveShutterAuditError("partitions do not cover frame window")
    partition_exact = all(
        np.array_equal(full[index][0], partitioned[index]) for index in indices
    )

    candidate_error = 0.0
    instantaneous_error = 0.0
    dense_error = 0.0
    element_count = 0
    finite = True
    in_range = True
    for candidate, dense, instantaneous, truth in full.values():
        candidate_error += float(np.sum(np.square(candidate - truth)))
        instantaneous_error += float(np.sum(np.square(instantaneous - truth)))
        dense_error += float(np.sum(np.square(candidate - dense)))
        element_count += candidate.size
        finite = finite and bool(np.all(np.isfinite(candidate)))
        in_range = in_range and bool(
            np.min(candidate) >= source_minimum
            and np.max(candidate) <= source_maximum
        )
    candidate_rmse = float(np.sqrt(candidate_error / element_count))
    instantaneous_rmse = float(np.sqrt(instantaneous_error / element_count))
    dense_rmse = float(np.sqrt(dense_error / element_count))
    improvement = 1.0 - candidate_rmse / instantaneous_rmse

    probe_index = indices[len(indices) // 2]
    _, probe_end = offsets(probe_index)
    static_integrated = integrate_padded_translation(
        source,
        output_shape=output_shape,
        padding_yx=padding,
        start_offset_yx=probe_end,
        end_offset_yx=probe_end,
        sample_count=candidate_samples,
    )
    static_sampled = sample_padded_translation(
        source,
        output_shape=output_shape,
        padding_yx=padding,
        offset_yx=probe_end,
        interpolation="bilinear",
    )
    static_exact = np.array_equal(static_integrated, static_sampled)
    constant = np.full_like(source, 0.375)
    first_start, first_end = offsets(indices[0])
    constant_exact = np.array_equal(
        integrate_padded_translation(
            constant,
            output_shape=output_shape,
            padding_yx=padding,
            start_offset_yx=first_start,
            end_offset_yx=first_end,
            sample_count=candidate_samples,
        ),
        np.full((*output_shape, 3), 0.375, dtype=np.float64),
    )
    measurements = {
        "parent_and_core_hashes_exact": hashes_exact,
        "candidate_replay_byte_exact": replay_exact,
        "candidate_reverse_frame_order_byte_exact": reverse_exact,
        "candidate_partition_byte_exact": partition_exact,
        "static_trajectory_pixel_exact": static_exact,
        "constant_field_pixel_exact": constant_exact,
        "all_outputs_finite": finite,
        "all_outputs_within_source_range": in_range,
        "instantaneous_analytic_rmse": instantaneous_rmse,
        "candidate_analytic_rmse": candidate_rmse,
        "candidate_rmse_improvement_over_instantaneous": improvement,
        "candidate_dense_reference_rmse": dense_rmse,
        "rgb_quantization_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_and_core_hashes_exact": hashes_exact
        is gates["parent_and_core_hashes_exact"],
        "candidate_replay_byte_exact": replay_exact
        is gates["candidate_replay_byte_exact"],
        "candidate_reverse_frame_order_byte_exact": reverse_exact
        is gates["candidate_reverse_frame_order_byte_exact"],
        "candidate_partition_byte_exact": partition_exact
        is gates["candidate_partition_byte_exact"],
        "static_trajectory_pixel_exact": static_exact
        is gates["static_trajectory_pixel_exact"],
        "constant_field_pixel_exact": constant_exact
        is gates["constant_field_pixel_exact"],
        "all_outputs_finite": finite is gates["all_outputs_finite"],
        "all_outputs_within_source_range": in_range
        is gates["all_outputs_within_source_range"],
        "minimum_instantaneous_analytic_rmse": instantaneous_rmse
        >= gates["minimum_instantaneous_analytic_rmse"],
        "maximum_candidate_analytic_rmse": candidate_rmse
        <= gates["maximum_candidate_analytic_rmse"],
        "minimum_candidate_rmse_improvement_over_instantaneous": improvement
        >= gates["minimum_candidate_rmse_improvement_over_instantaneous"],
        "maximum_candidate_dense_reference_rmse": dense_rmse
        <= gates["maximum_candidate_dense_reference_rmse"],
        "rgb_quantization_count_zero": measurements["rgb_quantization_count"] == 0,
    }
    if set(gate_results) != set(gates):
        raise GateWeaveShutterAuditError("P9L gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9l_gate_weave_shutter_integration_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p9l_gate_weave_shutter_integration_v1.json"
        ),
        "frame_count": frame_count,
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
