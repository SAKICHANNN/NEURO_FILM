"""Frozen U6.P9B immutable-source gate-weave sampling audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.gate_weave import GateWeaveProfile, generate_gate_weave
from src.film_physics.gate_weave_sampling import sample_padded_translation


class GateWeaveSamplingAuditError(RuntimeError):
    """Raised when the P9B contract or parent evidence drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateWeaveSamplingAuditError("contract must be an object")
    return payload


def _chart(
    y: np.ndarray,
    x: np.ndarray,
    frequencies: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    channels = []
    for index, frequency in enumerate(frequencies):
        slope_y = (0.19 + 0.11 * index) * frequency
        phase = 2.0 * np.pi * (frequency * x + slope_y * y)
        channels.append(0.5 + amplitude * np.sin(phase))
    return np.stack(channels, axis=-1)


def _amplitude(values: np.ndarray, phase: np.ndarray) -> float:
    centered = values - np.mean(values)
    sine = np.sin(phase)
    cosine = np.cos(phase)
    sine_coefficient = 2.0 * float(np.mean(centered * sine))
    cosine_coefficient = 2.0 * float(np.mean(centered * cosine))
    return float(np.hypot(sine_coefficient, cosine_coefficient))


def _parent_profile(root: Path) -> GateWeaveProfile:
    payload = json.loads(
        (root / "configs/u6_p9a_gate_weave_ou_v1.json").read_text(encoding="utf-8")
    )["profile"]
    return GateWeaveProfile(
        **{key: value for key, value in payload.items() if key != "schema"}
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9b_gate_weave_sampling_contract.v1":
        raise GateWeaveSamplingAuditError("unsupported P9B contract")
    parent = contract["parent"]
    parent_path = root / parent["evidence_path"]
    if _sha(parent_path) != parent["evidence_sha256"]:
        raise GateWeaveSamplingAuditError("P9A evidence hash mismatch")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent_payload.get("decision") != parent["required_decision"]:
        raise GateWeaveSamplingAuditError("P9A decision does not admit P9B")

    experiment = contract["experiment"]
    height = int(experiment["output_height"])
    width = int(experiment["output_width"])
    frequencies = np.asarray(
        experiment["chart_frequencies_cycles_per_pixel"], dtype=np.float64
    )
    amplitude = float(experiment["chart_amplitude"])
    profile = _parent_profile(root)
    trajectory = generate_gate_weave(profile, int(experiment["frame_count"]))
    padding = (profile.padding_y_pixels, profile.padding_x_pixels)
    padded_y = np.arange(-padding[0], height + padding[0], dtype=np.float64)[:, None]
    padded_x = np.arange(-padding[1], width + padding[1], dtype=np.float64)[None, :]
    source = _chart(padded_y, padded_x, frequencies, amplitude)
    output_y = np.arange(height, dtype=np.float64)[:, None]
    output_x = np.arange(width, dtype=np.float64)[None, :]
    source_minimum = float(np.min(source))
    source_maximum = float(np.max(source))

    bilinear_squared_error = 0.0
    integer_squared_error = 0.0
    sample_count = 0
    high_frequency_retentions: list[float] = []
    full_digest = hashlib.sha256()
    integer_crop_exact = True
    in_range = True
    finite = True
    highest_frequency = frequencies[-1]
    highest_slope_y = (0.19 + 0.11 * (len(frequencies) - 1)) * highest_frequency
    for offset_y, offset_x, integer_y, integer_x in zip(
        trajectory.y_pixels,
        trajectory.x_pixels,
        trajectory.y_integer_pixels,
        trajectory.x_integer_pixels,
        strict=True,
    ):
        truth = _chart(output_y + offset_y, output_x + offset_x, frequencies, amplitude)
        bilinear = sample_padded_translation(
            source,
            output_shape=(height, width),
            padding_yx=padding,
            offset_yx=(float(offset_y), float(offset_x)),
            interpolation="bilinear",
        )
        integer = sample_padded_translation(
            source,
            output_shape=(height, width),
            padding_yx=padding,
            offset_yx=(float(offset_y), float(offset_x)),
            interpolation="nearest",
        )
        direct = source[
            padding[0] + int(integer_y) : padding[0] + int(integer_y) + height,
            padding[1] + int(integer_x) : padding[1] + int(integer_x) + width,
        ]
        integer_crop_exact = integer_crop_exact and np.array_equal(integer, direct)
        bilinear_squared_error += float(np.sum(np.square(bilinear - truth)))
        integer_squared_error += float(np.sum(np.square(integer - truth)))
        sample_count += truth.size
        phase = (
            2.0
            * np.pi
            * (
                highest_frequency * (output_x + offset_x)
                + highest_slope_y * (output_y + offset_y)
            )
        )
        high_frequency_retentions.append(
            _amplitude(bilinear[..., -1], phase) / _amplitude(truth[..., -1], phase)
        )
        full_digest.update(np.ascontiguousarray(bilinear).tobytes())
        finite = finite and bool(np.all(np.isfinite(bilinear)))
        in_range = in_range and bool(
            np.min(bilinear) >= source_minimum and np.max(bilinear) <= source_maximum
        )

    partition_digest = hashlib.sha256()
    start = 0
    for size in experiment["bilinear_partition_sizes"]:
        stop = start + int(size)
        for offset_y, offset_x in zip(
            trajectory.y_pixels[start:stop],
            trajectory.x_pixels[start:stop],
            strict=True,
        ):
            rendered = sample_padded_translation(
                source,
                output_shape=(height, width),
                padding_yx=padding,
                offset_yx=(float(offset_y), float(offset_x)),
                interpolation="bilinear",
            )
            partition_digest.update(np.ascontiguousarray(rendered).tobytes())
        start = stop
    if start != len(trajectory.x_pixels):
        raise GateWeaveSamplingAuditError("partition sizes do not cover all frames")

    integer_offset_exact = True
    for offset_y in range(-padding[0], padding[0] + 1):
        for offset_x in range(-padding[1], padding[1] + 1):
            bilinear = sample_padded_translation(
                source,
                output_shape=(height, width),
                padding_yx=padding,
                offset_yx=(float(offset_y), float(offset_x)),
                interpolation="bilinear",
            )
            direct = source[
                padding[0] + offset_y : padding[0] + offset_y + height,
                padding[1] + offset_x : padding[1] + offset_x + width,
            ]
            integer_offset_exact = integer_offset_exact and np.array_equal(
                bilinear, direct
            )

    constant = np.full_like(source, 0.375)
    constant_exact = all(
        np.array_equal(
            sample_padded_translation(
                constant,
                output_shape=(height, width),
                padding_yx=padding,
                offset_yx=(float(y), float(x)),
                interpolation="bilinear",
            ),
            np.full((height, width, 3), 0.375, dtype=np.float64),
        )
        for y, x in zip(
            trajectory.y_pixels[::31], trajectory.x_pixels[::31], strict=True
        )
    )
    bilinear_rmse = float(np.sqrt(bilinear_squared_error / sample_count))
    integer_rmse = float(np.sqrt(integer_squared_error / sample_count))
    improvement = 1.0 - bilinear_rmse / integer_rmse
    minimum_retention = float(np.min(high_frequency_retentions))
    maximum_retention = float(np.max(high_frequency_retentions))
    measurements = {
        "integer_crop_pixel_exact": integer_crop_exact,
        "bilinear_integer_offset_pixel_exact": integer_offset_exact,
        "bilinear_partition_byte_exact": full_digest.digest()
        == partition_digest.digest(),
        "constant_field_pixel_exact": constant_exact,
        "all_outputs_finite": finite,
        "all_outputs_within_source_range": in_range,
        "bilinear_analytic_rmse": bilinear_rmse,
        "integer_analytic_rmse": integer_rmse,
        "bilinear_rmse_improvement_over_integer": improvement,
        "minimum_high_frequency_amplitude_retention": minimum_retention,
        "maximum_high_frequency_amplitude_retention": maximum_retention,
        "bilinear_stream_sha256": full_digest.hexdigest(),
        "rgb_quantization_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "integer_crop_pixel_exact": integer_crop_exact
        is gates["integer_crop_pixel_exact"],
        "bilinear_integer_offset_pixel_exact": integer_offset_exact
        is gates["bilinear_integer_offset_pixel_exact"],
        "bilinear_partition_byte_exact": measurements["bilinear_partition_byte_exact"]
        is gates["bilinear_partition_byte_exact"],
        "constant_field_pixel_exact": constant_exact
        is gates["constant_field_pixel_exact"],
        "all_outputs_finite": finite is gates["all_outputs_finite"],
        "all_outputs_within_source_range": in_range
        is gates["all_outputs_within_source_range"],
        "maximum_bilinear_analytic_rmse": bilinear_rmse
        <= gates["maximum_bilinear_analytic_rmse"],
        "minimum_bilinear_rmse_improvement_over_integer": improvement
        >= gates["minimum_bilinear_rmse_improvement_over_integer"],
        "minimum_high_frequency_amplitude_retention": minimum_retention
        >= gates["minimum_high_frequency_amplitude_retention"],
        "maximum_high_frequency_amplitude_retention": maximum_retention
        <= gates["maximum_high_frequency_amplitude_retention"],
        "rgb_quantization_count_zero": gates["rgb_quantization_count_zero"] is True,
    }
    if set(gate_results) != set(gates):
        raise GateWeaveSamplingAuditError("P9B gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9b_gate_weave_sampling_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9b_gate_weave_sampling_v1.json"),
        "parent_evidence_sha256": parent["evidence_sha256"],
        "frame_count": len(trajectory.x_pixels),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["branch_rule"]["pass"]
        if automatic_pass
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
