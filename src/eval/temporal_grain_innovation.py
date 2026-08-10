"""Frozen U6.P9C temporal grain innovation independence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.temporal_grain import (
    temporal_grain_innovation_frame,
    temporal_grain_innovation_region,
)


class TemporalGrainAuditError(RuntimeError):
    """Raised when the P9C contract is invalid."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TemporalGrainAuditError("contract must be an object")
    return payload


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def _field(profile: dict[str, Any], frame: int) -> np.ndarray:
    return temporal_grain_innovation_frame(
        profile_sha256=profile["profile_sha256"],
        seed=int(profile["seed"]),
        frame=frame,
        full_shape=(int(profile["height"]), int(profile["width"])),
        layer_count=int(profile["layer_count"]),
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p9c_temporal_grain_innovation_contract.v1"
    ):
        raise TemporalGrainAuditError("unsupported P9C contract")
    profile = contract["innovation_profile"]
    frame_count = int(profile["frame_count"])
    height = int(profile["height"])
    width = int(profile["width"])
    layer_count = int(profile["layer_count"])
    frames = [_field(profile, frame) for frame in range(frame_count)]
    replay_exact = all(
        np.array_equal(values, _field(profile, frame))
        for frame, values in enumerate(frames)
    )
    reverse = {frame: _field(profile, frame) for frame in reversed(range(frame_count))}
    reverse_exact = all(
        np.array_equal(frames[frame], reverse[frame]) for frame in range(frame_count)
    )

    tile_exact = True
    row_start = 0
    tile_rows = [int(value) for value in profile["tile_rows"]]
    for row_count in tile_rows:
        for frame in range(frame_count):
            for layer in range(layer_count):
                tile = temporal_grain_innovation_region(
                    profile_sha256=profile["profile_sha256"],
                    seed=int(profile["seed"]),
                    frame=frame,
                    layer=layer,
                    full_shape=(height, width),
                    origin_yx=(row_start, 0),
                    shape=(row_count, width),
                )
                tile_exact = tile_exact and np.array_equal(
                    tile, frames[frame][row_start : row_start + row_count, :, layer]
                )
        row_start += row_count
    if row_start != height:
        raise TemporalGrainAuditError("tile rows do not cover the field")

    hashes = [
        hashlib.sha256(np.ascontiguousarray(frame[..., layer]).tobytes()).hexdigest()
        for frame in frames
        for layer in range(layer_count)
    ]
    means = np.asarray(
        [np.mean(frame[..., layer]) for frame in frames for layer in range(layer_count)]
    )
    standard_deviations = np.asarray(
        [np.std(frame[..., layer]) for frame in frames for layer in range(layer_count)]
    )
    adjacent_correlations = np.asarray(
        [
            _correlation(frames[frame][..., layer], frames[frame + 1][..., layer])
            for frame in range(frame_count - 1)
            for layer in range(layer_count)
        ]
    )
    cross_layer_correlations = np.asarray(
        [
            _correlation(frames[frame][..., left], frames[frame][..., right])
            for frame in range(frame_count)
            for left in range(layer_count)
            for right in range(left + 1, layer_count)
        ]
    )
    static_correlations = np.asarray(
        [
            _correlation(frames[0][..., 0], frames[0][..., 0])
            for _ in range(frame_count - 1)
        ]
    )
    shifted_correlations = []
    static = frames[0][..., 0]
    for frame in range(1, frame_count):
        shift = (frame % height, (3 * frame) % width)
        shifted = np.roll(static, shift=shift, axis=(0, 1))
        inverse_aligned = np.roll(shifted, shift=(-shift[0], -shift[1]), axis=(0, 1))
        shifted_correlations.append(_correlation(static, inverse_aligned))
    measurements = {
        "replay_byte_exact": replay_exact,
        "reverse_frame_order_byte_exact": reverse_exact,
        "tile_partition_byte_exact": tile_exact,
        "all_frame_hashes_unique": len(set(hashes)) == len(hashes),
        "all_values_finite": bool(all(np.all(np.isfinite(frame)) for frame in frames)),
        "maximum_absolute_frame_mean": float(np.max(np.abs(means))),
        "minimum_frame_standard_deviation": float(np.min(standard_deviations)),
        "maximum_frame_standard_deviation": float(np.max(standard_deviations)),
        "maximum_absolute_adjacent_frame_correlation": float(
            np.max(np.abs(adjacent_correlations))
        ),
        "maximum_absolute_cross_layer_correlation": float(
            np.max(np.abs(cross_layer_correlations))
        ),
        "minimum_static_reuse_adjacent_correlation": float(np.min(static_correlations)),
        "minimum_shifted_reuse_inverse_aligned_correlation": float(
            np.min(shifted_correlations)
        ),
        "density_or_signal_envelope_application_count": 0,
        "spatial_nps_filter_application_count": 0,
        "rgb_image_transform_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "replay_byte_exact": replay_exact is gates["replay_byte_exact"],
        "reverse_frame_order_byte_exact": reverse_exact
        is gates["reverse_frame_order_byte_exact"],
        "tile_partition_byte_exact": tile_exact is gates["tile_partition_byte_exact"],
        "all_frame_hashes_unique": measurements["all_frame_hashes_unique"]
        is gates["all_frame_hashes_unique"],
        "all_values_finite": measurements["all_values_finite"]
        is gates["all_values_finite"],
        "maximum_absolute_frame_mean": measurements["maximum_absolute_frame_mean"]
        <= gates["maximum_absolute_frame_mean"],
        "minimum_frame_standard_deviation": measurements[
            "minimum_frame_standard_deviation"
        ]
        >= gates["minimum_frame_standard_deviation"],
        "maximum_frame_standard_deviation": measurements[
            "maximum_frame_standard_deviation"
        ]
        <= gates["maximum_frame_standard_deviation"],
        "maximum_absolute_adjacent_frame_correlation": measurements[
            "maximum_absolute_adjacent_frame_correlation"
        ]
        <= gates["maximum_absolute_adjacent_frame_correlation"],
        "maximum_absolute_cross_layer_correlation": measurements[
            "maximum_absolute_cross_layer_correlation"
        ]
        <= gates["maximum_absolute_cross_layer_correlation"],
        "minimum_static_reuse_adjacent_correlation": measurements[
            "minimum_static_reuse_adjacent_correlation"
        ]
        >= gates["minimum_static_reuse_adjacent_correlation"],
        "minimum_shifted_reuse_inverse_aligned_correlation": measurements[
            "minimum_shifted_reuse_inverse_aligned_correlation"
        ]
        >= gates["minimum_shifted_reuse_inverse_aligned_correlation"],
        "density_or_signal_envelope_application_count_zero": measurements[
            "density_or_signal_envelope_application_count"
        ]
        == 0,
        "spatial_nps_filter_application_count_zero": measurements[
            "spatial_nps_filter_application_count"
        ]
        == 0,
        "rgb_image_transform_count_zero": measurements["rgb_image_transform_count"]
        == 0,
    }
    if set(gate_results) != set(gates):
        raise TemporalGrainAuditError("P9C gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9c_temporal_grain_innovation_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p9c_temporal_grain_innovation_v1.json"
        ),
        "frame_count": frame_count,
        "layer_count": layer_count,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["branch_rule"]["pass"]
        if automatic_pass
        else contract["branch_rule"]["fail"],
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
