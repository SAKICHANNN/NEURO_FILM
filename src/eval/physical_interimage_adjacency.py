"""Frozen U6.P5F synthetic evaluation of cross-layer adjacency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.interimage_adjacency import (
    InterimageAdjacencyProfile,
    apply_interimage_adjacency,
    apply_interimage_adjacency_row_tiled,
    interimage_adjacency_profile_from_contract,
    required_interimage_adjacency_halo,
)


SCHEMA = "neuro_film.u6_p5f_interimage_adjacency_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5F contract")
    return payload


def _vertical_edge(
    shape: tuple[int, int],
    *,
    low: float,
    high: float,
    channel: int | None,
    background: float,
) -> np.ndarray:
    height, width = shape
    density = np.full((height, width, 3), background, dtype=np.float64)
    if channel is None:
        density[:, : width // 2, :] = low
        density[:, width // 2 :, :] = high
    else:
        density[:, : width // 2, channel] = low
        density[:, width // 2 :, channel] = high
    return density


def _opponent_peak_gradient(values: np.ndarray) -> float:
    opponent = values - np.mean(values, axis=-1, keepdims=True)
    gradient = np.diff(opponent, axis=1)
    return float(np.max(np.linalg.norm(gradient, axis=-1)))


def _hard_clip_fraction(
    source: np.ndarray,
    output: np.ndarray,
    profile: InterimageAdjacencyProfile,
) -> float:
    black = np.asarray(profile.black_reference_density).reshape(1, 1, 3)
    white = np.asarray(profile.white_reference_density).reshape(1, 1, 3)
    interior = (source > black) & (source < white)
    hard = interior & ((output == black) | (output == white))
    return float(np.count_nonzero(hard) / hard.size)


def evaluate_interimage_adjacency(contract: dict[str, Any]) -> dict[str, Any]:
    profile = interimage_adjacency_profile_from_contract(contract)
    witnesses = contract["synthetic_witnesses"]
    gates = contract["automatic_gates"]
    shape = tuple(int(value) for value in witnesses["shape"])

    constant_exact = True
    for level in witnesses["constant_density_levels"]:
        density = np.full((*shape, 3), float(level), dtype=np.float64)
        constant_exact = constant_exact and np.array_equal(
            apply_interimage_adjacency(density, profile), density
        )

    neutral = _vertical_edge(
        shape,
        low=float(witnesses["neutral_edge_low_high"][0]),
        high=float(witnesses["neutral_edge_low_high"][1]),
        channel=None,
        background=0.0,
    )
    neutral_output = apply_interimage_adjacency(neutral, profile)
    neutral_error = float(np.max(np.abs(neutral_output - neutral)))

    channel_rows = []
    material_correction = 0.0
    for channel in range(3):
        density = _vertical_edge(
            shape,
            low=float(witnesses["single_channel_edge_low_high"][0]),
            high=float(witnesses["single_channel_edge_low_high"][1]),
            channel=channel,
            background=float(witnesses["single_channel_neutral_background"]),
        )
        output = apply_interimage_adjacency(density, profile)
        baseline_gradient = _opponent_peak_gradient(density)
        output_gradient = _opponent_peak_gradient(output)
        gain = output_gradient / baseline_gradient - 1.0
        correction = float(np.max(np.abs(output - density)))
        material_correction = max(material_correction, correction)
        channel_rows.append(
            {
                "channel": channel,
                "baseline_peak_opponent_gradient": baseline_gradient,
                "output_peak_opponent_gradient": output_gradient,
                "gain_fraction": gain,
                "peak_absolute_density_correction": correction,
            }
        )

    rng = np.random.default_rng(int(witnesses["random_seed"]))
    low, high = (float(value) for value in witnesses["random_density_range"])
    random_density = rng.uniform(low, high, size=(*shape, 3)).astype(np.float64)
    random_output = apply_interimage_adjacency(random_density, profile)
    repeated_output = apply_interimage_adjacency(random_density, profile)
    correction = random_output - random_density
    transmittance_delta = np.power(10.0, -random_output) - np.power(
        10.0, -random_density
    )
    peak_density_correction = float(np.max(np.abs(correction)))
    peak_transmittance_correction = float(np.max(np.abs(transmittance_delta)))
    material_correction = max(material_correction, peak_density_correction)

    row_partition_exact = {}
    for rows in witnesses["row_partitions"]:
        row_partition_exact[str(rows)] = np.array_equal(
            random_output,
            apply_interimage_adjacency_row_tiled(
                random_density,
                profile,
                tile_rows=int(rows),
            ),
        )

    impulse = np.full(
        (*shape, 3),
        float(witnesses["impulse_background"]),
        dtype=np.float64,
    )
    centre_y, centre_x = shape[0] // 2, shape[1] // 2
    impulse[centre_y, centre_x, 0] += float(witnesses["impulse_delta"])
    impulse_output = apply_interimage_adjacency(impulse, profile)
    impulse_correction = np.max(np.abs(impulse_output - impulse), axis=-1)
    halo = required_interimage_adjacency_halo(profile)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    remote = (np.abs(yy - centre_y) > halo) | (np.abs(xx - centre_x) > halo)
    remote_impulse_correction = float(np.max(impulse_correction[remote]))

    black = np.asarray(profile.black_reference_density).reshape(1, 1, 3)
    white = np.asarray(profile.white_reference_density).reshape(1, 1, 3)
    density_domain_violation = max(
        0.0,
        float(np.max(black - random_output)),
        float(np.max(random_output - white)),
    )
    hard_clip_fraction = _hard_clip_fraction(
        random_density, random_output, profile
    )
    zero_profile = InterimageAdjacencyProfile(
        **{
            **profile.__dict__,
            "coupling_matrix": (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        }
    )
    zero_coupling_exact = np.array_equal(
        apply_interimage_adjacency(random_density, zero_profile),
        random_density,
    )

    decisions = {
        "repeat": np.array_equal(random_output, repeated_output),
        "constant_fields": constant_exact,
        "neutral_axis": neutral_error
        <= float(gates["maximum_neutral_axis_abs_error"]),
        "density_domain": density_domain_violation
        <= float(gates["maximum_density_domain_violation"]),
        "density_bound": peak_density_correction
        <= float(gates["maximum_absolute_density_correction"]),
        "transmittance_bound": peak_transmittance_correction
        <= float(gates["maximum_absolute_transmittance_correction"]),
        "material_response": material_correction
        >= float(gates["minimum_material_absolute_density_correction"]),
        "all_channel_opponent_gain": all(
            row["gain_fraction"]
            >= float(
                gates[
                    "minimum_single_channel_peak_opponent_gradient_gain_fraction"
                ]
            )
            for row in channel_rows
        ),
        "row_partitions": all(row_partition_exact.values()),
        "finite_support": remote_impulse_correction
        <= float(gates["maximum_remote_impulse_correction"]),
        "no_hard_clip": hard_clip_fraction
        <= float(gates["hard_clip_fraction_must_be_zero"]),
        "zero_coupling": zero_coupling_exact,
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p5f_interimage_adjacency_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "candidate": contract["candidate"]["name"],
        "metrics": {
            "halo_rows": halo,
            "neutral_axis_maximum_absolute_error": neutral_error,
            "peak_absolute_density_correction": peak_density_correction,
            "peak_absolute_transmittance_correction": (
                peak_transmittance_correction
            ),
            "material_peak_absolute_density_correction": material_correction,
            "remote_impulse_maximum_absolute_correction": (
                remote_impulse_correction
            ),
            "hard_clip_fraction": hard_clip_fraction,
            "density_domain_violation": density_domain_violation,
            "single_channel_edges": channel_rows,
            "row_partition_exact": row_partition_exact,
        },
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_interimage_adjacency",
    "load_contract",
    "write_report",
]
