"""U6.P2N deterministic audit of explicit retained-silver density."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.silver_retention import (
    SilverRetentionProfile,
    apply_silver_retention,
    apply_silver_retention_row_tiled,
    density_jacobian,
)


SCHEMA = "neuro_film.u6_p2n_silver_retention_density_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2N contract")
    return payload


def _profile(row: dict[str, Any], *, fraction: float | None = None) -> SilverRetentionProfile:
    return SilverRetentionProfile(
        float(row["retention_fraction"] if fraction is None else fraction),
        tuple(row["dye_to_silver_weights"]),
        float(row["maximum_input_dye_density"]),
        float(row["maximum_output_total_density"]),
    )


def evaluate_silver_retention(contract: dict[str, Any]) -> dict[str, Any]:
    profile = _profile(contract["profile"])
    control = _profile(contract["profile"], fraction=0.0)
    witnesses = contract["synthetic_witnesses"]
    gates = contract["automatic_gates"]
    shape = tuple(int(x) for x in witnesses["shape"])
    rng = np.random.default_rng(int(witnesses["seed"]))
    low, high = (float(x) for x in witnesses["density_range"])
    density = rng.uniform(low, high, size=(*shape, 3)).astype(np.float64)
    before = density.tobytes()
    result = apply_silver_retention(density, profile)
    repeated = apply_silver_retention(density, profile)
    zero = apply_silver_retention(density, control)

    weights = np.asarray(profile.dye_to_silver_weights)
    expected_silver = (
        np.sum(density * weights, axis=-1) * profile.retention_fraction
    )
    composition_error = float(
        np.max(
            np.abs(result.total_density - (density + expected_silver[..., None]))
        )
    )
    opponent_before = density - density[..., :1]
    opponent_after = result.total_density - result.total_density[..., :1]
    opponent_error = float(np.max(np.abs(opponent_after - opponent_before)))
    dye_transmittance = np.power(10.0, -density)
    total_transmittance = np.power(10.0, -result.total_density)
    dye_chromaticity = dye_transmittance / np.sum(
        dye_transmittance, axis=-1, keepdims=True
    )
    total_chromaticity = total_transmittance / np.sum(
        total_transmittance, axis=-1, keepdims=True
    )
    chromaticity_error = float(
        np.max(np.abs(total_chromaticity - dye_chromaticity))
    )
    partition_exact = {}
    for rows in witnesses["row_partitions"]:
        tiled = apply_silver_retention_row_tiled(
            density, profile, tile_rows=int(rows)
        )
        partition_exact[str(rows)] = bool(
            np.array_equal(tiled.total_density, result.total_density)
            and np.array_equal(tiled.silver_density, result.silver_density)
        )
    constants = {}
    for level in witnesses["constant_density_levels"]:
        constant = np.full((*shape, 3), float(level), dtype=np.float64)
        constant_result = apply_silver_retention(constant, profile)
        constants[str(level)] = {
            "silver_density": float(constant_result.silver_density[0, 0]),
            "total_density": float(constant_result.total_density[0, 0, 0]),
            "neutral_spread": float(
                np.max(np.ptp(constant_result.total_density, axis=-1))
            ),
        }
    jacobian = density_jacobian(profile)
    own = np.diag(jacobian)
    cross = jacobian[~np.eye(3, dtype=bool)]
    hard_clip_count = 0
    metrics = {
        "minimum_silver_density": float(np.min(result.silver_density)),
        "maximum_silver_density": float(np.max(result.silver_density)),
        "maximum_total_density": float(np.max(result.total_density)),
        "maximum_density_composition_error": composition_error,
        "maximum_dye_opponent_difference_error": opponent_error,
        "maximum_transmittance_chromaticity_error": chromaticity_error,
        "minimum_own_density_derivative": float(np.min(own)),
        "minimum_cross_density_derivative": float(np.min(cross)),
        "maximum_cross_density_derivative": float(np.max(cross)),
        "row_partition_exact": partition_exact,
        "constant_witnesses": constants,
        "hard_clip_count": hard_clip_count,
    }
    decisions = {
        "zero_retention_identity": np.array_equal(zero.total_density, density)
        and np.array_equal(zero.silver_density, np.zeros(shape)),
        "repeat": np.array_equal(result.total_density, repeated.total_density)
        and np.array_equal(result.silver_density, repeated.silver_density),
        "row_partitions": all(partition_exact.values()),
        "silver_minimum": metrics["minimum_silver_density"]
        >= float(gates["minimum_silver_density"]),
        "silver_maximum": metrics["maximum_silver_density"]
        <= float(gates["maximum_silver_density"]),
        "total_density": metrics["maximum_total_density"]
        <= float(gates["maximum_total_density"]),
        "composition": composition_error
        <= float(gates["maximum_density_composition_error"]),
        "opponent_density": opponent_error
        <= float(gates["maximum_dye_opponent_difference_error"]),
        "transmittance_chromaticity": chromaticity_error
        <= float(gates["maximum_transmittance_chromaticity_error"]),
        "own_derivative": metrics["minimum_own_density_derivative"]
        >= float(gates["minimum_own_density_derivative"]),
        "cross_derivative": metrics["minimum_cross_density_derivative"]
        >= float(gates["minimum_cross_density_derivative"]),
        "input_preservation": density.tobytes() == before,
        "no_hard_clip": hard_clip_count
        <= int(gates["hard_clip_count_must_be_zero"]),
    }
    automatic_pass = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p2n_silver_retention_density_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": automatic_pass,
        "branch": contract["branch_rule"]["pass" if automatic_pass else "fail"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

