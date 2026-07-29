"""Frozen U6.P3H functional evaluation of the FFT backing-return compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_backing_return,
    apply_fft_backing_return,
    apply_fft_backing_return_row_tiled,
    apply_reference_backing_return,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)
from src.film_physics.fft_backing_return import (
    _apply_fft_backing_return_values,
)


P3D_SCHEMA = "neuro_film.u6_p3d_backing_return_reference_contract.v1"
P3H_SCHEMA = "neuro_film.u6_p3h_fft_backing_return_compiler_contract.v1"


def load_json(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported contract: expected {schema}")
    return payload


def _array(values: np.ndarray, pitch: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pitch),
    )


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_fft_backing_return(
    parent_contract: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    if parent_contract.get("schema") != P3D_SCHEMA:
        raise ValueError("invalid P3D parent contract")
    if contract.get("schema") != P3H_SCHEMA:
        raise ValueError("invalid P3H contract")
    reference_profile = backing_return_profile_from_contract(parent_contract)
    compiled = compile_backing_return_profile(reference_profile)
    if (
        compiled.parent_profile_sha256
        != contract["parents"]["p3d_profile_sha256"]
    ):
        raise ValueError("P3D profile identity drift")
    witnesses = contract["witnesses"]
    gates = contract["automatic_gates"]

    impulse_shape = tuple(witnesses["impulse_shape"])
    impulse = np.zeros(impulse_shape, dtype=np.float64)
    impulse[impulse_shape[0] // 2, impulse_shape[1] // 2, :] = 1.0
    edge_shape = tuple(witnesses["edge_shape"])
    edge = np.zeros(edge_shape, dtype=np.float64)
    edge[:, edge_shape[1] // 2 :, :] = 1.0
    random_shape = tuple(witnesses["random_shape"])
    random = np.random.default_rng(int(witnesses["random_seed"])).random(
        random_shape, dtype=np.float64
    )

    cases = {"impulse": impulse, "edge": edge, "random": random}
    direct_errors: dict[str, dict[str, float]] = {}
    reference_errors: dict[str, dict[str, float]] = {}
    outputs: dict[str, np.ndarray] = {}
    minimum_raw_returned = 0.0
    minimum_direct_increment = float("inf")
    for name, values in cases.items():
        source32 = values.astype(np.float32)
        source_array = _array(source32, compiled.pixel_pitch_um)
        direct = apply_compiled_backing_return(source_array, compiled).values
        candidate, raw_minimum = _apply_fft_backing_return_values(
            source32, compiled
        )
        reference = apply_reference_backing_return(
            _array(values, compiled.pixel_pitch_um), reference_profile
        ).values
        direct_difference = candidate.astype(np.float64) - direct.astype(
            np.float64
        )
        reference_difference = candidate.astype(np.float64) - reference
        direct_errors[name] = {
            "max_abs": float(np.max(np.abs(direct_difference))),
            "mean_abs": float(np.mean(np.abs(direct_difference))),
        }
        reference_errors[name] = {
            "max_abs": float(np.max(np.abs(reference_difference))),
            "mean_abs": float(np.mean(np.abs(reference_difference))),
        }
        minimum_raw_returned = min(minimum_raw_returned, raw_minimum)
        minimum_direct_increment = min(
            minimum_direct_increment, float(np.min(candidate - source32))
        )
        outputs[name] = candidate

    repeat = apply_fft_backing_return(
        _array(random.astype(np.float32), compiled.pixel_pitch_um), compiled
    ).values
    partition_errors: dict[str, dict[str, float]] = {}
    for tile_rows in witnesses["partition_rows"]:
        tiled = apply_fft_backing_return_row_tiled(
            _array(random.astype(np.float32), compiled.pixel_pitch_um),
            compiled,
            tile_rows=int(tile_rows),
        ).values
        difference = tiled.astype(np.float64) - outputs["random"].astype(
            np.float64
        )
        partition_errors[str(tile_rows)] = {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
        }

    impulse_reference = apply_reference_backing_return(
        _array(impulse, compiled.pixel_pitch_um), reference_profile
    ).values
    candidate_energy = np.sum(outputs["impulse"], axis=(0, 1), dtype=np.float64)
    reference_energy = np.sum(
        impulse_reference, axis=(0, 1), dtype=np.float64
    )
    zero_contract = json.loads(json.dumps(parent_contract))
    for component in zero_contract["components"]:
        component["return_fraction_rgb"] = [0.0, 0.0, 0.0]
    zero_compiled = compile_backing_return_profile(
        backing_return_profile_from_contract(zero_contract)
    )
    base = random.astype(np.float32)
    zero_output = apply_fft_backing_return(
        _array(base, compiled.pixel_pitch_um), zero_compiled
    ).values
    metrics = {
        "direct_compiler_case_errors": direct_errors,
        "float64_reference_case_errors": reference_errors,
        "direct_compiler_max_abs_error": max(
            value["max_abs"] for value in direct_errors.values()
        ),
        "direct_compiler_mean_abs_error": max(
            value["mean_abs"] for value in direct_errors.values()
        ),
        "float64_reference_max_abs_error": max(
            value["max_abs"] for value in reference_errors.values()
        ),
        "returned_energy_abs_error": float(
            np.max(np.abs(candidate_energy - reference_energy))
        ),
        "partition_errors": partition_errors,
        "partition_max_abs_error": max(
            value["max_abs"] for value in partition_errors.values()
        ),
        "partition_mean_abs_error": max(
            value["mean_abs"] for value in partition_errors.values()
        ),
        "minimum_output": min(float(np.min(value)) for value in outputs.values()),
        "minimum_direct_increment": minimum_direct_increment,
        "minimum_raw_returned_value": minimum_raw_returned,
        "repeat_float32_exact": np.array_equal(outputs["random"], repeat),
        "zero_return_identity_exact": np.array_equal(base, zero_output),
        "required_halo_pixels": compiled.required_halo,
    }
    decisions = {
        "direct_max": metrics["direct_compiler_max_abs_error"]
        <= float(gates["direct_compiler_max_abs_error"]),
        "direct_mean": metrics["direct_compiler_mean_abs_error"]
        <= float(gates["direct_compiler_mean_abs_error"]),
        "reference_max": metrics["float64_reference_max_abs_error"]
        <= float(gates["float64_reference_max_abs_error"]),
        "energy": metrics["returned_energy_abs_error"]
        <= float(gates["returned_energy_abs_error"]),
        "partition_max": metrics["partition_max_abs_error"]
        <= float(gates["partition_max_abs_error"]),
        "partition_mean": metrics["partition_mean_abs_error"]
        <= float(gates["partition_mean_abs_error"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "direct_retention": metrics["minimum_direct_increment"]
        >= float(gates["minimum_direct_increment"]),
        "roundoff_floor": metrics["minimum_raw_returned_value"]
        >= float(gates["minimum_raw_returned_value"]),
        "repeat": bool(metrics["repeat_float32_exact"]),
        "zero_return": bool(metrics["zero_return_identity_exact"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3h_fft_backing_return_compiler_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "parent_profile_sha256": compiled.parent_profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
