"""Frozen U6.P3E Standard compiler comparison against the P3D reference."""

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
    apply_compiled_backing_return_row_tiled,
    apply_reference_backing_return,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)


P3D_SCHEMA = "neuro_film.u6_p3d_backing_return_reference_contract.v1"
P3E_SCHEMA = "neuro_film.u6_p3e_backing_return_standard_contract.v1"


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


def evaluate_compiled_backing_return(
    parent_contract: dict[str, Any], challenger_contract: dict[str, Any]
) -> dict[str, Any]:
    if parent_contract.get("schema") != P3D_SCHEMA:
        raise ValueError("invalid P3D parent contract")
    if challenger_contract.get("schema") != P3E_SCHEMA:
        raise ValueError("invalid P3E challenger contract")
    profile = backing_return_profile_from_contract(parent_contract)
    compiled = compile_backing_return_profile(profile)
    witnesses = challenger_contract["witnesses"]
    gates = challenger_contract["automatic_gates"]

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
    outputs: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, float]] = {}
    direct_retention = float("inf")
    for name, values in cases.items():
        reference = apply_reference_backing_return(
            _array(values, profile.pixel_pitch_um), profile
        ).values
        source32 = values.astype(np.float32)
        candidate = apply_compiled_backing_return(
            _array(source32, profile.pixel_pitch_um), compiled
        ).values
        difference = candidate.astype(np.float64) - reference
        errors[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
        }
        direct_retention = min(
            direct_retention,
            float(np.min(candidate - source32)),
        )
        outputs[name] = candidate

    repeat = apply_compiled_backing_return(
        _array(random.astype(np.float32), profile.pixel_pitch_um), compiled
    ).values
    partition_exact: dict[str, bool] = {}
    for tile_rows in witnesses["partition_rows"]:
        tiled = apply_compiled_backing_return_row_tiled(
            _array(random.astype(np.float32), profile.pixel_pitch_um),
            compiled,
            tile_rows=int(tile_rows),
        ).values
        partition_exact[str(tile_rows)] = np.array_equal(outputs["random"], tiled)

    impulse_reference = apply_reference_backing_return(
        _array(impulse, profile.pixel_pitch_um), profile
    ).values
    candidate_energy = np.sum(outputs["impulse"], axis=(0, 1), dtype=np.float64)
    reference_energy = np.sum(impulse_reference, axis=(0, 1), dtype=np.float64)
    scales = [float(value) for value in witnesses["exposure_scales"]]
    linearity_error = 0.0
    base = random.astype(np.float32)
    base_output = outputs["random"]
    for scale in scales:
        scaled = apply_compiled_backing_return(
            _array(np.asarray(base * np.float32(scale), dtype=np.float32), profile.pixel_pitch_um),
            compiled,
        ).values
        expected = np.asarray(base_output * np.float32(scale), dtype=np.float32)
        linearity_error = max(
            linearity_error,
            float(np.max(np.abs(scaled.astype(np.float64) - expected))),
        )

    zero_profile_contract = json.loads(json.dumps(parent_contract))
    for component in zero_profile_contract["components"]:
        component["return_fraction_rgb"] = [0.0, 0.0, 0.0]
    zero_compiled = compile_backing_return_profile(
        backing_return_profile_from_contract(zero_profile_contract)
    )
    zero_output = apply_compiled_backing_return(
        _array(base, profile.pixel_pitch_um), zero_compiled
    ).values
    kernel_sum_error = max(
        abs(float(np.sum(kernel.weights, dtype=np.float64)) - 1.0)
        for kernel in compiled.kernels
    )
    metrics = {
        "case_errors": errors,
        "reference_max_abs_error": max(item["max_abs"] for item in errors.values()),
        "reference_mean_abs_error": max(item["mean_abs"] for item in errors.values()),
        "candidate_impulse_energy_rgb": candidate_energy.tolist(),
        "reference_impulse_energy_rgb": reference_energy.tolist(),
        "returned_energy_abs_error": float(
            np.max(np.abs(candidate_energy - reference_energy))
        ),
        "minimum_output": min(float(np.min(value)) for value in outputs.values()),
        "minimum_direct_increment": direct_retention,
        "compiled_kernel_sum_abs_error": kernel_sum_error,
        "linearity_max_abs_error": linearity_error,
        "required_halo_pixels": compiled.required_halo,
        "zero_return_identity_exact": np.array_equal(base, zero_output),
        "repeat_float32_exact": np.array_equal(outputs["random"], repeat),
        "partition_reconstruction_exact": partition_exact,
    }
    decisions = {
        "reference_max": metrics["reference_max_abs_error"]
        <= float(gates["reference_max_abs_error"]),
        "reference_mean": metrics["reference_mean_abs_error"]
        <= float(gates["reference_mean_abs_error"]),
        "returned_energy": metrics["returned_energy_abs_error"]
        <= float(gates["returned_energy_abs_error"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "direct_retention": metrics["minimum_direct_increment"]
        >= float(gates["minimum_direct_retention"]) - 1.0,
        "kernel_sum": metrics["compiled_kernel_sum_abs_error"]
        <= float(gates["compiled_kernel_sum_abs_error"]),
        "linearity": metrics["linearity_max_abs_error"]
        <= float(gates["linearity_max_abs_error"]),
        "zero_return": bool(metrics["zero_return_identity_exact"]),
        "repeat": bool(metrics["repeat_float32_exact"]),
        "partition": all(partition_exact.values()),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3e_backing_return_standard_report.v1",
        "node": challenger_contract["node"],
        "claim_ceiling": challenger_contract["claim_ceiling"],
        "parent_profile_sha256": profile.profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": challenger_contract["branch_rule"]["pass" if passed else "fail"],
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
