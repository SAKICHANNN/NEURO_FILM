"""Frozen U6.P3A compiled-scatter comparison against U6.P1A."""

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
    apply_compiled_scatter,
    apply_compiled_scatter_row_tiled,
    compile_scatter_profile,
)
from src.film_physics.reference_scatter import (
    apply_reference_scatter,
    profile_from_contract,
)


P1_SCHEMA = "neuro_film.u6_p1_reference_scatter_simulator_contract.v1"
P3_SCHEMA = "neuro_film.u6_p3_compiled_scatter_challenger_contract.v1"


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
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def evaluate_compiled_scatter(
    parent_contract: dict[str, Any], challenger_contract: dict[str, Any]
) -> dict[str, Any]:
    if parent_contract.get("schema") != P1_SCHEMA:
        raise ValueError("invalid P1 parent contract")
    if challenger_contract.get("schema") != P3_SCHEMA:
        raise ValueError("invalid P3 challenger contract")
    profile = profile_from_contract(parent_contract)
    compiled = compile_scatter_profile(profile)
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
    errors = {}
    outputs = {}
    for name, values in cases.items():
        reference = apply_reference_scatter(
            _array(values, profile.pixel_pitch_um), profile
        ).values
        candidate = apply_compiled_scatter(
            _array(values.astype(np.float32), profile.pixel_pitch_um), compiled
        ).values
        difference = candidate.astype(np.float64) - reference
        errors[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
        }
        outputs[name] = candidate

    repeat = apply_compiled_scatter(
        _array(random.astype(np.float32), profile.pixel_pitch_um), compiled
    ).values
    repeat_exact = np.array_equal(outputs["random"], repeat)
    partition_exact = {}
    for tile_rows in witnesses["partition_rows"]:
        tiled = apply_compiled_scatter_row_tiled(
            _array(random.astype(np.float32), profile.pixel_pitch_um),
            compiled,
            tile_rows=int(tile_rows),
        ).values
        partition_exact[str(tile_rows)] = np.array_equal(outputs["random"], tiled)

    impulse_energy = np.sum(outputs["impulse"], axis=(0, 1), dtype=np.float64)
    kernel_sum_error = max(
        abs(float(np.sum(kernel.weights, dtype=np.float64)) - 1.0)
        for kernel in compiled.kernels
    )
    metrics = {
        "case_errors": errors,
        "reference_max_abs_error": max(item["max_abs"] for item in errors.values()),
        "reference_mean_abs_error": max(item["mean_abs"] for item in errors.values()),
        "impulse_energy_rgb": impulse_energy.tolist(),
        "impulse_energy_abs_error": float(np.max(np.abs(impulse_energy - 1.0))),
        "minimum_output": min(float(np.min(value)) for value in outputs.values()),
        "compiled_kernel_sum_abs_error": kernel_sum_error,
        "required_halo_pixels": compiled.required_halo,
        "repeat_float32_exact": repeat_exact,
        "partition_reconstruction_exact": partition_exact,
    }
    decisions = {
        "reference_max": metrics["reference_max_abs_error"]
        <= float(gates["reference_max_abs_error"]),
        "reference_mean": metrics["reference_mean_abs_error"]
        <= float(gates["reference_mean_abs_error"]),
        "impulse_energy": metrics["impulse_energy_abs_error"]
        <= float(gates["impulse_energy_abs_error"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "kernel_sum": metrics["compiled_kernel_sum_abs_error"]
        <= float(gates["compiled_kernel_sum_abs_error"]),
        "repeat": bool(metrics["repeat_float32_exact"]),
        "partition": all(partition_exact.values()),
    }
    core = {
        "schema": "neuro_film.u6_p3_compiled_scatter_report.v1",
        "node": challenger_contract["node"],
        "claim_ceiling": challenger_contract["claim_ceiling"],
        "parent_profile_sha256": profile.profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": challenger_contract["branch_rule"][
            "pass" if all(decisions.values()) else "fail"
        ],
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
