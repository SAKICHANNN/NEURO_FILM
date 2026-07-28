"""Deterministic U6.P1A synthetic evaluation."""

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
)
from src.film_physics.reference_scatter import (
    apply_reference_scatter,
    gaussian_kernel_2d,
    profile_from_contract,
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "neuro_film.u6_p1_reference_scatter_simulator_contract.v1"
    ):
        raise ValueError("unsupported U6.P1A contract")
    return payload


def _layer_image(values: np.ndarray, pixel_pitch_um: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.asarray(values, dtype=np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pixel_pitch_um),
    )


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_reference_scatter(contract: dict[str, Any]) -> dict[str, Any]:
    profile = profile_from_contract(contract)
    witnesses = contract["synthetic_witnesses"]
    gates = contract["automatic_gates"]
    impulse_shape = tuple(int(value) for value in witnesses["impulse_shape"])
    edge_shape = tuple(int(value) for value in witnesses["edge_shape"])
    if impulse_shape[-1] != 3 or edge_shape[-1] != 3:
        raise ValueError("synthetic witness shapes must end in three channels")

    impulse = np.zeros(impulse_shape, dtype=np.float64)
    center_y, center_x = impulse_shape[0] // 2, impulse_shape[1] // 2
    impulse[center_y, center_x, :] = 1.0
    impulse_result = apply_reference_scatter(
        _layer_image(impulse, profile.pixel_pitch_um), profile
    ).values

    kernel_errors = [
        abs(float(np.sum(gaussian_kernel_2d(component, profile.scale))) - 1.0)
        for component in profile.components
    ]
    impulse_energy = np.sum(impulse_result, axis=(0, 1), dtype=np.float64)
    impulse_energy_error = np.abs(impulse_energy - 1.0)

    linearity_errors = []
    for scale in witnesses["exposure_scales"]:
        scaled = apply_reference_scatter(
            _layer_image(impulse * float(scale), profile.pixel_pitch_um), profile
        ).values
        linearity_errors.append(float(np.max(np.abs(scaled - impulse_result * scale))))

    edge = np.zeros(edge_shape, dtype=np.float64)
    edge[:, edge_shape[1] // 2 :, :] = 1.0
    edge_result = apply_reference_scatter(
        _layer_image(edge, profile.pixel_pitch_um), profile
    ).values

    radius_px = int(
        round(float(witnesses["far_halo_radius_um"]) / profile.pixel_pitch_um)
    )
    sample_x = center_x + radius_px
    far_red = float(impulse_result[center_y, sample_x, 0])
    far_blue = float(impulse_result[center_y, sample_x, 2])
    far_ratio = far_red / far_blue

    metrics = {
        "kernel_sum_abs_error_max": max(kernel_errors),
        "impulse_energy_rgb": impulse_energy.tolist(),
        "impulse_energy_abs_error_max": float(np.max(impulse_energy_error)),
        "minimum_output": float(
            min(np.min(impulse_result), np.min(edge_result))
        ),
        "linearity_max_abs_error": max(linearity_errors),
        "far_halo_radius_um": float(witnesses["far_halo_radius_um"]),
        "far_halo_red_over_blue": far_ratio,
        "edge_midpoint_rgb": edge_result[
            edge_shape[0] // 2, edge_shape[1] // 2, :
        ].tolist(),
    }
    decisions = {
        "kernel_normalization": metrics["kernel_sum_abs_error_max"]
        <= float(gates["kernel_sum_abs_error_max"]),
        "impulse_energy": metrics["impulse_energy_abs_error_max"]
        <= float(gates["impulse_energy_abs_error_max"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "linearity": metrics["linearity_max_abs_error"]
        <= float(gates["linearity_max_abs_error"]),
        "far_halo_order": metrics["far_halo_red_over_blue"]
        >= float(gates["far_halo_red_over_blue_min"]),
    }
    core = {
        "schema": "neuro_film.u6_p1_reference_scatter_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": (
            contract["branch_rule"]["pass"]
            if all(decisions.values())
            else contract["branch_rule"]["fail"]
        ),
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()
