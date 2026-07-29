"""Deterministic U6.P3D backing-reflection reference evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics import (
    BackingReturnComponent,
    BackingReturnProfile,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_reference_backing_return,
    apply_reference_scatter,
    backing_return_kernel_2d,
    backing_return_profile_from_contract,
    profile_from_contract,
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "neuro_film.u6_p3d_backing_return_reference_contract.v1"
    ):
        raise ValueError("unsupported U6.P3D contract")
    return payload


def load_legacy_control(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "neuro_film.u6_p1_reference_scatter_simulator_contract.v1"
    ):
        raise ValueError("unsupported U6.P1A control contract")
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


def _zero_return_profile(profile: BackingReturnProfile) -> BackingReturnProfile:
    return BackingReturnProfile(
        pixel_pitch_um=profile.pixel_pitch_um,
        boundary_mode=profile.boundary_mode,
        components=tuple(
            BackingReturnComponent(
                component_id=component.component_id,
                sigma_um=component.sigma_um,
                cutoff_sigma=component.cutoff_sigma,
                return_fraction_rgb=(0.0, 0.0, 0.0),
                source_to_layer_coupling=component.source_to_layer_coupling,
            )
            for component in profile.components
        ),
    )


def evaluate_backing_return(
    contract: dict[str, Any],
    legacy_control_contract: dict[str, Any],
) -> dict[str, Any]:
    profile = backing_return_profile_from_contract(contract)
    legacy_profile = profile_from_contract(legacy_control_contract)
    if legacy_profile.scale != profile.scale:
        raise ValueError("legacy control and backing-return scale differ")
    witnesses = contract["synthetic_witnesses"]
    gates = contract["automatic_gates"]
    impulse_shape = tuple(int(value) for value in witnesses["impulse_shape"])
    edge_shape = tuple(int(value) for value in witnesses["edge_shape"])
    identity_shape = tuple(
        int(value) for value in witnesses["zero_return_identity_shape"]
    )
    if (
        impulse_shape[-1] != 3
        or edge_shape[-1] != 3
        or identity_shape[-1] != 3
    ):
        raise ValueError("synthetic witness shapes must end in three channels")

    center_y, center_x = impulse_shape[0] // 2, impulse_shape[1] // 2
    impulse = np.zeros(impulse_shape, dtype=np.float64)
    impulse[center_y, center_x, :] = 1.0
    impulse_result = apply_reference_backing_return(
        _layer_image(impulse, profile.pixel_pitch_um), profile
    ).values
    returned = impulse_result - impulse

    kernel_errors = [
        abs(
            float(np.sum(backing_return_kernel_2d(component, profile.scale)))
            - 1.0
        )
        for component in profile.components
    ]

    measured_returned_by_source: list[float] = []
    expected_returned_by_source: list[float] = []
    source_return_errors: list[float] = []
    for source_channel in range(3):
        source_impulse = np.zeros(impulse_shape, dtype=np.float64)
        source_impulse[center_y, center_x, source_channel] = 1.0
        source_result = apply_reference_backing_return(
            _layer_image(source_impulse, profile.pixel_pitch_um), profile
        ).values
        measured = float(np.sum(source_result - source_impulse, dtype=np.float64))
        expected = float(profile.maximum_return_fraction_by_source[source_channel])
        measured_returned_by_source.append(measured)
        expected_returned_by_source.append(expected)
        source_return_errors.append(abs(measured - expected))

    linearity_errors: list[float] = []
    for scale in witnesses["exposure_scales"]:
        scaled = apply_reference_backing_return(
            _layer_image(impulse * float(scale), profile.pixel_pitch_um), profile
        ).values
        linearity_errors.append(
            float(np.max(np.abs(scaled - impulse_result * float(scale))))
        )

    edge = np.zeros(edge_shape, dtype=np.float64)
    edge[:, edge_shape[1] // 2 :, :] = 1.0
    edge_result = apply_reference_backing_return(
        _layer_image(edge, profile.pixel_pitch_um), profile
    ).values

    radius_px = int(
        round(float(witnesses["far_halo_radius_um"]) / profile.pixel_pitch_um)
    )
    sample_x = center_x + radius_px
    far_red = float(returned[center_y, sample_x, 0])
    far_blue = float(returned[center_y, sample_x, 2])
    far_ratio = far_red / far_blue

    rng = np.random.default_rng(int(witnesses["random_seed"]))
    identity_input = rng.random(identity_shape, dtype=np.float64) * 4.0
    identity_output = apply_reference_backing_return(
        _layer_image(identity_input, profile.pixel_pitch_um),
        _zero_return_profile(profile),
    ).values
    identity_error = float(np.max(np.abs(identity_output - identity_input)))

    legacy_impulse = apply_reference_scatter(
        _layer_image(impulse, profile.pixel_pitch_um), legacy_profile
    ).values
    legacy_total_energy = float(np.sum(legacy_impulse, dtype=np.float64))
    additive_total_energy = float(np.sum(impulse_result, dtype=np.float64))
    additive_separation = additive_total_energy - legacy_total_energy

    returned_energy_epsilon = float(gates["returned_energy_bound_epsilon"])
    metrics = {
        "kernel_sum_abs_error_max": max(kernel_errors),
        "expected_returned_energy_by_source": expected_returned_by_source,
        "measured_returned_energy_by_source": measured_returned_by_source,
        "returned_energy_abs_error_max": max(source_return_errors),
        "returned_energy_bound_excess_max": max(
            measured - expected
            for measured, expected in zip(
                measured_returned_by_source,
                expected_returned_by_source,
                strict=True,
            )
        ),
        "minimum_output": float(min(np.min(impulse_result), np.min(edge_result))),
        "minimum_direct_retention": float(
            np.min(impulse_result[center_y, center_x, :])
        ),
        "linearity_max_abs_error": max(linearity_errors),
        "far_halo_radius_um": float(witnesses["far_halo_radius_um"]),
        "far_halo_red_over_blue": far_ratio,
        "zero_return_identity_max_abs_error": identity_error,
        "legacy_energy_redistribution_total": legacy_total_energy,
        "additive_backing_return_total": additive_total_energy,
        "additive_energy_above_legacy": additive_separation,
    }
    decisions = {
        "kernel_normalization": metrics["kernel_sum_abs_error_max"]
        <= float(gates["kernel_sum_abs_error_max"]),
        "returned_energy_exact": metrics["returned_energy_abs_error_max"]
        <= float(gates["returned_energy_abs_error_max"]),
        "returned_energy_bounded": metrics["returned_energy_bound_excess_max"]
        <= returned_energy_epsilon,
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "direct_retention": metrics["minimum_direct_retention"]
        >= float(gates["minimum_direct_retention"]),
        "linearity": metrics["linearity_max_abs_error"]
        <= float(gates["linearity_max_abs_error"]),
        "far_halo_order": metrics["far_halo_red_over_blue"]
        >= float(gates["far_halo_red_over_blue_min"]),
        "zero_return_identity": metrics["zero_return_identity_max_abs_error"]
        <= float(gates["zero_return_identity_max_abs_error"]),
        "distinct_from_energy_redistribution": metrics[
            "additive_energy_above_legacy"
        ]
        > returned_energy_epsilon,
    }
    core = {
        "schema": "neuro_film.u6_p3d_backing_return_reference_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "legacy_control_profile_sha256": legacy_profile.profile_sha256,
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
