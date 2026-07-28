"""Frozen U6.P3C pyramid comparison against the P1A float64 reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_compiled_scatter import P1_SCHEMA, P3_SCHEMA, load_json
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_pyramid_scatter,
    compile_pyramid_scatter_profile,
)
from src.film_physics.reference_scatter import (
    apply_reference_scatter,
    profile_from_contract,
)


PYRAMID_SCHEMA = "neuro_film.u6_p3_pyramid_scatter_challenger_contract.v1"


def _array(values: np.ndarray, pitch: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pitch),
    )


def evaluate_pyramid_scatter(
    parent_contract: dict[str, Any],
    direct_contract: dict[str, Any],
    pyramid_contract: dict[str, Any],
) -> dict[str, Any]:
    if parent_contract.get("schema") != P1_SCHEMA:
        raise ValueError("invalid P1 parent contract")
    if direct_contract.get("schema") != P3_SCHEMA:
        raise ValueError("invalid P3 direct contract")
    if pyramid_contract.get("schema") != PYRAMID_SCHEMA:
        raise ValueError("invalid P3 pyramid contract")
    reference_profile = profile_from_contract(parent_contract)
    candidate_config = pyramid_contract["candidate"]
    profile = compile_pyramid_scatter_profile(
        reference_profile,
        target_coarse_sigma_pixels=float(
            candidate_config["target_coarse_sigma_pixels"]
        ),
        minimum_pyramid_factor=int(candidate_config["minimum_pyramid_factor"]),
    )
    witnesses = direct_contract["witnesses"]
    impulse_shape = tuple(witnesses["impulse_shape"])
    impulse = np.zeros(impulse_shape, dtype=np.float64)
    center_y, center_x = impulse_shape[0] // 2, impulse_shape[1] // 2
    impulse[center_y, center_x, :] = 1.0
    edge_shape = tuple(witnesses["edge_shape"])
    edge = np.zeros(edge_shape, dtype=np.float64)
    edge[:, edge_shape[1] // 2 :, :] = 1.0
    random = np.random.default_rng(int(witnesses["random_seed"])).random(
        tuple(witnesses["random_shape"]), dtype=np.float64
    )
    cases = {"impulse": impulse, "edge": edge, "random": random}
    errors = {}
    outputs = {}
    for name, values in cases.items():
        reference = apply_reference_scatter(
            _array(values, reference_profile.pixel_pitch_um), reference_profile
        ).values
        candidate = apply_pyramid_scatter(
            _array(values.astype(np.float32), reference_profile.pixel_pitch_um),
            profile,
        ).values
        difference = candidate.astype(np.float64) - reference
        errors[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
        }
        outputs[name] = candidate
    repeat = apply_pyramid_scatter(
        _array(random.astype(np.float32), reference_profile.pixel_pitch_um), profile
    ).values
    radius_px = int(round(160.0 / reference_profile.pixel_pitch_um))
    far_red = float(outputs["impulse"][center_y, center_x + radius_px, 0])
    far_blue = float(outputs["impulse"][center_y, center_x + radius_px, 2])
    impulse_energy = np.sum(outputs["impulse"], axis=(0, 1), dtype=np.float64)
    metrics = {
        "case_errors": errors,
        "impulse_energy_rgb": impulse_energy.tolist(),
        "impulse_energy_abs_error": float(np.max(np.abs(impulse_energy - 1.0))),
        "minimum_output": min(float(np.min(value)) for value in outputs.values()),
        "far_halo_red_over_blue": far_red / far_blue,
        "repeat_float32_exact": np.array_equal(outputs["random"], repeat),
        "components": [
            {
                "component_id": component.component_id,
                "sigma_pixels": component.sigma_pixels,
                "pyramid_factor": component.pyramid_factor,
                "uses_pyramid": component.uses_pyramid,
            }
            for component in profile.components
        ],
    }
    gates = pyramid_contract["automatic_gates"]
    decisions = {
        "impulse_reference": errors["impulse"]["max_abs"]
        <= float(gates["impulse_reference_max_abs_error"]),
        "edge_reference": errors["edge"]["max_abs"]
        <= float(gates["edge_reference_max_abs_error"]),
        "random_reference_max": errors["random"]["max_abs"]
        <= float(gates["random_reference_max_abs_error"]),
        "random_reference_mean": errors["random"]["mean_abs"]
        <= float(gates["random_reference_mean_abs_error"]),
        "impulse_energy": metrics["impulse_energy_abs_error"]
        <= float(gates["impulse_energy_abs_error"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "far_halo_order": metrics["far_halo_red_over_blue"]
        >= float(gates["far_halo_red_over_blue_min"]),
        "repeat": bool(metrics["repeat_float32_exact"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3_pyramid_scatter_report.v1",
        "node": pyramid_contract["node"],
        "claim_ceiling": pyramid_contract["claim_ceiling"],
        "parent_profile_sha256": reference_profile.profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": pyramid_contract["branch_rule"][
            "pass" if passed else "numerical_fail"
        ],
    }
    stable = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
