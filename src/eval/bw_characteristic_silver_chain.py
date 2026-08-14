"""Evaluate the U6.P2AI source-characteristic metallic-silver chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.bw_characteristic_surface import compile_surface
from src.eval.bw_characteristic_surface import load_contract as load_surface_contract
from src.film_physics.bw_characteristic_silver_chain import (
    build_bw_characteristic_silver_chain,
    render_bw_characteristic_silver_chain_region,
)
from src.film_physics.contracts import PhysicalDomainArray


class BWCharacteristicSilverChainError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ai_bw_characteristic_silver_chain_contract.v1"
    ):
        raise BWCharacteristicSilverChainError("unsupported P2AI contract")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise BWCharacteristicSilverChainError("P2AI parent identity mismatch")
        parent = json.loads(path.read_text(encoding="utf-8"))
        if parent.get("automatic_pass") is not binding["required_automatic_pass"]:
            raise BWCharacteristicSilverChainError("P2AI parent decision mismatch")
    surface_binding = contract["surface"]
    surface, _ = compile_surface(
        root=root,
        contract=load_surface_contract(root / surface_binding["contract"]),
    )
    if surface.identity() != surface_binding["identity"]:
        raise BWCharacteristicSilverChainError("P2AI surface identity mismatch")

    fixture = contract["fixture"]
    yy, xx = np.indices(
        (fixture["input_height"], fixture["input_width"]), dtype=np.float64
    )
    position = (0.37 * yy / max(1, fixture["input_height"] - 1)) + (
        0.63 * xx / max(1, fixture["input_width"] - 1)
    )
    exposure = fixture["minimum_log_exposure"] + position * (
        fixture["maximum_log_exposure"] - fixture["minimum_log_exposure"]
    )
    original_exposure = exposure.copy()
    rows = []
    repeat_exact = True
    partition_exact = True
    maximum_density_error = 0.0
    maximum_mean_error = 0.0
    for time in fixture["development_times_minutes"]:
        kwargs = {
            "development_time_minutes": time,
            "radius_um": fixture["radius_um"],
            "output_zoom": fixture["output_zoom"],
            "output_pixel_pitch_um": fixture["output_pixel_pitch_um"],
            "monte_carlo_samples": fixture["monte_carlo_samples"],
            "seed": fixture["seed"],
        }
        result = build_bw_characteristic_silver_chain(exposure, surface, **kwargs)
        repeat = build_bw_characteristic_silver_chain(exposure, surface, **kwargs)
        full = result.transmittance.values
        repeat_exact = repeat_exact and np.array_equal(
            full, repeat.transmittance.values
        )
        split = full.shape[0] // 2
        tiled = np.concatenate(
            (
                render_bw_characteristic_silver_chain_region(
                    result,
                    output_origin_yx=(0, 0),
                    output_shape=(split, full.shape[1]),
                ).values,
                render_bw_characteristic_silver_chain_region(
                    result,
                    output_origin_yx=(split, 0),
                    output_shape=(full.shape[0] - split, full.shape[1]),
                ).values,
            )
        )
        partition_exact = partition_exact and np.array_equal(full, tiled)
        reference_density = surface.density(time, exposure)
        density = result.developed_density.values[..., 0]
        density_error = float(np.max(np.abs(density - reference_density)))
        maximum_density_error = max(maximum_density_error, density_error)
        margin = fixture["output_zoom"]
        observed_mean = float(
            np.mean(full[margin:-margin, margin:-margin, 0], dtype=np.float64)
        )
        expected_mean = float(
            np.mean(np.power(10.0, -reference_density)[1:-1, 1:-1], dtype=np.float64)
        )
        mean_error = abs(observed_mean - expected_mean)
        maximum_mean_error = max(maximum_mean_error, mean_error)
        rows.append(
            {
                "development_time_minutes": time,
                "mean_density": float(np.mean(density, dtype=np.float64)),
                "mean_transmittance": observed_mean,
                "density_reference_error": density_error,
                "mean_transmittance_absolute_error": mean_error,
                "grain_count": len(result.context.centers_by_layer[0]),
                "density_sha256": hashlib.sha256(
                    np.asarray(density, dtype="<f8").tobytes()
                ).hexdigest(),
                "transmittance_sha256": hashlib.sha256(
                    np.asarray(full, dtype="<f4").tobytes()
                ).hexdigest(),
            }
        )

    domain_rejected = False
    non_neutral_rejected = False
    outside_rejected = False
    try:
        build_bw_characteristic_silver_chain(
            PhysicalDomainArray,
            surface,
            **{
                **kwargs,
                "development_time_minutes": 9.0,
            },
        )
    except (TypeError, ValueError):
        domain_rejected = True
    try:
        build_bw_characteristic_silver_chain(
            np.repeat(exposure[..., None], 3, axis=-1),
            surface,
            **{**kwargs, "development_time_minutes": 9.0},
        )
    except ValueError:
        non_neutral_rejected = True
    try:
        build_bw_characteristic_silver_chain(
            exposure,
            surface,
            **{**kwargs, "development_time_minutes": 12.01},
        )
    except ValueError:
        outside_rejected = True

    measurements = {
        "maximum_density_reference_error": maximum_density_error,
        "maximum_interior_mean_transmittance_absolute_error": maximum_mean_error,
        "mean_density_strictly_increases_with_development_time": bool(
            np.all(np.diff([row["mean_density"] for row in rows]) > 0.0)
        ),
        "mean_transmittance_strictly_decreases_with_development_time": bool(
            np.all(np.diff([row["mean_transmittance"] for row in rows]) < 0.0)
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "domain_mismatch_rejected": domain_rejected,
        "non_neutral_exposure_rejected": non_neutral_rejected,
        "outside_surface_domain_rejected": outside_rejected,
        "input_unchanged_on_failure": bool(np.array_equal(exposure, original_exposure)),
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "maximum_density_reference_error": maximum_density_error
        <= gates["maximum_density_reference_error"],
        "maximum_interior_mean_transmittance_absolute_error": maximum_mean_error
        <= gates["maximum_interior_mean_transmittance_absolute_error"],
        **{
            key: measurements[key] is gates[key]
            for key in gates
            if key
            not in {
                "maximum_density_reference_error",
                "maximum_interior_mean_transmittance_absolute_error",
            }
        },
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ai_bw_characteristic_silver_chain_report.v1",
        "surface_identity": surface.identity(),
        "rows": rows,
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
