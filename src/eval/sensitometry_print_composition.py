"""Frozen U2.2B sensitometry-to-print composition evaluator."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.eval.cave_conditional_variability import array_sha256
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.eval.sensitometry_primitive import build_operator as build_sensitometry
from src.roll2film.density_domain import finite_difference_jacobians
from src.roll2film.sensitometry_print import (
    DensityToPrintInterpretation,
    SensitometryPrintOperator,
)


def build_composition(
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    print_config: Mapping[str, Any],
) -> SensitometryPrintOperator:
    source = print_config["witnesses"][config["print_source_witness"]]
    sensitometry = build_sensitometry(parent_config)
    references = sensitometry.apply(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    )
    interpretation = DensityToPrintInterpretation(
        np.asarray(source["dye_absorption_matrix"], dtype=np.float64),
        np.asarray(source["print_matrix"], dtype=np.float64),
        np.asarray(source["paper_midpoints"], dtype=np.float64),
        np.asarray(source["paper_slopes"], dtype=np.float64),
        np.asarray(source["paper_maximum_densities"], dtype=np.float64),
        references[0],
        references[1],
    )
    return SensitometryPrintOperator(sensitometry, interpretation)


def evaluate_composition(
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    print_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    operator = build_composition(config, parent_config, print_config)
    size = int(config["grid_size"])
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3)
    before = flat.copy()
    output = operator.apply(flat)
    endpoints = operator.apply(np.array([[0.0] * 3, [1.0] * 3], dtype=np.float64))
    endpoint_error = float(
        np.max(np.abs(endpoints - np.array([[0.0] * 3, [1.0] * 3])))
    )
    source_preserved = bool(np.array_equal(flat, before))
    partitioned = np.concatenate(
        [operator.apply(part) for part in np.array_split(flat, 11)], axis=0
    )
    partition_exact = bool(np.array_equal(output, partitioned))
    replay = SensitometryPrintOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    replay_exact = bool(np.array_equal(replay.apply(flat), output))

    interior_axis = axis[1:-1]
    interior = np.stack(
        np.meshgrid(interior_axis, interior_axis, interior_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    jacobians = finite_difference_jacobians(
        operator, interior, step=float(config["finite_difference_step"])
    )
    determinants = np.linalg.det(jacobians)
    minimum_direction = float(np.min(jacobians))
    minimum_determinant = float(np.min(determinants))

    identity_rmse = float(np.sqrt(np.mean((output - flat) ** 2)))
    design = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(design, output, rcond=None)
    affine = design @ coefficients
    affine_residual = float(np.sqrt(np.mean((output - affine) ** 2)))

    rng = np.random.default_rng(int(config["seed"]))
    random_input = rng.random((int(config["random_probes"]), 3))
    random_output = operator.apply(random_input)
    random_replay_exact = bool(np.array_equal(replay.apply(random_input), random_output))

    guards = {}
    try:
        operator.apply(np.array([[1.0 + 1e-6, 0.5, 0.5]]))
        guards["rgb_domain_rejected"] = False
    except ValueError:
        guards["rgb_domain_rejected"] = True
    invalid_density = operator.interpretation.black_reference_density.copy()
    invalid_density[0] -= 1e-6
    try:
        operator.interpretation.apply(invalid_density[None, :])
        guards["density_domain_rejected"] = False
    except ValueError:
        guards["density_domain_rejected"] = True

    gates = config["gates"]
    checks = {
        "endpoints": endpoint_error <= float(gates["endpoint_max_abs"]),
        "range": float(np.min(output)) >= -float(gates["range_tolerance"])
        and float(np.max(output)) <= 1.0 + float(gates["range_tolerance"]),
        "partition": partition_exact,
        "replay": replay_exact and random_replay_exact,
        "source_preserved": source_preserved,
        "directional_derivative": minimum_direction
        >= float(gates["minimum_directional_derivative"]),
        "jacobian": minimum_determinant
        > float(gates["minimum_jacobian_determinant_exclusive"]),
        "identity_distance": identity_rmse >= float(gates["identity_rgb_rmse_min"]),
        "non_affine": affine_residual
        >= float(gates["best_affine_residual_rgb_rmse_min"]),
        "domain_guards": all(guards.values()),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": "sensitometry_print_composition_pass" if passed else "sensitometry_print_composition_fail",
        "architecture_audit": {
            "sensitometry_stage_count": 1,
            "used_print_fields": sorted(config["allowed_print_fields"]),
            "forbidden_print_fields_used": [],
        },
        "metrics": {
            "endpoint_max_abs": endpoint_error,
            "output_min": float(np.min(output)),
            "output_max": float(np.max(output)),
            "partition_exact": partition_exact,
            "replay_exact": replay_exact and random_replay_exact,
            "source_preserved": source_preserved,
            "minimum_directional_derivative": minimum_direction,
            "minimum_jacobian_determinant": minimum_determinant,
            "identity_rgb_rmse": identity_rmse,
            "best_affine_residual_rgb_rmse": affine_residual,
            "domain_guards": guards,
            "black_reference_density": operator.interpretation.black_reference_density.tolist(),
            "white_reference_density": operator.interpretation.white_reference_density.tolist(),
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "grid_input": flat,
        "grid_output": output,
        "jacobians": jacobians,
        "jacobian_determinants": determinants,
        "random_input": random_input,
        "random_output": random_output,
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }

