"""Frozen U2.3A smooth tetrahedral residual composition evaluator."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.eval.cave_conditional_variability import array_sha256
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.eval.sensitometry_print_composition import build_composition
from src.roll2film.constrained import LUTConstraintSpec, audit_lut_constraints, identity_lut
from src.roll2film.lut import DenseLUT3D
from src.roll2film.residual import SensitometryResidualLUTOperator


def _constraint_spec(config: Mapping[str, Any]) -> LUTConstraintSpec:
    return LUTConstraintSpec(**config["constraints"])


def _cyclic_lut(size: int, amplitude: float) -> DenseLUT3D:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    following = np.roll(grid, -1, axis=-1)
    preceding = np.roll(grid, 1, axis=-1)
    values = grid + amplitude * grid * (1.0 - grid) * (following - preceding)
    return DenseLUT3D(values, np.zeros(3), np.ones(3), "tetrahedral")


def _scalar_tetrahedral(lut: DenseLUT3D, point: np.ndarray) -> np.ndarray:
    coordinate = point * (lut.size - 1)
    lower = np.minimum(np.floor(coordinate).astype(int), lut.size - 2)
    f = coordinate - lower
    r, g, b = lower
    red, green, blue = f
    c000 = lut.values[r, g, b]
    c100 = lut.values[r + 1, g, b]
    c010 = lut.values[r, g + 1, b]
    c001 = lut.values[r, g, b + 1]
    c110 = lut.values[r + 1, g + 1, b]
    c101 = lut.values[r + 1, g, b + 1]
    c011 = lut.values[r, g + 1, b + 1]
    c111 = lut.values[r + 1, g + 1, b + 1]
    if red >= green >= blue:
        return c000 + red * (c100 - c000) + green * (c110 - c100) + blue * (c111 - c110)
    if red >= blue > green:
        return c000 + red * (c100 - c000) + blue * (c101 - c100) + green * (c111 - c101)
    if blue > red >= green:
        return c000 + blue * (c001 - c000) + red * (c101 - c001) + green * (c111 - c101)
    if green > red >= blue:
        return c000 + green * (c010 - c000) + red * (c110 - c010) + blue * (c111 - c110)
    if green >= blue > red:
        return c000 + green * (c010 - c000) + blue * (c011 - c010) + red * (c111 - c011)
    return c000 + blue * (c001 - c000) + green * (c011 - c001) + red * (c111 - c011)


def _finite_difference_jacobians(operator: Any, points: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append((operator.apply(points + offset) - operator.apply(points - offset)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def evaluate_composition(
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    print_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base = build_composition(base_config, parent_config, print_config)
    spec = _constraint_spec(config)
    size = int(config["lut_size"])
    identity = SensitometryResidualLUTOperator(base, identity_lut(size), spec)
    candidate_lut = _cyclic_lut(size, float(config["residual_amplitude"]))
    constraint_report = audit_lut_constraints(candidate_lut, spec)

    construction_error = None
    candidate = None
    try:
        candidate = SensitometryResidualLUTOperator(base, candidate_lut, spec)
    except ValueError as exc:
        construction_error = str(exc)

    rng = np.random.default_rng(int(config["seed"]))
    probes = rng.random((int(config["random_probes"]), 3))
    before = probes.copy()
    base_output = base.apply(probes)
    identity_output = identity.apply(probes)
    identity_error = float(np.max(np.abs(identity_output - base_output)))
    scalar = np.asarray([_scalar_tetrahedral(candidate_lut, point) for point in base_output])
    tetrahedral_error = float(np.max(np.abs(candidate_lut.apply(base_output) - scalar)))

    metrics: dict[str, Any] = {
        "identity_composition_max_abs": identity_error,
        "tetrahedral_scalar_reference_max_abs": tetrahedral_error,
        "source_preserved": bool(np.array_equal(probes, before)),
        "constraint_report": constraint_report.to_dict(),
        "construction_error": construction_error,
    }
    arrays: dict[str, np.ndarray] = {
        "input": probes,
        "base_output": base_output,
        "candidate_lut_values": candidate_lut.values,
    }
    gates = config["gates"]
    checks: dict[str, bool] = {
        "identity_composition": identity_error <= float(gates["identity_composition_max_abs"]),
        "tetrahedral_reference": tetrahedral_error <= float(gates["tetrahedral_scalar_reference_max_abs"]),
        "lut_constraints": constraint_report.passes,
        "construction": candidate is not None,
        "source_preserved": metrics["source_preserved"],
    }

    if candidate is not None:
        output = candidate.apply(probes)
        replay = SensitometryResidualLUTOperator.from_dict(json.loads(json.dumps(candidate.to_dict(), sort_keys=True)))
        replay_error = float(np.max(np.abs(replay.apply(probes) - output)))
        partitioned = np.concatenate([candidate.apply(part) for part in np.array_split(probes, 11)], axis=0)
        residual_rmse = float(np.sqrt(np.mean((output - base_output) ** 2)))
        base_style = float(np.sqrt(np.mean((base_output - probes) ** 2)))
        candidate_style = float(np.sqrt(np.mean((output - probes) ** 2)))
        style_retention = candidate_style / base_style
        step = float(config["finite_difference_step"])
        interior = rng.uniform(step * 2.0, 1.0 - step * 2.0, size=(2048, 3))
        jacobians = _finite_difference_jacobians(candidate, interior, step)
        determinants = np.linalg.det(jacobians)
        metrics.update({
            "output_min": float(np.min(output)),
            "output_max": float(np.max(output)),
            "serialization_replay_max_abs": replay_error,
            "partition_exact": bool(np.array_equal(output, partitioned)),
            "residual_rgb_rmse": residual_rmse,
            "base_style_rgb_rmse": base_style,
            "candidate_style_rgb_rmse": candidate_style,
            "base_style_retention_ratio": style_retention,
            "minimum_composition_jacobian_determinant": float(np.min(determinants)),
        })
        tolerance = float(gates["composition_output_tolerance"])
        checks.update({
            "composition_range": metrics["output_min"] >= -tolerance and metrics["output_max"] <= 1.0 + tolerance,
            "composition_jacobian": metrics["minimum_composition_jacobian_determinant"] > float(gates["minimum_composition_jacobian_determinant_exclusive"]),
            "residual_nontrivial_bounded": float(gates["residual_rgb_rmse_min"]) <= residual_rmse <= float(gates["residual_rgb_rmse_max"]),
            "base_style_retained": style_retention >= float(gates["base_style_retention_ratio_min"]),
            "replay": replay_error <= float(gates["serialization_replay_max_abs"]),
            "partition": metrics["partition_exact"],
        })
        arrays.update({"candidate_output": output, "composition_jacobians": jacobians, "composition_jacobian_determinants": determinants})

    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": "smooth_residual_composition_pass" if passed else "smooth_residual_composition_fail",
        "metrics": metrics,
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {"report": canonical_sha256(report), "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())}}
