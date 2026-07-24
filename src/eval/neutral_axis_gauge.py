"""Frozen U5.R2I1 data-independent neutral-axis gauge evaluator."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.eval.cave_conditional_variability import array_sha256
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.eval.sensitometry_print_composition import build_composition
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator


def build_gauge(
    config: Mapping[str, Any],
    composition_config: Mapping[str, Any],
    sensitometry_config: Mapping[str, Any],
    print_config: Mapping[str, Any],
) -> NeutralAxisGaugeOperator:
    base = build_composition(composition_config, sensitometry_config, print_config)
    return NeutralAxisGaugeOperator.from_base(base, int(config["gauge_knots"]))


def _finite_difference_jacobians(operator: NeutralAxisGaugeOperator, points: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append((operator.apply(points + offset) - operator.apply(points - offset)) / (2.0 * step))
    return np.stack(columns, axis=-1)


def evaluate_gauge(
    config: Mapping[str, Any],
    composition_config: Mapping[str, Any],
    sensitometry_config: Mapping[str, Any],
    print_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    operator = build_gauge(config, composition_config, sensitometry_config, print_config)
    neutral = np.linspace(0.0, 1.0, int(config["neutral_audit_samples"]), dtype=np.float64)
    neutral_input = np.stack((neutral, neutral, neutral), axis=-1)
    ungauged = operator.base.apply(neutral_input)
    gauged = operator.apply(neutral_input)
    ungauged_spread = float(np.max(np.ptp(ungauged, axis=-1)))
    neutral_error = float(np.max(np.abs(gauged - neutral_input)))
    neutral_spread = float(np.max(np.ptp(gauged, axis=-1)))
    endpoints = operator.apply(np.asarray([[0.0] * 3, [1.0] * 3]))
    endpoint_error = float(np.max(np.abs(endpoints - np.asarray([[0.0] * 3, [1.0] * 3]))))

    rng = np.random.default_rng(int(config["seed"]))
    probes = rng.random((int(config["random_probes"]), 3))
    before = probes.copy()
    output = operator.apply(probes)
    partitioned = np.concatenate([operator.apply(item) for item in np.array_split(probes, 13)], axis=0)
    replay = NeutralAxisGaugeOperator.from_dict(json.loads(json.dumps(operator.to_dict(), sort_keys=True)))
    replay_error = float(np.max(np.abs(replay.apply(probes) - output)))
    identity_rmse = float(np.sqrt(np.mean((output - probes) ** 2)))
    design = np.column_stack((probes, np.ones(len(probes), dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(design, output, rcond=None)
    affine_residual = float(np.sqrt(np.mean((output - design @ coefficients) ** 2)))

    step = float(config["finite_difference_step"])
    jacobian_points = rng.uniform(step * 2.0, 1.0 - step * 2.0, size=(int(config["jacobian_probes"]), 3))
    jacobians = _finite_difference_jacobians(operator, jacobian_points, step)
    determinants = np.linalg.det(jacobians)
    minimum_determinant = float(np.min(determinants))

    guard_rejected = False
    try:
        operator.apply(np.asarray([[1.000001, 0.5, 0.5]]))
    except ValueError:
        guard_rejected = True
    gates = config["gates"]
    checks = {
        "ungauged_defect_present": ungauged_spread >= float(gates["ungauged_neutral_spread_min"]),
        "neutral_absolute": neutral_error <= float(gates["gauged_neutral_max_abs"]),
        "neutral_spread": neutral_spread <= float(gates["gauged_neutral_spread_max"]),
        "endpoints": endpoint_error <= float(gates["endpoint_max_abs"]),
        "output_range": float(np.min(output)) >= -float(gates["output_tolerance"]) and float(np.max(output)) <= 1.0 + float(gates["output_tolerance"]),
        "jacobian": minimum_determinant >= float(gates["minimum_jacobian_determinant"]),
        "identity_distance": identity_rmse >= float(gates["identity_rgb_rmse_min"]),
        "non_affine": affine_residual >= float(gates["best_affine_residual_rgb_rmse_min"]),
        "partition": bool(np.array_equal(output, partitioned)),
        "replay": replay_error <= float(gates["serialization_replay_max_abs"]),
        "source_preserved": bool(np.array_equal(probes, before)),
        "domain_guard": guard_rejected,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": "neutral_axis_gauge_pass" if all(checks.values()) else "neutral_axis_gauge_fail",
        "metrics": {
            "ungauged_neutral_maximum_channel_spread": ungauged_spread,
            "gauged_neutral_maximum_absolute_error": neutral_error,
            "gauged_neutral_maximum_channel_spread": neutral_spread,
            "endpoint_maximum_absolute_error": endpoint_error,
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "minimum_jacobian_determinant": minimum_determinant,
            "identity_rgb_rmse": identity_rmse,
            "best_affine_residual_rgb_rmse": affine_residual,
            "serialization_replay_maximum_absolute_error": replay_error,
            "partition_exact": bool(np.array_equal(output, partitioned)),
            "source_preserved": bool(np.array_equal(probes, before)),
            "domain_guard_rejected": guard_rejected,
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "neutral_input": neutral_input,
        "ungauged_neutral": ungauged,
        "gauged_neutral": gauged,
        "random_input": probes,
        "random_output": output,
        "jacobian_points": jacobian_points,
        "jacobians": jacobians,
        "jacobian_determinants": determinants,
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {"report": canonical_sha256(report), "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())}}
