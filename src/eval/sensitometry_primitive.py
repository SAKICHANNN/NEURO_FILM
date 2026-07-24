"""Frozen U2.2A clean-room sensitometry primitive evaluator."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from src.eval.cave_conditional_variability import array_sha256
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.roll2film.sensitometry import (
    AnchoredCharacteristicCurve,
    LogExposureEncoder,
    RGBSensitometryOperator,
)
from src.roll2film.splines import RationalQuadraticSpline


def build_operator(config: Mapping[str, Any]) -> RGBSensitometryOperator:
    encoder_config = config["encoder"]
    encoder = LogExposureEncoder(
        float(encoder_config["reference_linear"]),
        float(encoder_config["black_offset"]),
    )
    x = np.asarray(config["curve_x_knots"], dtype=np.float64)
    anchor = config["anchor"]
    curves = tuple(
        AnchoredCharacteristicCurve(
            RationalQuadraticSpline.from_knots(
                x, np.asarray(config["curve_y_knots"][layer], dtype=np.float64)
            ),
            float(anchor["log_exposure"]),
            float(anchor["density"]),
            layer,
        )
        for layer in ("red", "green", "blue")
    )
    return RGBSensitometryOperator(encoder, curves)


def evaluate_sensitometry(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    operator = build_operator(config)
    rng = np.random.default_rng(int(config["seed"]))
    samples = int(config["samples"])
    linear = rng.uniform(0.0, float(config["linear_domain_max"]), size=(samples, 3))
    boundary = np.array(
        [[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [16.0, 16.0, 16.0]],
        dtype=np.float64,
    )
    linear = np.concatenate((boundary, linear), axis=0)
    exposure = operator.log_exposure(linear)
    density = operator.apply(linear)
    restored = operator.inverse(density)
    encoder_restored = operator.encoder.inverse(exposure)
    replay = RGBSensitometryOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    roundtrip = float(np.max(np.abs(restored - linear)))
    encoder_roundtrip = float(np.max(np.abs(encoder_restored - linear)))
    replay_exact = bool(np.array_equal(replay.apply(linear), density))
    jacobian = operator.jacobian_determinant(linear)

    anchor_linear = np.full((1, 3), float(config["encoder"]["reference_linear"]))
    anchor_density = operator.apply(anchor_linear)[0]
    anchor_error = float(np.max(np.abs(anchor_density - float(config["anchor"]["density"]))))
    probes = config["slope_probes_log_exposure"]
    slopes: dict[str, dict[str, float]] = {}
    all_derivatives = []
    shape_checks = {}
    for curve in operator.curves:
        layer_slopes = {
            name: float(curve.derivative(np.asarray(float(value))))
            for name, value in probes.items()
        }
        slopes[curve.layer] = layer_slopes
        all_derivatives.extend(layer_slopes.values())
        shape_checks[curve.layer] = (
            layer_slopes["toe"] < layer_slopes["mid"]
            and layer_slopes["shoulder"] < layer_slopes["mid"]
        )

    diversity_exposure = np.array([-3.0, -2.0, -1.0, 1.0, 1.8], dtype=np.float64)
    layer_density = np.stack(
        [curve.apply(diversity_exposure) for curve in operator.curves], axis=0
    )
    pairwise_separation = {
        f"{first}-{second}": float(
            np.max(np.abs(layer_density[first_index] - layer_density[second_index]))
        )
        for (first_index, first), (second_index, second) in combinations(
            enumerate(("red", "green", "blue")), 2
        )
    }
    minimum_pairwise_separation = min(pairwise_separation.values())

    guards = {}
    try:
        operator.apply(np.array([[-1e-9, 0.1, 0.1]]))
        guards["negative_linear_rejected"] = False
    except ValueError:
        guards["negative_linear_rejected"] = True
    try:
        operator.encoder.inverse(
            np.array([operator.encoder.minimum_log_exposure - 1e-9])
        )
        guards["below_boundary_inverse_rejected"] = False
    except ValueError:
        guards["below_boundary_inverse_rejected"] = True

    gates = config["gates"]
    checks = {
        "roundtrip": roundtrip <= float(gates["roundtrip_max_abs"]),
        "encoder_roundtrip": encoder_roundtrip
        <= float(gates["encoder_roundtrip_max_abs"]),
        "anchor": anchor_error <= float(gates["neutral_anchor_density_error_max"]),
        "jacobian": float(np.min(jacobian))
        > float(gates["jacobian_determinant_min_exclusive"]),
        "curve_derivative": min(all_derivatives)
        > float(gates["curve_derivative_min_exclusive"]),
        "toe_shoulder_shape": all(shape_checks.values()),
        "layer_diversity": minimum_pairwise_separation
        >= float(gates["pairwise_layer_density_separation_max_min"]),
        "replay": replay_exact,
        "domain_guards": all(guards.values()),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": "sensitometry_primitive_pass" if passed else "sensitometry_primitive_fail",
        "metrics": {
            "roundtrip_max_abs": roundtrip,
            "encoder_roundtrip_max_abs": encoder_roundtrip,
            "neutral_anchor_density": anchor_density.tolist(),
            "neutral_anchor_density_error_max": anchor_error,
            "jacobian_determinant_min": float(np.min(jacobian)),
            "jacobian_determinant_p50": float(np.median(jacobian)),
            "curve_slopes": slopes,
            "curve_shape_checks": shape_checks,
            "pairwise_layer_density_separation_max": pairwise_separation,
            "minimum_pairwise_layer_separation": minimum_pairwise_separation,
            "replay_exact": replay_exact,
            "domain_guards": guards,
            "log_exposure_range": [float(np.min(exposure)), float(np.max(exposure))],
            "density_range": [float(np.min(density)), float(np.max(density))],
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "linear": linear,
        "log_exposure": exposure,
        "density": density,
        "restored": restored,
        "jacobian_determinant": jacobian,
        "diversity_layer_density": layer_density,
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }

