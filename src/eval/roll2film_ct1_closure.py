"""Known-truth evaluator for Roll2Film CT1 L0/gauge/shaper closure."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.eval.cave_conditional_variability import array_sha256
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.roll2film.lut import LogShaperSpec, ShapedLUT3D, bake_shaped_lut
from src.roll2film.photometric import PhotometricColorOperator, canonicalize_roll_nuisance
from src.roll2film.simulator import default_truth_operator
from src.roll2film.splines import AffineMonotoneSplineOperator, RationalQuadraticSpline


def _fixed_l2_operator() -> AffineMonotoneSplineOperator:
    x = np.array([-0.25, 0.0, 0.18, 0.45, 0.75, 1.0, 1.35], dtype=np.float64)
    curves = (
        np.array([-0.24, 0.0, 0.15, 0.47, 0.82, 1.04, 1.37]),
        np.array([-0.27, -0.01, 0.17, 0.44, 0.72, 0.98, 1.32]),
        np.array([-0.23, 0.01, 0.20, 0.49, 0.78, 1.02, 1.36]),
    )
    return AffineMonotoneSplineOperator(
        default_truth_operator(),
        tuple(RationalQuadraticSpline.from_knots(x, y) for y in curves),
    )


def evaluate_ct1_closure(
    config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(config["seed"]))
    probes = int(config["probes"])
    truth = config["l0_truth"]
    l0 = PhotometricColorOperator(
        float(truth["log_exposure"]),
        np.asarray(truth["log_white_balance"], dtype=np.float64),
        str(truth["working_space"]),
    )
    l0_input = rng.uniform(-0.5, float(config["hdr_domain_max"]), size=(probes, 3))
    l0_output = l0.apply(l0_input)
    l0_restored = l0.inverse(l0_output)
    l0_replay = PhotometricColorOperator.from_dict(
        json.loads(json.dumps(l0.to_dict(), sort_keys=True))
    )
    l0_roundtrip = float(np.max(np.abs(l0_restored - l0_input)))
    l0_replay_exact = bool(np.array_equal(l0_replay.apply(l0_input), l0_output))

    gauge_truth = config["gauge_truth"]
    raw_exposure = np.asarray(gauge_truth["frame_log_exposure"], dtype=np.float64)
    raw_wb = np.asarray(gauge_truth["frame_log_white_balance"], dtype=np.float64)
    gauge = canonicalize_roll_nuisance(raw_exposure, raw_wb)
    gauge_wb_sum = float(np.max(np.abs(np.sum(gauge.frame_log_white_balance, axis=1))))
    gauge_exposure_mean = abs(float(np.mean(gauge.frame_log_exposure)))
    gauge_recomposition = float(
        np.max(np.abs(gauge.recomposed_log_gains - (raw_exposure[:, None] + raw_wb)))
    )

    shaper = LogShaperSpec(
        float(config["hdr_domain_max"]), float(config["shaper_compression"])
    )
    shaped_probe = rng.uniform(0.0, 1.0, size=(probes, 3))
    linear_probe = shaper.inverse(shaped_probe)
    shaper_roundtrip = float(
        np.max(np.abs(shaper.inverse(shaper.apply(linear_probe)) - linear_probe))
    )
    shaper_derivative_min = float(np.min(shaper.derivative(linear_probe)))

    operator = _fixed_l2_operator()
    analytic = operator.apply(linear_probe)
    bundles: dict[int, ShapedLUT3D] = {}
    errors: dict[int, np.ndarray] = {}
    replay_exact: dict[int, bool] = {}
    for size in config["lut_sizes"]:
        cube_size = int(size)
        bundle = bake_shaped_lut(
            operator,
            cube_size,
            shaper,
            interpolation="trilinear",
            working_space=operator.working_space,
        )
        replay = ShapedLUT3D.from_dict(
            json.loads(json.dumps(bundle.to_dict(), sort_keys=True))
        )
        rendered = bundle.apply(linear_probe)
        bundles[cube_size] = bundle
        errors[cube_size] = np.abs(rendered - analytic)
        replay_exact[cube_size] = bool(np.array_equal(replay.apply(linear_probe), rendered))
    error33 = float(np.max(errors[33]))
    error65 = float(np.max(errors[65]))
    ratio = error65 / max(error33, 1e-30)

    guards = {}
    try:
        shaper.apply(np.array([-1e-6], dtype=np.float64))
        guards["negative_linear_rejected"] = False
    except ValueError:
        guards["negative_linear_rejected"] = True
    try:
        bundles[33].apply(
            np.array([[float(config["hdr_domain_max"]) + 1e-6, 1.0, 1.0]])
        )
        guards["above_domain_rejected"] = False
    except ValueError:
        guards["above_domain_rejected"] = True

    gates = config["gates"]
    checks = {
        "l0_roundtrip": l0_roundtrip <= float(gates["l0_roundtrip_max_abs"]),
        "l0_replay": l0_replay_exact,
        "l0_jacobian": l0.jacobian_determinant > 0.0,
        "gauge_wb": gauge_wb_sum <= float(gates["gauge_wb_row_sum_max_abs"]),
        "gauge_exposure": gauge_exposure_mean
        <= float(gates["gauge_exposure_mean_max_abs"]),
        "gauge_recomposition": gauge_recomposition
        <= float(gates["gauge_recomposition_max_abs"]),
        "shaper_roundtrip": shaper_roundtrip
        <= float(gates["shaper_roundtrip_max_abs"]),
        "shaper_derivative": shaper_derivative_min
        > float(gates["shaper_derivative_min_exclusive"]),
        "lut33_error": error33 <= float(gates["lut33_max_abs_rgb_error"]),
        "lut65_error": error65 <= float(gates["lut65_max_abs_rgb_error"]),
        "lut_refinement": ratio
        <= float(gates["lut65_to_lut33_error_ratio_max"]),
        "lut_replay": all(replay_exact.values()),
        "domain_guards": all(guards.values()),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": "ct1_l0_gauge_shaper_pass" if passed else "ct1_closure_failed",
        "metrics": {
            "l0_roundtrip_max_abs": l0_roundtrip,
            "l0_replay_exact": l0_replay_exact,
            "l0_jacobian_determinant": l0.jacobian_determinant,
            "gauge_wb_row_sum_max_abs": gauge_wb_sum,
            "gauge_exposure_mean_max_abs": gauge_exposure_mean,
            "gauge_recomposition_max_abs": gauge_recomposition,
            "gauge_shared_log_exposure": gauge.shared_log_exposure,
            "shaper_roundtrip_max_abs": shaper_roundtrip,
            "shaper_derivative_min": shaper_derivative_min,
            "lut33_max_abs_rgb_error": error33,
            "lut33_rmse": float(np.sqrt(np.mean(errors[33] ** 2))),
            "lut65_max_abs_rgb_error": error65,
            "lut65_rmse": float(np.sqrt(np.mean(errors[65] ** 2))),
            "lut65_to_lut33_error_ratio": ratio,
            "lut_replay_exact": {str(key): value for key, value in replay_exact.items()},
            "domain_guards": guards,
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "l0_input": l0_input,
        "l0_output": l0_output,
        "linear_probe": linear_probe,
        "analytic_output": analytic,
        "lut33_error": errors[33],
        "lut65_error": errors[65],
        "gauge_recomposed_log_gains": gauge.recomposed_log_gains,
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }

