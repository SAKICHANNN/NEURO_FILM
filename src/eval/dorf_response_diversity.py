"""U5.R2AB1 synthetic diversity and safety evaluator for CAVE DoRF curves."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations, product
from typing import Any, Mapping

import numpy as np

from src.eval.dorf_film_response import DorfCurve
from src.eval.velvia_datasheet_witness import (
    _RGB_TO_XYZ,
    delta_e76,
    encoded_srgb_to_linear,
    xyz_to_lab,
)
from src.roll2film.baselines import fit_joint_basic_adjustment


def _lab(encoded: np.ndarray) -> np.ndarray:
    return xyz_to_lab(encoded_srgb_to_linear(np.asarray(encoded)) @ _RGB_TO_XYZ.T)


def _median_de(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(delta_e76(_lab(a), _lab(b))))


def synthetic_rgb(levels: list[float]) -> np.ndarray:
    return np.asarray(list(product(levels, repeat=3)), dtype=np.float64)


def apply_triplet(
    rgb: np.ndarray,
    channels: Mapping[str, DorfCurve],
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    result = np.empty_like(values)
    for index, channel in enumerate(("red", "green", "blue")):
        curve = channels[channel]
        result[:, index] = np.interp(
            values[:, index], curve.irradiance, curve.brightness
        )
    return result


def shared_curve_output(
    rgb: np.ndarray,
    channels: Mapping[str, DorfCurve],
) -> np.ndarray:
    curves = [channels[channel] for channel in ("red", "green", "blue")]
    base = curves[0].irradiance
    if any(not np.array_equal(curve.irradiance, base) for curve in curves[1:]):
        raise ValueError("triplet irradiance supports differ")
    mean = np.mean([curve.brightness for curve in curves], axis=0)
    return np.column_stack(
        [np.interp(rgb[:, index], base, mean) for index in range(3)]
    )


def _power_exponents(channels: Mapping[str, DorfCurve]) -> np.ndarray:
    exponents = []
    for channel in ("red", "green", "blue"):
        curve = channels[channel]
        mask = (
            (curve.irradiance > 0.0)
            & (curve.irradiance < 1.0)
            & (curve.brightness > 0.0)
            & (curve.brightness < 1.0)
        )
        log_x = np.log(curve.irradiance[mask])
        log_y = np.log(curve.brightness[mask])
        exponent = float(np.dot(log_x, log_y) / np.dot(log_x, log_x))
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("invalid fitted power exponent")
        exponents.append(exponent)
    return np.asarray(exponents)


def evaluate_candidate(
    name: str,
    channels: Mapping[str, DorfCurve],
    config: Mapping[str, Any],
    rgb: np.ndarray,
) -> dict[str, Any]:
    target = apply_triplet(rgb, channels)
    shared = shared_curve_output(rgb, channels)
    gamma = _power_exponents(channels)
    power = rgb**gamma[None, :]
    basic_operator = fit_joint_basic_adjustment(rgb, target)
    basic = basic_operator.apply(rgb)

    neutral_levels = np.linspace(0.0, 1.0, config["synthetic_population"]["neutral_levels"])
    neutral = np.repeat(neutral_levels[:, None], 3, axis=1)
    neutral_output = apply_triplet(neutral, channels)
    neutral_lab = _lab(neutral_output)
    neutral_chroma = np.sqrt(np.sum(neutral_lab[:, 1:] ** 2, axis=1))

    low, high = config["synthetic_population"]["derivative_domain"]
    derivative_x = np.linspace(
        low, high, config["synthetic_population"]["derivative_samples"]
    )
    derivatives: dict[str, dict[str, float]] = {}
    all_derivatives = []
    for channel in ("red", "green", "blue"):
        curve = channels[channel]
        values = np.interp(derivative_x, curve.irradiance, curve.brightness)
        slope = np.gradient(values, derivative_x)
        all_derivatives.append(slope)
        derivatives[channel] = {
            "minimum": float(np.min(slope)),
            "maximum": float(np.max(slope)),
        }
    derivative_values = np.concatenate(all_derivatives)
    gates = config["gates"]
    metrics = {
        "style_delta_e76_median": _median_de(rgb, target),
        "shared_curve_residual_delta_e76_median": _median_de(shared, target),
        "per_channel_power_residual_delta_e76_median": _median_de(power, target),
        "joint_basic_residual_delta_e76_median": _median_de(basic, target),
        "neutral_chroma_max": float(np.max(neutral_chroma)),
        "channel_derivative_minimum": float(np.min(derivative_values)),
        "channel_derivative_maximum": float(np.max(derivative_values)),
        "raw_output_minimum": float(np.min(target)),
        "raw_output_maximum": float(np.max(target)),
        "power_exponents": gamma.tolist(),
        "per_channel_derivatives": derivatives,
        "output_sha256": sha256(target.astype("<f8").tobytes()).hexdigest(),
    }
    checks = {
        "style": metrics["style_delta_e76_median"]
        >= gates["candidate_style_delta_e76_median_min"],
        "shared_curve_residual": metrics["shared_curve_residual_delta_e76_median"]
        >= gates["candidate_shared_curve_residual_delta_e76_median_min"],
        "per_channel_power_residual": metrics[
            "per_channel_power_residual_delta_e76_median"
        ]
        >= gates["candidate_per_channel_power_residual_delta_e76_median_min"],
        "joint_basic_residual": metrics["joint_basic_residual_delta_e76_median"]
        >= gates["candidate_joint_basic_residual_delta_e76_median_min"],
        "neutral_chroma": metrics["neutral_chroma_max"] <= gates["neutral_chroma_max"],
        "derivative_floor": metrics["channel_derivative_minimum"]
        >= gates["minimum_channel_derivative"],
        "derivative_cap": metrics["channel_derivative_maximum"]
        <= gates["maximum_channel_derivative"],
        "raw_range": (
            np.all(np.isfinite(target))
            and metrics["raw_output_minimum"] >= gates["raw_output_min"]
            and metrics["raw_output_maximum"] <= gates["raw_output_max"]
        ),
    }
    return {
        "name": name,
        "scale": channels["red"].scale,
        "metrics": metrics,
        "checks": checks,
        "survives": all(checks.values()),
        "_output": target,
    }


def evaluate_bank(
    triplets: Mapping[str, Mapping[str, DorfCurve]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rgb = synthetic_rgb(config["synthetic_population"]["encoded_rgb_levels"])
    records = [
        evaluate_candidate(name, channels, config, rgb)
        for name, channels in triplets.items()
    ]
    survivors = [record for record in records if record["survives"]]
    pairwise = []
    for left, right in combinations(survivors, 2):
        rmse = float(np.sqrt(np.mean((left["_output"] - right["_output"]) ** 2)))
        pairwise.append({"left": left["name"], "right": right["name"], "rmse": rmse})
    minimum_pairwise = min((item["rmse"] for item in pairwise), default=None)
    gates = config["gates"]
    bank_checks = {
        "minimum_candidate_survivors": len(survivors)
        >= gates["minimum_candidate_survivors"],
        "minimum_survivor_pairwise_output_rmse": minimum_pairwise is not None
        and minimum_pairwise >= gates["minimum_survivor_pairwise_output_rmse"],
    }
    if not survivors:
        decision = "no_structural_survivor"
    elif not bank_checks["minimum_candidate_survivors"]:
        decision = "insufficient_diversity"
    elif not bank_checks["minimum_survivor_pairwise_output_rmse"]:
        decision = "insufficient_diversity"
    else:
        decision = "full_pass"
    public_records = []
    for record in records:
        public_records.append({key: value for key, value in record.items() if key != "_output"})
    return {
        "candidate_count": len(records),
        "survivor_count": len(survivors),
        "survivor_names": [record["name"] for record in survivors],
        "minimum_survivor_pairwise_output_rmse": minimum_pairwise,
        "bank_checks": bank_checks,
        "decision": decision,
        "candidates": public_records,
        "survivor_pairwise": pairwise,
    }
