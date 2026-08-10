"""Clean-room capacity gate for a learnable-axis cylindrical colour operator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.roll2film.factorized_boundary_guard import _maximum_safe_scale


class LearnableAxisOperatorError(ValueError):
    """Raised when the frozen contract or explicit operator is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "neuro_film.u5_r2ca0_learnable_axis_cylindrical_operator_contract.v1",
        "experiment_id": "U5.R2CA0",
        "status": "contract_frozen_implementation_ready",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise LearnableAxisOperatorError("contract identity drift")
    fixture = payload.get("fixture", {})
    gates = payload.get("gates", {})
    scope = payload.get("clean_room_scope", {})
    if (
        fixture.get("build_grid_size") != 9
        or fixture.get("confirmation_grid_size") != 15
        or fixture.get("axis_bank", [[None]])[0] != [1.0, 1.0, 1.0]
        or fixture.get("truth_axis_indices") != [1, 2, 3, 4, 5, 6]
        or fixture.get("wrong_axis_offset") != 2
        or gates.get("minimum_median_improvement_over_fixed_axis") != 0.7
        or scope.get("paper_code_weights_or_data_used")
        or scope.get("neural_network_used")
        or scope.get("learned_final_rgb_allowed")
    ):
        raise LearnableAxisOperatorError("contract boundary drift")
    return payload


def _normalise_axis(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (3,) or not np.all(np.isfinite(values)) or norm <= 0.0:
        raise LearnableAxisOperatorError("invalid luminance axis")
    normal = values / norm
    if np.any(normal <= 0.0):
        raise LearnableAxisOperatorError("luminance axis must remain positive")
    return normal


def _grid(size: int, low: float, high: float) -> np.ndarray:
    axis = np.linspace(low, high, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _raw_cylindrical_operator(
    rgb: np.ndarray,
    axis: np.ndarray,
    *,
    tone_amplitude: float,
    chroma_gain: float,
    maximum_rotation_radians: float,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    normal = _normalise_axis(axis)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise LearnableAxisOperatorError("invalid RGB values")
    centered = values - 0.5
    longitudinal = centered @ normal
    transverse = centered - longitudinal[:, None] * normal[None, :]
    extent = 0.5 * float(np.sum(np.abs(normal)))
    coordinate = np.clip(longitudinal / extent, -1.0, 1.0)
    envelope = 1.0 - np.square(coordinate)
    tone_delta = tone_amplitude * envelope * (0.35 + 0.65 * coordinate)
    angle = maximum_rotation_radians * envelope
    gain = 1.0 + chroma_gain * envelope
    rotated = (
        np.cos(angle)[:, None] * transverse
        + np.sin(angle)[:, None] * np.cross(normal[None, :], transverse)
    )
    return 0.5 + (longitudinal + tone_delta)[:, None] * normal + gain[:, None] * rotated


def apply_cylindrical_operator(
    rgb: np.ndarray,
    axis: np.ndarray,
    *,
    tone_amplitude: float,
    chroma_gain: float,
    maximum_rotation_radians: float,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rgb, dtype=np.float64)
    raw = _raw_cylindrical_operator(
        values,
        axis,
        tone_amplitude=tone_amplitude,
        chroma_gain=chroma_gain,
        maximum_rotation_radians=maximum_rotation_radians,
    )
    lower = np.minimum(values, boundary_epsilon)
    upper = np.maximum(values, 1.0 - boundary_epsilon)
    scale = _maximum_safe_scale(values, raw - values, lower, upper)
    output = values + scale[:, None] * (raw - values)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise LearnableAxisOperatorError("analytical safety escaped the RGB cube")
    return output, scale


def _fit_independent_rgb_cubic(rgb: np.ndarray, target: np.ndarray, degree: int) -> np.ndarray:
    coefficients = []
    for channel in range(3):
        design = np.stack(
            [np.power(rgb[:, channel], exponent) for exponent in range(degree + 1)],
            axis=1,
        )
        coefficients.append(np.linalg.lstsq(design, target[:, channel], rcond=None)[0])
    return np.stack(coefficients)


def _apply_independent_rgb_cubic(rgb: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    degree = coefficients.shape[1] - 1
    output = np.empty_like(rgb)
    for channel in range(3):
        design = np.stack(
            [np.power(rgb[:, channel], exponent) for exponent in range(degree + 1)],
            axis=1,
        )
        output[:, channel] = design @ coefficients[channel]
    return output


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _improvement(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise LearnableAxisOperatorError("baseline error must be positive")
    return 1.0 - candidate / baseline


def _new_boundary_fraction(source: np.ndarray, output: np.ndarray, epsilon: float) -> float:
    source_boundary = np.any((source <= epsilon) | (source >= 1.0 - epsilon), axis=1)
    output_boundary = np.any((output <= epsilon) | (output >= 1.0 - epsilon), axis=1)
    return float(np.mean(output_boundary & ~source_boundary))


def _minimum_jacobian(
    points: np.ndarray,
    axis: np.ndarray,
    fixture: Mapping[str, Any],
) -> float:
    step = float(fixture["jacobian_step"])
    determinants = []
    for point in points:
        columns = []
        for channel in range(3):
            positive = point.copy()
            negative = point.copy()
            positive[channel] += step
            negative[channel] -= step
            positive_output, _ = apply_cylindrical_operator(
                positive[None, :], axis, **_operator_kwargs(fixture)
            )
            negative_output, _ = apply_cylindrical_operator(
                negative[None, :], axis, **_operator_kwargs(fixture)
            )
            columns.append((positive_output[0] - negative_output[0]) / (2.0 * step))
        determinants.append(float(np.linalg.det(np.stack(columns, axis=1))))
    return float(np.min(determinants))


def _operator_kwargs(fixture: Mapping[str, Any]) -> dict[str, float]:
    return {
        "tone_amplitude": float(fixture["tone_amplitude"]),
        "chroma_gain": float(fixture["chroma_gain"]),
        "maximum_rotation_radians": float(fixture["maximum_rotation_radians"]),
        "boundary_epsilon": float(fixture["boundary_epsilon"]),
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_contract(contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    fixture = contract["fixture"]
    gates = contract["gates"]
    axis_bank = np.asarray(fixture["axis_bank"], dtype=np.float64)
    build = _grid(int(fixture["build_grid_size"]), 0.0, 1.0)
    confirmation = _grid(
        int(fixture["confirmation_grid_size"]),
        float(fixture["confirmation_minimum"]),
        float(fixture["confirmation_maximum"]),
    )
    jacobian_points = _grid(
        int(fixture["jacobian_grid_size"]),
        float(fixture["jacobian_minimum"]),
        float(fixture["jacobian_maximum"]),
    )
    operator_kwargs = _operator_kwargs(fixture)
    rows: list[dict[str, Any]] = []
    for truth_index in fixture["truth_axis_indices"]:
        truth_axis = axis_bank[int(truth_index)]
        build_target, _ = apply_cylindrical_operator(build, truth_axis, **operator_kwargs)
        confirmation_target, truth_scale = apply_cylindrical_operator(
            confirmation, truth_axis, **operator_kwargs
        )
        bank_build_errors = []
        for axis in axis_bank:
            output, _ = apply_cylindrical_operator(build, axis, **operator_kwargs)
            bank_build_errors.append(_rmse(output, build_target))
        order = np.argsort(np.asarray(bank_build_errors), kind="stable")
        selected_index = int(order[0])
        selected_output, selected_scale = apply_cylindrical_operator(
            confirmation, axis_bank[selected_index], **operator_kwargs
        )
        fixed_output, _ = apply_cylindrical_operator(
            confirmation, axis_bank[0], **operator_kwargs
        )
        wrong_index = 1 + (
            int(truth_index) - 1 + int(fixture["wrong_axis_offset"])
        ) % (len(axis_bank) - 1)
        wrong_output, _ = apply_cylindrical_operator(
            confirmation, axis_bank[wrong_index], **operator_kwargs
        )
        cubic = _fit_independent_rgb_cubic(
            build, build_target, int(fixture["rgb_curve_degree"])
        )
        cubic_output = _apply_independent_rgb_cubic(confirmation, cubic)
        errors = {
            "identity": _rmse(confirmation, confirmation_target),
            "fixed_equal_rgb_axis": _rmse(fixed_output, confirmation_target),
            "independent_rgb_cubic": _rmse(cubic_output, confirmation_target),
            "wrong_axis_cylindrical": _rmse(wrong_output, confirmation_target),
            "selected_axis_cylindrical_analytical_safe": _rmse(
                selected_output, confirmation_target
            ),
        }
        second_error = float(bank_build_errors[int(order[1])])
        rows.append(
            {
                "truth_axis_index": int(truth_index),
                "selected_axis_index": selected_index,
                "wrong_axis_index": wrong_index,
                "build_errors": [float(value) for value in bank_build_errors],
                "selection_margin_fraction": _improvement(
                    float(bank_build_errors[selected_index]), second_error
                ),
                "errors": errors,
                "improvement_over_fixed_axis": _improvement(
                    errors["selected_axis_cylindrical_analytical_safe"],
                    errors["fixed_equal_rgb_axis"],
                ),
                "improvement_over_rgb_cubic": _improvement(
                    errors["selected_axis_cylindrical_analytical_safe"],
                    errors["independent_rgb_cubic"],
                ),
                "improvement_over_wrong_axis": _improvement(
                    errors["selected_axis_cylindrical_analytical_safe"],
                    errors["wrong_axis_cylindrical"],
                ),
                "minimum_safe_scale": float(np.min(selected_scale)),
                "limited_pixel_fraction": float(np.mean(selected_scale < 1.0 - 1e-12)),
                "truth_selected_scale_max_error": float(
                    np.max(np.abs(truth_scale - selected_scale))
                ),
                "minimum_local_jacobian_determinant": _minimum_jacobian(
                    jacobian_points, axis_bank[selected_index], fixture
                ),
                "new_boundary_fraction": _new_boundary_fraction(
                    confirmation,
                    selected_output,
                    float(fixture["boundary_epsilon"]),
                ),
            }
        )
    selected_errors = [
        row["errors"]["selected_axis_cylindrical_analytical_safe"] for row in rows
    ]
    fixed_improvements = [row["improvement_over_fixed_axis"] for row in rows]
    cubic_improvements = [row["improvement_over_rgb_cubic"] for row in rows]
    wrong_improvements = [row["improvement_over_wrong_axis"] for row in rows]
    gate_results = {
        "axis_recovery": all(row["selected_axis_index"] == row["truth_axis_index"] for row in rows),
        "build_error": max(min(row["build_errors"]) for row in rows)
        <= gates["maximum_selected_axis_build_rmse"],
        "confirmation_error": max(selected_errors)
        <= gates["maximum_selected_axis_confirmation_rmse"],
        "selection_margin": min(row["selection_margin_fraction"] for row in rows)
        >= gates["minimum_axis_selection_margin_fraction"],
        "median_over_fixed": float(np.median(fixed_improvements))
        >= gates["minimum_median_improvement_over_fixed_axis"],
        "worst_over_fixed": min(fixed_improvements)
        >= gates["minimum_worst_improvement_over_fixed_axis"],
        "median_over_rgb_cubic": float(np.median(cubic_improvements))
        >= gates["minimum_median_improvement_over_rgb_cubic"],
        "worst_over_rgb_cubic": min(cubic_improvements)
        >= gates["minimum_worst_improvement_over_rgb_cubic"],
        "median_over_wrong_axis": float(np.median(wrong_improvements))
        >= gates["minimum_median_improvement_over_wrong_axis"],
        "safe_scale": min(row["minimum_safe_scale"] for row in rows)
        >= gates["minimum_safe_scale"],
        "limited_fraction": max(row["limited_pixel_fraction"] for row in rows)
        <= gates["maximum_limited_pixel_fraction"],
        "positive_jacobian": min(row["minimum_local_jacobian_determinant"] for row in rows)
        >= gates["minimum_local_jacobian_determinant"],
        "boundary": max(row["new_boundary_fraction"] for row in rows)
        <= gates["maximum_new_boundary_fraction"],
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2ca0_learnable_axis_cylindrical_operator_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract": contract_path.as_posix(),
        "contract_sha256": _sha256(contract_path),
        "primary_source": contract["primary_source"],
        "fixture": fixture,
        "rows": rows,
        "summary": {
            "selected_error": _summary(selected_errors),
            "improvement_over_fixed_axis": _summary(fixed_improvements),
            "improvement_over_rgb_cubic": _summary(cubic_improvements),
            "improvement_over_wrong_axis": _summary(wrong_improvements),
            "minimum_safe_scale": min(row["minimum_safe_scale"] for row in rows),
            "maximum_limited_pixel_fraction": max(
                row["limited_pixel_fraction"] for row in rows
            ),
            "minimum_local_jacobian_determinant": min(
                row["minimum_local_jacobian_determinant"] for row in rows
            ),
        },
        "gates": gate_results,
        "passed": passed,
        "decision": (
            "retain_synthetic_capacity_open_independent_photographic_challenge"
            if passed
            else "close_exact_learnable_axis_cylindrical_representation"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity = dict(report)
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
