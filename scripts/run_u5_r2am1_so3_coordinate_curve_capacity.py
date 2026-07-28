#!/usr/bin/env python
"""Run the frozen U5.R2AM1 SO(3) coordinate-curve capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.analytic_chroma_sector_curves import (  # noqa: E402
    GlobalBernsteinCurveOperator,
    fit_global_bernstein_curve_operator,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    fit_cube_diffeomorphic_colour_flow,
)
from src.roll2film.density_domain import operator_from_config  # noqa: E402
from src.roll2film.positive_film import (  # noqa: E402
    positive_film_operator_from_config,
)
from src.roll2film.so3_coordinate_curves import (  # noqa: E402
    PositiveMatrixBernsteinCurveOperator,
    SO3CoordinateCurveOperator,
    fit_positive_matrix_bernstein_curve_operator,
    fit_so3_coordinate_curve_operator,
)


V1_SHA256 = "2f081922c090ce7dd6c24b2cbe570fed13a6085318fc527ba8d10ed9cb4b7e59"
V2_SHA256 = "5c7a77eb6928a1761f51b0bf22b036cc861cf5911881c819c5ced44c19dac9be"
V3_SHA256 = "2ea192df0bf706edc62f9e5438d41340a61db93ed0d1fab168a78690e349746c"
REPORT_SCHEMA = "neuro-film.u5.r2am1.so3-coordinate-curve-capacity-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2am1.repeat-decision.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(args, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AM1 requires a clean tracked worktree")


def _read_json_with_hash(path: Path, expected: str) -> tuple[dict[str, Any], bytes]:
    value = path.read_bytes()
    if _sha256(value) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(value), value


def _load_contracts(
    v3_path: Path,
    *,
    expected_v3_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[bytes, ...],
]:
    if expected_v3_sha256 != V3_SHA256:
        raise ValueError("AM1 expected v3 hash mismatch")
    v3, v3_bytes = _read_json_with_hash(v3_path, V3_SHA256)
    v1_path = ROOT / str(v3["base_v1_path"])
    v2_path = ROOT / str(v3["base_v2_path"])
    v1, v1_bytes = _read_json_with_hash(v1_path, V1_SHA256)
    v2, v2_bytes = _read_json_with_hash(v2_path, V2_SHA256)
    if (
        v3["base_v1_raw_sha256"] != V1_SHA256
        or v3["base_v2_raw_sha256"] != V2_SHA256
        or v2["base_v1_raw_sha256"] != V1_SHA256
        or v1["experiment_id"] != "u5.r2am1-so3-coordinate-curve-capacity-v1"
        or v2["experiment_id"] != "u5.r2am1-so3-coordinate-curve-capacity-v2"
        or v3["experiment_id"] != "u5.r2am1-so3-coordinate-curve-capacity-v3"
    ):
        raise ValueError("AM1 contract chain mismatch")
    density_source = v1["target_sources"]["density_config"]
    positive_source = v1["target_sources"]["positive_config"]
    density, density_bytes = _read_json_with_hash(
        ROOT / str(density_source["path"]),
        str(density_source["raw_sha256"]),
    )
    positive, positive_bytes = _read_json_with_hash(
        ROOT / str(positive_source["path"]),
        str(positive_source["raw_sha256"]),
    )
    return (
        v1,
        v2,
        v3,
        density,
        positive,
        (v1_bytes, v2_bytes, v3_bytes, density_bytes, positive_bytes),
    )


def _grid(axis_size: int, *, cell_centres: bool) -> np.ndarray:
    if cell_centres:
        axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    else:
        axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _confirmation_points(v1: dict[str, Any]) -> np.ndarray:
    config = v1["audit_geometry"]["confirmation_grid"]
    interior = _grid(int(config["interior_axis_size"]), cell_centres=True)
    axis = np.linspace(
        0.0,
        1.0,
        int(config["boundary_axis_size"]),
        dtype=np.float64,
    )
    rows: list[np.ndarray] = [row.copy() for row in interior]
    seen = {row.tobytes(order="C") for row in rows}
    for channel in range(3):
        other = [index for index in range(3) if index != channel]
        for value in (0.0, 1.0):
            for first in axis:
                for second in axis:
                    row = np.empty(3, dtype=np.float64)
                    row[channel] = value
                    row[other[0]] = first
                    row[other[1]] = second
                    key = row.tobytes(order="C")
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _target_functions(
    v1: dict[str, Any],
    density: dict[str, Any],
    positive: dict[str, Any],
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    result: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for name, target in v1["targets"].items():
        strength = float(target["strength"])
        if target["family"] == "identity":
            result[name] = lambda rgb: rgb.copy()
        elif target["family"] == "u5_r2e0_density":
            operator = operator_from_config(
                density["witnesses"][target["witness"]]
            )
            result[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        elif target["family"] == "u5_r2j0_positive":
            operator = positive_film_operator_from_config(
                positive["witnesses"][target["witness"]]
            )
            result[name] = (
                lambda rgb, op=operator, value=strength: op.apply(
                    rgb, strength=value
                )
            )
        else:
            raise ValueError(f"unsupported AM1 target family: {target['family']}")
    return result


def _finite_difference_jacobians(
    function: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (function(points + offset) - function(points - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _truth_prerequisites(
    v1: dict[str, Any],
    v3: dict[str, Any],
    functions: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> dict[str, Any]:
    geometry = v1["audit_geometry"]
    cube = _grid(int(geometry["cube_grid"]["axis_size"]), cell_centres=False)
    jacobian_config = geometry["jacobian_grid"]
    jacobian_axis = np.linspace(
        float(jacobian_config["axis"].split("_")[1]),
        float(jacobian_config["axis"].split("_")[2]),
        int(jacobian_config["axis_size"]),
        dtype=np.float64,
    )
    jacobian_points = np.stack(
        np.meshgrid(
            jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    step = float(jacobian_config["finite_difference_step"])
    thresholds = v3["audit_execution"]["target_prerequisites"]
    reports: dict[str, Any] = {}
    for name, function in functions.items():
        output = function(cube)
        jacobians = _finite_difference_jacobians(
            function, jacobian_points, step=step
        )
        determinants = np.linalg.det(jacobians)
        norms = np.linalg.svd(jacobians, compute_uv=False)[..., 0]
        reports[name] = {
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "maximum_jacobian_spectral_norm": float(np.max(norms)),
            "finite": bool(
                np.all(np.isfinite(output))
                and np.all(np.isfinite(jacobians))
            ),
        }
    checks = {
        name: bool(
            row["finite"]
            and row["output_minimum"] >= float(thresholds["output_minimum"])
            and row["output_maximum"] <= float(thresholds["output_maximum"])
            and row["minimum_jacobian_determinant"]
            > float(
                thresholds[
                    "minimum_finite_difference_jacobian_determinant_exclusive"
                ]
            )
            and row["maximum_jacobian_spectral_norm"]
            <= float(
                thresholds["maximum_finite_difference_jacobian_spectral_norm"]
            )
        )
        for name, row in reports.items()
    }
    return {
        "targets": reports,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _identity_controls() -> np.ndarray:
    return np.broadcast_to(
        np.linspace(0.0, 1.0, 7, dtype=np.float64), (3, 7)
    ).copy()


def _fit_operators(
    name: str,
    source: np.ndarray,
    target: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> tuple[
    SO3CoordinateCurveOperator,
    GlobalBernsteinCurveOperator,
    PositiveMatrixBernsteinCurveOperator,
    CubeDiffeomorphicColourFlow,
    dict[str, Any],
]:
    candidate_config = v1["candidate"]
    global_config = v1["controls"]["global_curves"]
    positive_config = v1["controls"]["positive_matrix_curves"]
    stationary_config = v1["controls"]["stationary_k3"]
    fit = v1["fit"]
    initialization = v2["restart_initialization"]
    if name == "identity":
        return (
            SO3CoordinateCurveOperator.identity(
                maximum_rotation_angle_radians=float(
                    candidate_config["maximum_rotation_angle_radians"]
                ),
                minimum_control_increment=float(
                    candidate_config["minimum_control_increment"]
                ),
            ),
            GlobalBernsteinCurveOperator(
                _identity_controls(),
                minimum_control_increment=float(
                    global_config["minimum_control_increment"]
                ),
            ),
            PositiveMatrixBernsteinCurveOperator.identity(
                minimum_control_increment=float(
                    positive_config["minimum_control_increment"]
                )
            ),
            CubeDiffeomorphicColourFlow(
                velocity_grid=np.zeros((3, 3, 3, 3), dtype=np.float64),
                integration_steps=int(stationary_config["integration_steps"]),
            ),
            {
                "candidate": {"selected_restart": 0, "identity": True},
                "global_curves": {"selected_restart": 0, "identity": True},
                "positive_matrix_curves": {
                    "selected_restart": 0,
                    "identity": True,
                },
                "stationary_k3": {"selected_restart": 0, "identity": True},
            },
        )
    common = {
        "seed": int(fit["seed"]),
        "restart_standard_deviation": float(
            initialization["restart_standard_deviation"]
        ),
        "steps": int(fit["steps"]),
        "learning_rate": float(fit["learning_rate"]),
        "gradient_clip_norm": float(fit["gradient_clip_norm"]),
        "thread_count": int(fit["thread_count"]),
    }
    candidate, candidate_fit = fit_so3_coordinate_curve_operator(
        source,
        target,
        maximum_rotation_angle_radians=float(
            candidate_config["maximum_rotation_angle_radians"]
        ),
        minimum_control_increment=float(
            candidate_config["minimum_control_increment"]
        ),
        restarts=int(initialization["candidate"]["restart_count"]),
        rotation_l2=float(fit["candidate_rotation_l2"]),
        curve_l2_to_identity=float(
            fit["candidate_curve_l2_to_identity"]
        ),
        **common,
    )
    global_curves, global_fit = fit_global_bernstein_curve_operator(
        source,
        target,
        minimum_control_increment=float(
            global_config["minimum_control_increment"]
        ),
        seed=int(fit["seed"]) + 2000,
        restarts=int(initialization["global_curves"]["restart_count"]),
        restart_standard_deviation=float(
            initialization["restart_standard_deviation"]
        ),
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        identity_regularization=float(
            fit["candidate_curve_l2_to_identity"]
        ),
        gradient_clip_norm=float(fit["gradient_clip_norm"]),
        thread_count=int(fit["thread_count"]),
    )
    positive_curves, positive_fit = (
        fit_positive_matrix_bernstein_curve_operator(
            source,
            target,
            minimum_control_increment=float(
                positive_config["minimum_control_increment"]
            ),
            maximum_matrix_mix=float(
                positive_config["maximum_matrix_mix"]
            ),
            restarts=int(
                initialization["positive_matrix_curves"]["restart_count"]
            ),
            curve_l2_to_identity=float(
                fit["candidate_curve_l2_to_identity"]
            ),
            matrix_l2_to_identity=float(
                fit["positive_matrix_l2_to_identity"]
            ),
            **common,
        )
    )
    stationary = fit_cube_diffeomorphic_colour_flow(
        source,
        target,
        axis_size=int(stationary_config["velocity_grid_axis_size"]),
        integration_steps=int(stationary_config["integration_steps"]),
        maximum_absolute_coefficient=float(
            stationary_config["maximum_absolute_velocity_grid_coefficient"]
        ),
        seed=int(fit["seed"]) + 5000,
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        coefficient_l2=float(fit["stationary_coefficient_l2"]),
        velocity_smoothness_l2=float(
            fit["stationary_smoothness_l2"]
        ),
        gradient_clip_norm=float(fit["gradient_clip_norm"]),
        thread_count=int(fit["thread_count"]),
    )
    return (
        candidate,
        global_curves,
        positive_curves,
        stationary,
        {
            "candidate": candidate_fit,
            "global_curves": global_fit,
            "positive_matrix_curves": positive_fit,
            "stationary_k3": {"selected_restart": 0},
        },
    )


def _hue_pairs(v1: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    config = v1["audit_geometry"]["hue_neighbour"]
    delta = float(config["delta_degrees"])
    bases = np.arange(0.0, 360.0, 10.0, dtype=np.float64)
    luma_coefficients = np.asarray(
        v1["audit_geometry"]["red_ramp"]["luma_coefficients"],
        dtype=np.float64,
    )
    opponent_u = np.asarray(
        [2.0, -1.0, -1.0], dtype=np.float64
    ) / np.sqrt(6.0)
    opponent_v = np.asarray(
        [0.0, 1.0, -1.0], dtype=np.float64
    ) / np.sqrt(2.0)
    system = np.stack((luma_coefficients, opponent_u, opponent_v))
    radius = float(config["radius_in_opponent_plane"])
    minus_rows: list[np.ndarray] = []
    plus_rows: list[np.ndarray] = []
    for luma in config["luma_values"]:
        for base in bases:
            pair = []
            for angle in (base - delta, base + delta):
                radians = np.deg2rad(angle)
                rhs = np.asarray(
                    [
                        float(luma),
                        radius * np.cos(radians),
                        radius * np.sin(radians),
                    ],
                    dtype=np.float64,
                )
                pair.append(np.linalg.solve(system, rhs))
            if all(
                np.all(np.isfinite(row))
                and np.all(row >= 0.0)
                and np.all(row <= 1.0)
                for row in pair
            ):
                minus_rows.append(pair[0])
                plus_rows.append(pair[1])
    return np.asarray(minus_rows), np.asarray(plus_rows)


def _candidate_metrics(
    candidate: SO3CoordinateCurveOperator,
    *,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    confirm_source: np.ndarray,
    confirm_target: np.ndarray,
    global_curves: GlobalBernsteinCurveOperator,
    positive_curves: PositiveMatrixBernsteinCurveOperator,
    stationary: CubeDiffeomorphicColourFlow,
    v1: dict[str, Any],
) -> dict[str, Any]:
    geometry = v1["audit_geometry"]
    prediction = candidate.apply(confirm_source)
    global_prediction = global_curves.apply(confirm_source)
    positive_prediction = positive_curves.apply(confirm_source)
    stationary_prediction = stationary.apply(confirm_source)
    candidate_rmse = _rmse(prediction, confirm_target)
    global_rmse = _rmse(global_prediction, confirm_target)
    positive_rmse = _rmse(positive_prediction, confirm_target)
    stationary_rmse = _rmse(stationary_prediction, confirm_target)

    cube = _grid(int(geometry["cube_grid"]["axis_size"]), cell_centres=False)
    cube_output = candidate.apply(cube)
    jacobian_config = geometry["jacobian_grid"]
    jacobian_axis = np.linspace(
        0.02,
        0.98,
        int(jacobian_config["axis_size"]),
        dtype=np.float64,
    )
    jacobian_points = np.stack(
        np.meshgrid(
            jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    jacobians = _finite_difference_jacobians(
        candidate.apply,
        jacobian_points,
        step=float(jacobian_config["finite_difference_step"]),
    )
    determinants = np.linalg.det(jacobians)
    norms = np.linalg.svd(jacobians, compute_uv=False)[..., 0]

    inverse_config = geometry["inverse_samples"]
    inverse_source = np.random.default_rng(
        int(inverse_config["seed"])
    ).uniform(
        float(inverse_config["minimum"]),
        float(inverse_config["maximum"]),
        size=(int(inverse_config["count"]), 3),
    )
    inverse_converged = True
    try:
        restored = candidate.inverse(
            candidate.apply(inverse_source),
            bisection_iterations=int(
                inverse_config["curve_bisection_iterations"]
            ),
        )
        inverse_error = float(np.max(np.abs(restored - inverse_source)))
    except (RuntimeError, ValueError):
        inverse_converged = False
        inverse_error = 1.0

    matrix = candidate.rotation_matrix
    orthogonality = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant_error = float(abs(np.linalg.det(matrix) - 1.0))
    angle = float(np.linalg.norm(candidate.rotation_vector))
    minimum_increment = float(np.min(np.diff(candidate.control_values, axis=1)))

    neutral_config = geometry["neutral_axis"]
    neutral_axis = np.linspace(
        0.0,
        1.0,
        int(neutral_config["sample_count"]),
        dtype=np.float64,
    )
    neutral_output = candidate.apply(
        np.repeat(neutral_axis[:, None], 3, axis=1)
    )
    neutral_spread = float(
        np.max(np.max(neutral_output, axis=1) - np.min(neutral_output, axis=1))
    )

    hue_minus, hue_plus = _hue_pairs(v1)
    hue_jumps = np.linalg.norm(
        candidate.apply(hue_plus) - candidate.apply(hue_minus), axis=1
    )

    ramp_config = geometry["red_ramp"]
    ramp_t = np.linspace(
        0.0, 1.0, int(ramp_config["sample_count"]), dtype=np.float64
    )
    ramp = np.stack((ramp_t, 0.18 * ramp_t, 0.12 * ramp_t), axis=1)
    ramp_output = candidate.apply(ramp)
    ramp_luma = ramp_output @ np.asarray(
        ramp_config["luma_coefficients"], dtype=np.float64
    )
    luma_steps = np.diff(ramp_luma)
    luma_second = np.diff(ramp_luma, n=2)

    full_delta = prediction - confirm_source
    full_distance = np.linalg.norm(full_delta, axis=1)
    nonzero = full_distance > 1e-15
    collinearity_error = 0.0
    order_error = 0.0
    previous_distance = np.zeros_like(full_distance)
    for strength in geometry["strength_path"]["strengths"]:
        strength_operator = SO3CoordinateCurveOperator(
            rotation_vector=candidate.rotation_vector,
            control_values=candidate.control_values,
            strength=float(strength),
            maximum_rotation_angle_radians=(
                candidate.maximum_rotation_angle_radians
            ),
            minimum_control_increment=candidate.minimum_control_increment,
        )
        strength_output = strength_operator.apply(confirm_source)
        expected = confirm_source + float(strength) * full_delta
        collinearity_error = max(
            collinearity_error,
            float(np.max(np.abs(strength_output - expected))),
        )
        distance = np.linalg.norm(strength_output - confirm_source, axis=1)
        if np.any(nonzero):
            collinearity_error = max(
                collinearity_error,
                float(
                    np.max(
                        np.abs(
                            distance[nonzero] / full_distance[nonzero]
                            - float(strength)
                        )
                    )
                ),
            )
        order_error = max(
            order_error, float(np.max(previous_distance - distance))
        )
        previous_distance = distance

    replay = SO3CoordinateCurveOperator.from_dict(
        json.loads(json.dumps(candidate.to_dict(), sort_keys=True))
    )
    replay_error = float(
        np.max(np.abs(replay.apply(confirm_source) - prediction))
    )
    split = len(confirm_source) // 3
    partitioned = np.concatenate(
        (
            candidate.apply(confirm_source[:split]),
            candidate.apply(confirm_source[split : 2 * split]),
            candidate.apply(confirm_source[2 * split :]),
        ),
        axis=0,
    )
    partition_error = float(np.max(np.abs(partitioned - prediction)))
    return {
        "fit_rgb_rmse": _rmse(candidate.apply(fit_source), fit_target),
        "confirmation_rgb_rmse": candidate_rmse,
        "confirmation_maximum_absolute_error": float(
            np.max(np.abs(prediction - confirm_target))
        ),
        "global_curves_confirmation_rgb_rmse": global_rmse,
        "positive_matrix_curves_confirmation_rgb_rmse": positive_rmse,
        "stationary_k3_confirmation_rgb_rmse": stationary_rmse,
        "improvement_over_global_curves_fraction": (
            float((global_rmse - candidate_rmse) / global_rmse)
            if global_rmse > 1e-15
            else 0.0
        ),
        "improvement_over_positive_matrix_curves_fraction": (
            float((positive_rmse - candidate_rmse) / positive_rmse)
            if positive_rmse > 1e-15
            else 0.0
        ),
        "candidate_to_stationary_k3_rmse_ratio": (
            float(candidate_rmse / stationary_rmse)
            if stationary_rmse > 1e-15
            else 1.0
        ),
        "cube_minimum": float(np.min(cube_output)),
        "cube_maximum": float(np.max(cube_output)),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "negative_jacobian_fraction": float(np.mean(determinants <= 0.0)),
        "maximum_jacobian_spectral_norm": float(np.max(norms)),
        "inverse_converged": inverse_converged,
        "inverse_roundtrip_maximum_absolute_error": inverse_error,
        "rotation_orthogonality_maximum_absolute_error": orthogonality,
        "rotation_determinant_maximum_absolute_error_from_one": (
            determinant_error
        ),
        "rotation_angle_radians": angle,
        "minimum_control_increment": minimum_increment,
        "neutral_axis_maximum_channel_spread": neutral_spread,
        "hue_neighbour_retained_pair_count": int(len(hue_minus)),
        "hue_neighbour_maximum_output_jump": float(np.max(hue_jumps)),
        "red_ramp_minimum_luma_step": float(np.min(luma_steps)),
        "red_ramp_maximum_second_difference": float(
            np.max(np.abs(luma_second))
        ),
        "strength_path_maximum_collinearity_error": collinearity_error,
        "strength_path_maximum_order_error": order_error,
        "serialization_replay_maximum_absolute_error": replay_error,
        "partition_replay_maximum_absolute_error": partition_error,
        "raw_parameter_count": candidate.raw_parameter_count,
        "effective_parameter_count": candidate.effective_parameter_count,
        "positive_matrix_control_determinant": float(
            np.linalg.det(positive_curves.matrix)
        ),
    }


def _reconstruct_checks(
    report: dict[str, Any],
    v1: dict[str, Any],
) -> dict[str, bool]:
    gates = v1["gates"]
    targets = report["targets"]
    rows = list(targets.values())
    styled = [
        row
        for name, row in targets.items()
        if name != "identity"
    ]
    return {
        "parameter_budget": all(
            row["metrics"]["raw_parameter_count"]
            <= int(gates["parameter_budget_raw_maximum"])
            and row["metrics"]["effective_parameter_count"]
            <= int(gates["parameter_budget_effective_maximum"])
            for row in rows
        ),
        "identity": (
            targets["identity"]["metrics"]["confirmation_maximum_absolute_error"]
            <= float(gates["identity_maximum_absolute_error"])
        ),
        "absolute_fidelity": all(
            row["metrics"]["confirmation_rgb_rmse"]
            <= float(gates["candidate_confirmation_rgb_rmse_maximum"])
            for row in styled
        ),
        "global_curve_advantage": all(
            row["metrics"]["improvement_over_global_curves_fraction"]
            >= float(
                gates[
                    "candidate_improvement_over_global_curves_fraction_minimum"
                ]
            )
            for row in styled
        ),
        "positive_matrix_curve_advantage": all(
            row["metrics"][
                "improvement_over_positive_matrix_curves_fraction"
            ]
            >= float(
                gates[
                    "candidate_improvement_over_positive_matrix_curves_fraction_minimum"
                ]
            )
            for row in styled
        ),
        "stationary_k3_efficiency": all(
            row["metrics"]["candidate_to_stationary_k3_rmse_ratio"]
            <= float(gates["candidate_to_stationary_k3_rmse_ratio_maximum"])
            for row in styled
        ),
        "cube": all(
            row["metrics"]["cube_minimum"] >= float(gates["output_minimum"])
            and row["metrics"]["cube_maximum"] <= float(gates["output_maximum"])
            for row in rows
        ),
        "positive_jacobian": all(
            row["metrics"]["minimum_jacobian_determinant"]
            >= float(gates["minimum_finite_difference_jacobian_determinant"])
            for row in rows
        ),
        "bounded_jacobian": all(
            row["metrics"]["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_finite_difference_jacobian_spectral_norm"])
            for row in rows
        ),
        "inverse": all(
            row["metrics"]["inverse_converged"]
            and row["metrics"]["inverse_roundtrip_maximum_absolute_error"]
            <= float(gates["inverse_roundtrip_maximum_absolute_error"])
            for row in rows
        ),
        "rotation": all(
            row["metrics"]["rotation_orthogonality_maximum_absolute_error"]
            <= float(gates["rotation_orthogonality_maximum_absolute_error"])
            and row["metrics"][
                "rotation_determinant_maximum_absolute_error_from_one"
            ]
            <= float(
                gates[
                    "rotation_determinant_maximum_absolute_error_from_one"
                ]
            )
            and row["metrics"]["rotation_angle_radians"]
            <= float(gates["rotation_angle_radians_maximum"])
            for row in rows
        ),
        "curves": all(
            row["metrics"]["minimum_control_increment"]
            >= float(gates["minimum_control_increment"]) - 1e-12
            for row in rows
        ),
        "neutral_axis": all(
            row["metrics"]["neutral_axis_maximum_channel_spread"]
            <= float(gates["neutral_axis_maximum_channel_spread"])
            for row in rows
        ),
        "hue_neighbour": all(
            row["metrics"]["hue_neighbour_retained_pair_count"]
            >= int(
                v1["audit_geometry"]["hue_neighbour"][
                    "minimum_retained_pair_count"
                ]
            )
            and row["metrics"]["hue_neighbour_maximum_output_jump"]
            <= float(gates["hue_neighbour_maximum_output_jump"])
            for row in rows
        ),
        "red_ramp": all(
            row["metrics"]["red_ramp_minimum_luma_step"]
            >= float(gates["red_ramp_minimum_luma_step"])
            and row["metrics"]["red_ramp_maximum_second_difference"]
            <= float(gates["red_ramp_maximum_second_difference"])
            for row in rows
        ),
        "strength_path": all(
            row["metrics"]["strength_path_maximum_collinearity_error"]
            <= float(gates["strength_path_maximum_collinearity_error"])
            and row["metrics"]["strength_path_maximum_order_error"]
            <= float(gates["strength_path_maximum_order_error"])
            for row in rows
        ),
        "replay": all(
            row["metrics"]["serialization_replay_maximum_absolute_error"]
            <= float(gates["serialization_replay_maximum_absolute_error"])
            and row["metrics"]["partition_replay_maximum_absolute_error"]
            <= float(gates["partition_replay_maximum_absolute_error"])
            for row in rows
        ),
        "target_prerequisites": bool(
            report["truth_prerequisites"]["all_checks_passed"]
        ),
    }


def run_child_audit(
    v1: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
    density: dict[str, Any],
    positive: dict[str, Any],
    *,
    software_commit: str,
) -> dict[str, Any]:
    functions = _target_functions(v1, density, positive)
    prerequisites = _truth_prerequisites(v1, v3, functions)
    if not prerequisites["all_checks_passed"]:
        raise RuntimeError("AM1 target truth prerequisites failed before fit")
    fit_source = _grid(
        int(v1["fit"]["development_grid"]["axis_size"]),
        cell_centres=False,
    )
    confirm_source = _confirmation_points(v1)
    targets: dict[str, Any] = {}
    for name, function in functions.items():
        fit_target = function(fit_source)
        confirm_target = function(confirm_source)
        (
            candidate,
            global_curves,
            positive_curves,
            stationary,
            histories,
        ) = _fit_operators(
            name, fit_source, fit_target, v1, v2
        )
        metrics = _candidate_metrics(
            candidate,
            fit_source=fit_source,
            fit_target=fit_target,
            confirm_source=confirm_source,
            confirm_target=confirm_target,
            global_curves=global_curves,
            positive_curves=positive_curves,
            stationary=stationary,
            v1=v1,
        )
        targets[name] = {
            "candidate": candidate.to_dict(),
            "global_curves": global_curves.to_dict(),
            "positive_matrix_curves": positive_curves.to_dict(),
            "stationary_k3": stationary.to_dict(),
            "fit": histories,
            "metrics": metrics,
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "experiment_id": v3["experiment_id"],
        "software_commit": software_commit,
        "config_v1_sha256": V1_SHA256,
        "config_v2_sha256": V2_SHA256,
        "config_v3_sha256": V3_SHA256,
        "target_source_sha256": {
            "density": v1["target_sources"]["density_config"]["raw_sha256"],
            "positive": v1["target_sources"]["positive_config"]["raw_sha256"],
        },
        "fit_point_count": int(len(fit_source)),
        "confirmation_point_count": int(len(confirm_source)),
        "truth_prerequisites": prerequisites,
        "targets": targets,
        "checks": {},
        "all_checks_passed": False,
        "claim_ceiling": v3["claim_ceiling"],
    }
    report["checks"] = _reconstruct_checks(report, v1)
    report["all_checks_passed"] = all(report["checks"].values())
    return report


def validate_report(
    report: dict[str, Any],
    *,
    v1: dict[str, Any],
    v3: dict[str, Any],
    software_commit: str,
) -> None:
    expected_target_sources = {
        "density": v1["target_sources"]["density_config"]["raw_sha256"],
        "positive": v1["target_sources"]["positive_config"]["raw_sha256"],
    }
    expected_fit_count = int(v1["fit"]["development_grid"]["axis_size"]) ** 3
    expected_confirmation_count = len(_confirmation_points(v1))
    if (
        report["schema"] != REPORT_SCHEMA
        or report["schema_version"] != 1
        or report["experiment_id"] != v3["experiment_id"]
        or report["software_commit"] != software_commit
        or report["config_v1_sha256"] != V1_SHA256
        or report["config_v2_sha256"] != V2_SHA256
        or report["config_v3_sha256"] != V3_SHA256
        or report["target_source_sha256"] != expected_target_sources
        or report["fit_point_count"] != expected_fit_count
        or report["confirmation_point_count"] != expected_confirmation_count
        or report["claim_ceiling"] != v3["claim_ceiling"]
        or list(report["targets"]) != list(v1["targets"])
    ):
        raise ValueError("AM1 report identity mismatch")
    reconstructed = _reconstruct_checks(report, v1)
    if report["checks"] != reconstructed:
        raise ValueError("AM1 report checks do not reconstruct")
    if report["all_checks_passed"] is not all(reconstructed.values()):
        raise ValueError("AM1 aggregate decision does not reconstruct")


def _child_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    v1, v2, v3, density, positive, before = _load_contracts(
        args.config,
        expected_v3_sha256=args.expected_config_sha256,
    )
    software_commit = _git_commit()
    if software_commit != args.expected_software_commit:
        raise RuntimeError("AM1 software commit mismatch")
    report = run_child_audit(
        v1,
        v2,
        v3,
        density,
        positive,
        software_commit=software_commit,
    )
    validate_report(
        report, v1=v1, v3=v3, software_commit=software_commit
    )
    _, _, _, _, _, after = _load_contracts(
        args.config,
        expected_v3_sha256=args.expected_config_sha256,
    )
    if before != after:
        raise RuntimeError("AM1 source bytes changed during child audit")
    _atomic_write(args.child_output, _canonical_json(report))
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    v1, _v2, v3, _density, _positive, before = _load_contracts(
        args.config,
        expected_v3_sha256=args.expected_config_sha256,
    )
    software_commit = _git_commit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        args.output_dir / "run_a" / "report.json",
        args.output_dir / "run_b" / "report.json",
    ]
    for path in paths:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config.resolve()),
                "--expected-config-sha256",
                V3_SHA256,
                "--expected-software-commit",
                software_commit,
                "--child-output",
                str(path.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    first_bytes = paths[0].read_bytes()
    second_bytes = paths[1].read_bytes()
    if first_bytes != second_bytes:
        raise RuntimeError("AM1 child reports are not byte-identical")
    report = json.loads(first_bytes)
    validate_report(
        report, v1=v1, v3=v3, software_commit=software_commit
    )
    _, _, _, _, _, after = _load_contracts(
        args.config,
        expected_v3_sha256=args.expected_config_sha256,
    )
    if before != after:
        raise RuntimeError("AM1 source bytes changed during parent audit")
    automatic_pass = bool(report["all_checks_passed"])
    decision = {
        "schema": REPEAT_SCHEMA,
        "schema_version": 1,
        "experiment_id": v3["experiment_id"],
        "software_commit": software_commit,
        "config_v1_sha256": V1_SHA256,
        "config_v2_sha256": V2_SHA256,
        "config_v3_sha256": V3_SHA256,
        "independent_report_count": 2,
        "reports_byte_identical": True,
        "report_sha256": _sha256(first_bytes),
        "automatic_checks": report["checks"],
        "automatic_pass": automatic_pass,
        "decision": (
            v3["repeat_decision"]["pass"]
            if automatic_pass
            else v3["repeat_decision"]["fail"]
        ),
        "claim_ceiling": v3["claim_ceiling"],
    }
    decision_bytes = _canonical_json(decision)
    _atomic_write(args.output_dir / "repeat_decision.json", decision_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": automatic_pass,
                "decision": decision["decision"],
                "repeat_decision_sha256": _sha256(decision_bytes),
                "report_sha256": _sha256(first_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if automatic_pass else int(
        v3["repeat_decision"]["valid_failure_exit_code"]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2am1_so3_coordinate_curve_capacity_v3.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-software-commit")
    parser.add_argument("--child-output", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2am1_so3_coordinate_curve_capacity_v3",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.child_output is not None:
        if not args.expected_software_commit:
            raise ValueError("child execution requires software commit")
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
