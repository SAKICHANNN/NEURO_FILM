#!/usr/bin/env python
"""Run the frozen U5.R2AL1 analytic colour-sector capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.analytic_chroma_sector_curves import (  # noqa: E402
    AnalyticChromaSectorCurveOperator,
    GlobalBernsteinCurveOperator,
    fit_analytic_chroma_sector_curve_operator,
    fit_global_bernstein_curve_operator,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    _smoothness_loss,
    finite_difference_jacobians,
)
from src.roll2film.time_dependent_cube_flow import (  # noqa: E402
    _sample_grid_torch_vectorized,
)


REPORT_SCHEMA = "neuro-film.u5.r2al1.capacity-report.v2"
REPEAT_SCHEMA = "neuro-film.u5.r2al1.repeat-decision.v2"
V1_SHA256 = "53fe4134ba384d0523d5b2d87c6b0029e15c0e414dcdf5501deddcef71780559"
V2_SHA256 = "6adaba33dbcbd9ec84de81cb35b75534da8a40127244e9c0d1981d77d3aef00f"
V3_SHA256 = "e4e8e812f1bbffe8a4124066e34cb6e3dcd0d549d807eaa9297d9cdaa6f26384"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("formal AL1 audit requires a clean tracked worktree")


def _load_contracts(
    v3_path: Path,
    *,
    expected_v3_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[bytes, bytes, bytes]]:
    v3_bytes = v3_path.read_bytes()
    if _sha256(v3_bytes) != V3_SHA256:
        raise RuntimeError("AL1 v3 config hash mismatch")
    if expected_v3_sha256 and _sha256(v3_bytes) != expected_v3_sha256:
        raise RuntimeError("AL1 v3 config does not match caller expectation")
    v3 = json.loads(v3_bytes)
    v1_path = ROOT / str(v3["base_v1_path"])
    v2_path = ROOT / str(v3["base_v2_path"])
    v1_bytes, v2_bytes = v1_path.read_bytes(), v2_path.read_bytes()
    if _sha256(v1_bytes) != V1_SHA256 or _sha256(v2_bytes) != V2_SHA256:
        raise RuntimeError("AL1 base config hash mismatch")
    v1, v2 = json.loads(v1_bytes), json.loads(v2_bytes)
    if (
        v2["base_config_raw_sha256"] != V1_SHA256
        or v3["base_v1_raw_sha256"] != V1_SHA256
        or v3["base_v2_raw_sha256"] != V2_SHA256
    ):
        raise RuntimeError("AL1 contract hash chain mismatch")
    return v1, v2, v3, (v1_bytes, v2_bytes, v3_bytes)


def _grid(axis: np.ndarray) -> np.ndarray:
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _development_grid(v3: dict[str, Any]) -> np.ndarray:
    size = int(v3["audit_geometry"]["development_grid"]["axis_size"])
    return _grid(np.linspace(0.0, 1.0, size, dtype=np.float64))


def _confirmation_grid(v3: dict[str, Any]) -> np.ndarray:
    geometry = v3["audit_geometry"]["confirmation_grid"]
    interior_size = int(geometry["interior_axis_size"])
    interior = _grid(
        (np.arange(interior_size, dtype=np.float64) + 0.5) / interior_size
    )
    boundary_axis = np.linspace(
        0.0,
        1.0,
        int(geometry["boundary_axis_size"]),
        dtype=np.float64,
    )
    mesh = np.stack(
        np.meshgrid(boundary_axis, boundary_axis, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    rows = [row.copy() for row in interior]
    seen = {row.tobytes() for row in rows}
    for channel in range(3):
        other = [index for index in range(3) if index != channel]
        for boundary in (0.0, 1.0):
            face = np.empty((len(mesh), 3), dtype=np.float64)
            face[:, channel] = boundary
            face[:, other[0]] = mesh[:, 0]
            face[:, other[1]] = mesh[:, 1]
            for row in face:
                key = row.tobytes()
                if key not in seen:
                    seen.add(key)
                    rows.append(row.copy())
    return np.asarray(rows, dtype=np.float64)


def _lab_ab(rgb: np.ndarray, v2: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    math = v2["exact_colour_math"]
    matrix = np.asarray(math["linear_srgb_to_xyz_d65"], dtype=np.float64)
    white = np.asarray(math["d65_white"], dtype=np.float64)
    relative = (rgb @ matrix.T) / white
    transformed = np.where(
        relative > float(math["lab_epsilon"]),
        np.cbrt(relative),
        float(math["lab_linear_scale"]) * relative + float(math["lab_offset"]),
    )
    return (
        500.0 * (transformed[..., 0] - transformed[..., 1]),
        200.0 * (transformed[..., 1] - transformed[..., 2]),
    )


def _truth_weights(
    rgb: np.ndarray,
    rows: list[dict[str, Any]],
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> np.ndarray:
    truth = v1["synthetic_truth"]
    lab_a, lab_b = _lab_ab(rgb, v2)
    centres = np.deg2rad(
        np.asarray([row["angle_degrees"] for row in rows], dtype=np.float64)
    )
    denominator = np.sqrt(
        lab_a**2 + lab_b**2 + float(truth["direction_softening_lab"]) ** 2
    )
    scores = float(truth["direction_concentration"]) * (
        lab_a[..., None] * np.cos(centres)
        + lab_b[..., None] * np.sin(centres)
    ) / denominator[..., None]
    scores -= np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= np.sum(probability, axis=-1, keepdims=True)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    opponent_u = (2.0 * red - green - blue) / np.sqrt(6.0)
    opponent_v = (green - blue) / np.sqrt(2.0)
    chroma_squared = opponent_u**2 + opponent_v**2
    half = float(truth["neutral_chroma_half_activation"])
    return (
        chroma_squared[..., None]
        / (chroma_squared[..., None] + half * half)
        * probability
    )


def _truth_stage(
    rgb: np.ndarray,
    rows: list[dict[str, Any]],
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> np.ndarray:
    truth = v1["synthetic_truth"]
    weights = _truth_weights(rgb, rows, v1, v2)
    gains = np.asarray([row["rgb_gain"] for row in rows], dtype=np.float64)
    slopes = np.asarray([row["luma_slope"] for row in rows], dtype=np.float64)
    luma = rgb @ np.asarray(
        v2["exact_colour_math"]["luma_coefficients"], dtype=np.float64
    )
    response = float(truth["gain_scale"]) * np.einsum(
        "...j,jc,...j->...c",
        weights,
        gains,
        1.0 + slopes * (2.0 * luma[..., None] - 1.0),
    )
    result = rgb + rgb * (1.0 - rgb) * response
    if (
        not np.all(np.isfinite(result))
        or np.any(result < -1e-12)
        or np.any(result > 1.0 + 1e-12)
    ):
        raise RuntimeError("AL1 frozen truth escaped the RGB cube")
    return result


def _target(
    name: str,
    rgb: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> np.ndarray:
    truth = v1["synthetic_truth"]
    if name == "stage_a_then_stage_b":
        order = ("stage_a", "stage_b")
    elif name == "stage_b_then_stage_a":
        order = ("stage_b", "stage_a")
    else:
        raise ValueError(f"unsupported AL1 target {name}")
    result = rgb.copy()
    for stage in order:
        result = _truth_stage(result, truth[stage], v1, v2)
    return result


class _TruthOperator:
    def __init__(self, name: str, v1: dict[str, Any], v2: dict[str, Any]):
        self.name, self.v1, self.v2 = name, v1, v2

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return _target(self.name, rgb, self.v1, self.v2)


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _fit_candidate(
    source: np.ndarray,
    target: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> tuple[AnalyticChromaSectorCurveOperator, dict[str, Any]]:
    representation = v1["representation"]
    fit = v1["fit"]
    corrections = v2["fit_corrections"]
    return fit_analytic_chroma_sector_curve_operator(
        source,
        target,
        minimum_control_increment=float(
            representation["minimum_control_increment"]
        ),
        direction_softening_lab=float(
            representation["direction_softening_lab"]
        ),
        direction_concentration=float(
            representation["direction_concentration"]
        ),
        neutral_chroma_half_activation=float(
            representation["neutral_chroma_half_activation"]
        ),
        seed=int(fit["seed"]),
        restarts=int(fit["restarts"]),
        restart_standard_deviation=float(
            corrections["nonzero_restart_standard_deviation"]
        ),
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        identity_regularization=float(
            corrections["candidate_identity_regularization"]
        ),
        gradient_clip_norm=float(corrections["gradient_clip_norm"]),
        thread_count=int(fit["threads"]),
    )


def _fit_global(
    source: np.ndarray,
    target: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> tuple[GlobalBernsteinCurveOperator, dict[str, Any]]:
    representation = v1["representation"]
    fit = v1["fit"]
    corrections = v2["fit_corrections"]
    return fit_global_bernstein_curve_operator(
        source,
        target,
        minimum_control_increment=float(
            representation["minimum_control_increment"]
        ),
        seed=int(fit["seed"]),
        restarts=int(fit["restarts"]),
        restart_standard_deviation=float(
            corrections["nonzero_restart_standard_deviation"]
        ),
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        identity_regularization=float(
            corrections["global_curve_identity_regularization"]
        ),
        gradient_clip_norm=float(corrections["gradient_clip_norm"]),
        thread_count=int(fit["threads"]),
    )


def _fit_stationary_k3(
    source: np.ndarray,
    target: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, Any]]:
    fit = v1["fit"]
    control = fit["controls"]["stationary_cube_flow_k3"]
    corrections = v2["fit_corrections"]
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(fit["threads"]))
    source_tensor = torch.from_numpy(source.copy())
    target_tensor = torch.from_numpy(target.copy())
    best: tuple[float, int, np.ndarray] | None = None
    histories = []
    axis_size = 3
    integration_steps = int(control["integration_steps"])
    step = 1.0 / integration_steps
    coefficient_cap = float(
        corrections["stationary_k3_maximum_absolute_coefficient"]
    )
    for restart in range(int(fit["restarts"])):
        if restart == 0:
            initial = torch.zeros(
                (axis_size, axis_size, axis_size, 3), dtype=torch.float64
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(fit["seed"]) + restart)
            initial = torch.randn(
                (axis_size, axis_size, axis_size, 3),
                generator=generator,
                dtype=torch.float64,
            ) * float(corrections["nonzero_restart_standard_deviation"])
        grid = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam(
            [grid], lr=float(fit["learning_rate"])
        )

        def velocity(values: torch.Tensor) -> torch.Tensor:
            return (
                values
                * (1.0 - values)
                * _sample_grid_torch_vectorized(values, grid)
            )

        objective = float("inf")
        for _ in range(int(fit["steps"])):
            optimizer.zero_grad(set_to_none=True)
            current = source_tensor
            for _integration_index in range(integration_steps):
                k1 = velocity(current)
                k2 = velocity(current + 0.5 * step * k1)
                k3 = velocity(current + 0.5 * step * k2)
                k4 = velocity(current + step * k3)
                current = current + (step / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
            loss = torch.mean((current - target_tensor) ** 2)
            loss = loss + float(
                corrections["stationary_k3_coefficient_l2"]
            ) * torch.mean(grid**2)
            loss = loss + float(
                corrections["stationary_k3_spatial_smoothness_l2"]
            ) * _smoothness_loss(grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [grid], float(corrections["gradient_clip_norm"])
            )
            optimizer.step()
            with torch.no_grad():
                grid.clamp_(-coefficient_cap, coefficient_cap)
            objective = float(loss.detach())
        histories.append({"restart": restart, "final_objective": objective})
        array = grid.detach().cpu().numpy()
        if best is None or (objective, restart) < (best[0], best[1]):
            best = (objective, restart, array.copy())
    assert best is not None
    return (
        CubeDiffeomorphicColourFlow(
            best[2], integration_steps=integration_steps
        ),
        {
            "selected_restart": best[1],
            "objective": best[0],
            "restarts": histories,
        },
    )


def _truth_prerequisites(
    v1: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> dict[str, Any]:
    truth = v1["synthetic_truth"]
    cube = _grid(
        np.linspace(
            0.0,
            1.0,
            int(v3["audit_geometry"]["cube_grid"]["axis_size"]),
            dtype=np.float64,
        )
    )
    names = list(truth["target_orders"])
    outputs = {name: _target(name, cube, v1, v2) for name in names}
    geometry = v3["audit_geometry"]["jacobian_grid"]
    points = _grid(
        np.linspace(
            0.02,
            0.98,
            int(geometry["axis_size"]),
            dtype=np.float64,
        )
    )
    metrics = {}
    for name in names:
        jacobians = finite_difference_jacobians(
            _TruthOperator(name, v1, v2),
            points,
            step=float(geometry["finite_difference_step"]),
        )
        determinants = np.linalg.det(jacobians)
        norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
        metrics[name] = {
            "output_minimum": float(np.min(outputs[name])),
            "output_maximum": float(np.max(outputs[name])),
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "maximum_jacobian_spectral_norm": float(np.max(norms)),
            "negative_jacobian_fraction": float(np.mean(determinants <= 0.0)),
        }
    pair_rmse = _rmse(outputs[names[0]], outputs[names[1]])
    checks = {
        "target_noncommutation": pair_rmse
        >= float(truth["minimum_target_pair_rgb_rmse"]),
        "target_cube": all(
            row["output_minimum"]
            >= -float(truth["maximum_target_cube_violation"])
            and row["output_maximum"]
            <= 1.0 + float(truth["maximum_target_cube_violation"])
            for row in metrics.values()
        ),
        "target_positive_jacobian": all(
            row["minimum_jacobian_determinant"]
            >= float(truth["minimum_target_jacobian_determinant"])
            and row["negative_jacobian_fraction"] == 0.0
            for row in metrics.values()
        ),
        "target_bounded_jacobian": all(
            row["maximum_jacobian_spectral_norm"]
            <= float(truth["maximum_target_jacobian_spectral_norm"])
            for row in metrics.values()
        ),
    }
    return {
        "target_pair_rgb_rmse": pair_rmse,
        "targets": metrics,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _hue_boundary_jump(
    operator: AnalyticChromaSectorCurveOperator,
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> tuple[float, int]:
    geometry = v3["audit_geometry"]["hue_boundary"]
    math = v2["exact_colour_math"]
    matrix = np.stack(
        (
            np.asarray(math["luma_coefficients"], dtype=np.float64),
            np.asarray(math["opponent_u_coefficients"], dtype=np.float64),
            np.asarray(math["opponent_v_coefficients"], dtype=np.float64),
        )
    )
    inverse = np.linalg.inv(matrix)
    pairs = []
    radius = float(geometry["radius_in_opponent_plane"])
    delta = float(geometry["delta_degrees"])
    for luma in geometry["luma_values"]:
        for angle in geometry["bisector_degrees"]:
            rows = []
            for signed in (-delta, delta):
                radians = np.deg2rad(float(angle) + signed)
                rows.append(
                    inverse
                    @ np.asarray(
                        [luma, radius * np.cos(radians), radius * np.sin(radians)]
                    )
                )
            if all(np.all((row >= 0.0) & (row <= 1.0)) for row in rows):
                pairs.append(rows)
    if len(pairs) < int(geometry["minimum_retained_pair_count"]):
        raise RuntimeError("AL1 hue-boundary geometry retained too few pairs")
    values = np.asarray(pairs, dtype=np.float64)
    jumps = np.linalg.norm(
        operator.apply(values[:, 0]) - operator.apply(values[:, 1]), axis=1
    )
    return float(np.max(jumps)), len(pairs)


def _candidate_metrics(
    operator: AnalyticChromaSectorCurveOperator,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    confirm_source: np.ndarray,
    confirm_target: np.ndarray,
    v1: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
) -> dict[str, Any]:
    prediction = operator.apply(confirm_source)
    fit_prediction = operator.apply(fit_source)
    geometry = v3["audit_geometry"]
    identity_axis = np.linspace(
        0.0,
        1.0,
        int(geometry["identity_grid"]["axis_size"]),
        dtype=np.float64,
    )
    identity_points = _grid(identity_axis)
    identity = AnalyticChromaSectorCurveOperator.identity()
    identity_error = float(
        np.max(np.abs(identity.apply(identity_points) - identity_points))
    )
    cube_points = _grid(
        np.linspace(
            0.0,
            1.0,
            int(geometry["cube_grid"]["axis_size"]),
            dtype=np.float64,
        )
    )
    cube_output = operator.apply(cube_points)
    partition = operator.partition(cube_points)
    jacobian_geometry = geometry["jacobian_grid"]
    jacobian_points = _grid(
        np.linspace(
            0.02,
            0.98,
            int(jacobian_geometry["axis_size"]),
            dtype=np.float64,
        )
    )
    jacobians = finite_difference_jacobians(
        operator,
        jacobian_points,
        step=float(jacobian_geometry["finite_difference_step"]),
    )
    determinants = np.linalg.det(jacobians)
    norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    inverse_geometry = geometry["inverse_samples"]
    generator = np.random.default_rng(int(inverse_geometry["seed"]))
    inverse_source = generator.uniform(
        float(inverse_geometry["minimum"]),
        float(inverse_geometry["maximum"]),
        size=(int(inverse_geometry["count"]), 3),
    )
    inverse_converged = True
    try:
        inverse_roundtrip = operator.inverse(
            operator.apply(inverse_source),
            maximum_iterations=int(
                v2["inverse_corrections"]["maximum_iterations"]
            ),
            convergence_tolerance=float(
                v2["inverse_corrections"]["convergence_tolerance"]
            ),
        )
        inverse_error = float(
            np.max(np.abs(inverse_roundtrip - inverse_source))
        )
    except RuntimeError:
        inverse_converged = False
        inverse_error = 1.0
    replay = AnalyticChromaSectorCurveOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    ).apply(confirm_source)
    partitioned = []
    start = 0
    chunk_sizes = list(v1["structural_audit"]["partition_chunk_sizes"])
    chunk_index = 0
    while start < len(confirm_source):
        size = int(chunk_sizes[chunk_index % len(chunk_sizes)])
        partitioned.append(operator.apply(confirm_source[start : start + size]))
        start += size
        chunk_index += 1
    partitioned_output = np.concatenate(partitioned)
    hue_jump, hue_pairs = _hue_boundary_jump(operator, v2, v3)

    full_delta = prediction - confirm_source
    collinearity = 0.0
    order_error = 0.0
    full_distance = np.linalg.norm(full_delta, axis=1)
    previous_distance = np.zeros_like(full_distance)
    for strength in v1["representation"]["strength_values"]:
        payload = operator.to_dict()
        payload["strength"] = float(strength)
        strength_output = AnalyticChromaSectorCurveOperator.from_dict(payload).apply(
            confirm_source
        )
        expected = confirm_source + float(strength) * full_delta
        collinearity = max(
            collinearity, float(np.max(np.abs(strength_output - expected)))
        )
        distance = np.linalg.norm(strength_output - confirm_source, axis=1)
        order_error = max(
            order_error,
            float(np.max(np.maximum(previous_distance - distance, 0.0))),
            float(np.max(np.abs(distance - float(strength) * full_distance))),
        )
        previous_distance = distance

    ramp_geometry = geometry["red_ramp"]
    t_axis = np.linspace(
        0.0,
        1.0,
        int(ramp_geometry["sample_count"]),
        dtype=np.float64,
    )
    ramp = np.stack((t_axis, 0.18 * t_axis, 0.12 * t_axis), axis=1)
    luma_coefficients = np.asarray(
        v2["exact_colour_math"]["luma_coefficients"], dtype=np.float64
    )
    ramp_luma = operator.apply(ramp) @ luma_coefficients
    return {
        "fit_rgb_rmse": _rmse(fit_prediction, fit_target),
        "confirmation_rgb_rmse": _rmse(prediction, confirm_target),
        "confirmation_maximum_absolute_error": float(
            np.max(np.abs(prediction - confirm_target))
        ),
        "identity_maximum_absolute_error": identity_error,
        "partition_sum_maximum_absolute_error": float(
            np.max(np.abs(np.sum(partition, axis=1) - 1.0))
        ),
        "minimum_partition_weight": float(np.min(partition)),
        "neutral_axis_maximum_channel_spread": float(
            np.max(
                np.ptp(
                    operator.apply(
                        np.repeat(
                            np.linspace(0.0, 1.0, 1025)[:, None], 3, axis=1
                        )
                    ),
                    axis=1,
                )
            )
        ),
        "cube_minimum": float(np.min(cube_output)),
        "cube_maximum": float(np.max(cube_output)),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "negative_jacobian_fraction": float(np.mean(determinants <= 0.0)),
        "maximum_jacobian_spectral_norm": float(np.max(norms)),
        "inverse_converged": inverse_converged,
        "inverse_roundtrip_maximum_absolute_error": inverse_error,
        "serialization_replay_maximum_absolute_error": float(
            np.max(np.abs(replay - prediction))
        ),
        "partition_replay_maximum_absolute_error": float(
            np.max(np.abs(partitioned_output - prediction))
        ),
        "hue_boundary_maximum_output_jump": hue_jump,
        "hue_boundary_retained_pair_count": hue_pairs,
        "strength_path_maximum_collinearity_error": collinearity,
        "strength_path_maximum_order_error": order_error,
        "red_ramp_minimum_luma_step": float(np.min(np.diff(ramp_luma))),
        "red_ramp_maximum_second_difference": float(
            np.max(np.abs(np.diff(ramp_luma, n=2)))
        ),
    }


def _reconstruct_checks(report: dict[str, Any], v1: dict[str, Any]) -> dict[str, bool]:
    gates = v1["gates"]
    rows = report["targets"].values()
    candidates = [row["candidate"] for row in rows]
    return {
        **report["truth_prerequisites"]["checks"],
        "identity": all(
            row["identity_maximum_absolute_error"]
            <= float(gates["identity_maximum_absolute_error"])
            for row in candidates
        ),
        "partition": all(
            row["partition_sum_maximum_absolute_error"]
            <= float(gates["partition_sum_maximum_absolute_error"])
            and row["minimum_partition_weight"]
            >= float(gates["minimum_partition_weight"])
            for row in candidates
        ),
        "neutral_axis": all(
            row["neutral_axis_maximum_channel_spread"]
            <= float(gates["neutral_axis_maximum_channel_spread"])
            for row in candidates
        ),
        "cube": all(
            row["cube_minimum"] >= float(gates["cube_minimum"])
            and row["cube_maximum"] <= float(gates["cube_maximum"])
            for row in candidates
        ),
        "positive_jacobian": all(
            row["minimum_jacobian_determinant"]
            >= float(gates["minimum_jacobian_determinant"])
            and row["negative_jacobian_fraction"]
            <= float(gates["maximum_negative_jacobian_fraction"])
            for row in candidates
        ),
        "bounded_jacobian": all(
            row["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_jacobian_spectral_norm"])
            for row in candidates
        ),
        "inverse": all(
            row["inverse_converged"]
            and row["inverse_roundtrip_maximum_absolute_error"]
            <= float(gates["inverse_roundtrip_maximum_absolute_error"])
            for row in candidates
        ),
        "replay": all(
            row["serialization_replay_maximum_absolute_error"]
            <= float(gates["serialization_replay_maximum_absolute_error"])
            and row["partition_replay_maximum_absolute_error"]
            <= float(gates["partition_replay_maximum_absolute_error"])
            for row in candidates
        ),
        "hue_continuity": all(
            row["hue_boundary_maximum_output_jump"]
            <= float(gates["hue_boundary_maximum_output_jump"])
            for row in candidates
        ),
        "strength_path": all(
            row["strength_path_maximum_collinearity_error"]
            <= float(gates["strength_path_maximum_collinearity_error"])
            and row["strength_path_maximum_order_error"]
            <= float(gates["strength_path_maximum_order_error"])
            for row in candidates
        ),
        "red_ramp": all(
            row["red_ramp_minimum_luma_step"]
            >= float(gates["red_ramp_minimum_luma_step"])
            and row["red_ramp_maximum_second_difference"]
            <= float(gates["red_ramp_maximum_second_difference"])
            for row in candidates
        ),
        "absolute_fidelity": all(
            row["confirmation_rgb_rmse"]
            <= float(gates["candidate_confirmation_rgb_rmse_maximum"])
            for row in candidates
        ),
        "global_curve_advantage": all(
            row["candidate_improvement_over_global_curves_fraction"]
            >= float(
                gates[
                    "candidate_improvement_over_global_curves_fraction_minimum"
                ]
            )
            for row in report["targets"].values()
        ),
        "stationary_k3_efficiency": all(
            row["candidate_to_stationary_k3_rmse_ratio"]
            <= float(gates["candidate_to_stationary_k3_rmse_ratio_maximum"])
            for row in report["targets"].values()
        ),
    }


def run_child_audit(
    v1: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
    *,
    software_commit: str,
) -> dict[str, Any]:
    truth_prerequisites = _truth_prerequisites(v1, v2, v3)
    if not truth_prerequisites["all_checks_passed"]:
        raise RuntimeError("AL1 frozen truth prerequisites failed")
    fit_source = _development_grid(v3)
    confirm_source = _confirmation_grid(v3)
    targets: dict[str, Any] = {}
    for name in v1["synthetic_truth"]["target_orders"]:
        fit_target = _target(name, fit_source, v1, v2)
        confirm_target = _target(name, confirm_source, v1, v2)
        candidate, candidate_fit = _fit_candidate(
            fit_source, fit_target, v1, v2
        )
        global_curve, global_fit = _fit_global(
            fit_source, fit_target, v1, v2
        )
        stationary, stationary_fit = _fit_stationary_k3(
            fit_source, fit_target, v1, v2
        )
        candidate_metrics = _candidate_metrics(
            candidate,
            fit_source,
            fit_target,
            confirm_source,
            confirm_target,
            v1,
            v2,
            v3,
        )
        global_rmse = _rmse(global_curve.apply(confirm_source), confirm_target)
        stationary_rmse = _rmse(stationary.apply(confirm_source), confirm_target)
        candidate_rmse = float(candidate_metrics["confirmation_rgb_rmse"])
        targets[name] = {
            "candidate": candidate_metrics,
            "candidate_fit": candidate_fit,
            "global_curves": {
                "confirmation_rgb_rmse": global_rmse,
                "fit": global_fit,
            },
            "stationary_k3": {
                "confirmation_rgb_rmse": stationary_rmse,
                "fit": stationary_fit,
            },
            "candidate_improvement_over_global_curves_fraction": (
                (global_rmse - candidate_rmse) / global_rmse
            ),
            "candidate_to_stationary_k3_rmse_ratio": (
                candidate_rmse / stationary_rmse
            ),
            "candidate_operator": candidate.to_dict(),
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": 2,
        "experiment_id": v3["experiment_id"],
        "software_commit": software_commit,
        "config_v1_sha256": V1_SHA256,
        "config_v2_sha256": V2_SHA256,
        "config_v3_sha256": V3_SHA256,
        "fit_point_count": int(len(fit_source)),
        "confirmation_point_count": int(len(confirm_source)),
        "truth_prerequisites": truth_prerequisites,
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
    if (
        report["schema"] != REPORT_SCHEMA
        or report["schema_version"] != 2
        or report["experiment_id"] != v3["experiment_id"]
        or report["software_commit"] != software_commit
        or report["config_v1_sha256"] != V1_SHA256
        or report["config_v2_sha256"] != V2_SHA256
        or report["config_v3_sha256"] != V3_SHA256
        or report["claim_ceiling"] != v3["claim_ceiling"]
        or list(report["targets"]) != list(
            v1["synthetic_truth"]["target_orders"]
        )
    ):
        raise ValueError("AL1 report identity mismatch")
    reconstructed = _reconstruct_checks(report, v1)
    if report["checks"] != reconstructed:
        raise ValueError("AL1 report checks do not reconstruct")
    if report["all_checks_passed"] is not all(reconstructed.values()):
        raise ValueError("AL1 aggregate decision does not reconstruct")


def _child_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    v1, v2, v3, before = _load_contracts(
        args.config, expected_v3_sha256=args.expected_config_sha256
    )
    software_commit = _git_commit()
    if software_commit != args.expected_software_commit:
        raise RuntimeError("AL1 software commit mismatch")
    report = run_child_audit(v1, v2, v3, software_commit=software_commit)
    validate_report(
        report, v1=v1, v3=v3, software_commit=software_commit
    )
    _, _, _, after = _load_contracts(
        args.config, expected_v3_sha256=args.expected_config_sha256
    )
    if before != after:
        raise RuntimeError("AL1 config bytes changed during child audit")
    _atomic_write(args.child_output, _canonical_json(report))
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    _require_clean_tracked_worktree()
    v1, _v2, v3, before = _load_contracts(
        args.config, expected_v3_sha256=args.expected_config_sha256
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
    first, second = paths[0].read_bytes(), paths[1].read_bytes()
    if first != second:
        raise RuntimeError("AL1 independent reports are not byte-identical")
    report = json.loads(first)
    validate_report(
        report, v1=v1, v3=v3, software_commit=software_commit
    )
    _, _, _, after = _load_contracts(
        args.config, expected_v3_sha256=V3_SHA256
    )
    if before != after:
        raise RuntimeError("AL1 config bytes changed across repeat audit")
    decision = {
        "schema": REPEAT_SCHEMA,
        "schema_version": 2,
        "experiment_id": v3["experiment_id"],
        "software_commit": software_commit,
        "config_v1_sha256": V1_SHA256,
        "config_v2_sha256": V2_SHA256,
        "config_v3_sha256": V3_SHA256,
        "report_sha256": _sha256(first),
        "independent_report_count": 2,
        "reports_byte_identical": True,
        "automatic_checks": report["checks"],
        "automatic_pass": report["all_checks_passed"],
        "decision": (
            "retain_fixed_analytic_chroma_sector_curve_representation"
            if report["all_checks_passed"]
            else "close_fixed_analytic_chroma_sector_curve_representation"
        ),
        "claim_ceiling": v3["claim_ceiling"],
    }
    decision_bytes = _canonical_json(decision)
    _atomic_write(args.output_dir / "repeat_decision.json", decision_bytes)
    print(
        json.dumps(
            {
                "report_sha256": _sha256(first),
                "repeat_decision_sha256": _sha256(decision_bytes),
                "automatic_pass": report["all_checks_passed"],
                "decision": decision["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2al1_analytic_chroma_sector_curve_capacity_v3.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2al1_analytic_chroma_sector_curve_capacity_v3"
        ),
    )
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-software-commit")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child_output is not None:
        if not args.expected_config_sha256 or not args.expected_software_commit:
            parser.error(
                "child mode requires expected config hash and software commit"
            )
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
