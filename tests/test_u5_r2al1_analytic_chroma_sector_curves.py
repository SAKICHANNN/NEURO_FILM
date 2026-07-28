from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.roll2film.analytic_chroma_sector_curves import (
    AnalyticChromaSectorCurveOperator,
    GlobalBernsteinCurveOperator,
    _apply_candidate_torch,
    fit_analytic_chroma_sector_curve_operator,
    fit_global_bernstein_curve_operator,
)
from src.roll2film.cube_diffeomorphic_flow import finite_difference_jacobians
from scripts.run_u5_r2al1_analytic_chroma_sector_curve_capacity import (
    V3_SHA256,
    _confirmation_grid,
    _load_contracts,
    _truth_prerequisites,
)


def _grid(size: int, *, interior: bool = False) -> np.ndarray:
    axis = (
        np.linspace(0.05, 0.95, size, dtype=np.float64)
        if interior
        else np.linspace(0.0, 1.0, size, dtype=np.float64)
    )
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _curved_operator() -> AnalyticChromaSectorCurveOperator:
    identity = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    achromatic = identity.copy()
    achromatic[1:-1] += np.asarray([0.015, 0.02, 0.0, -0.015, -0.01])
    chromatic = np.broadcast_to(identity, (5, 3, 7)).copy()
    for sector in range(5):
        channel = sector % 3
        direction = 1.0 if sector % 2 == 0 else -1.0
        chromatic[sector, channel, 1:-1] += direction * np.asarray(
            [0.01, 0.018, 0.02, 0.014, 0.006]
        )
    return AnalyticChromaSectorCurveOperator(
        achromatic_control_values=achromatic,
        chromatic_control_values=chromatic,
    )


def test_identity_partition_and_neutral_axis_are_exact() -> None:
    operator = AnalyticChromaSectorCurveOperator.identity()
    rgb = _grid(9)
    assert np.max(np.abs(operator.apply(rgb) - rgb)) < 1e-15
    weights = operator.partition(rgb)
    assert float(weights.min()) >= 0.0
    assert np.max(np.abs(weights.sum(axis=-1) - 1.0)) < 1e-15

    neutral = np.repeat(
        np.linspace(0.0, 1.0, 257, dtype=np.float64)[:, None],
        3,
        axis=1,
    )
    neutral_weights = operator.partition(neutral)
    assert np.array_equal(neutral_weights[:, 0], np.ones(len(neutral)))
    assert np.array_equal(neutral_weights[:, 1:], np.zeros((len(neutral), 5)))


def test_replay_partition_strength_inverse_and_orientation() -> None:
    operator = _curved_operator()
    rgb = _grid(11)
    rows = rgb.reshape(-1, 3)
    output = operator.apply(rows)
    replay = AnalyticChromaSectorCurveOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(output, replay.apply(rows))
    partitioned = np.concatenate(
        [
            operator.apply(rows[:17]),
            operator.apply(rows[17:274]),
            operator.apply(rows[274:]),
        ]
    )
    assert np.array_equal(output, partitioned)
    assert np.max(np.abs(operator.inverse(output) - rows)) < 1e-10
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0

    points = _grid(7, interior=True).reshape(-1, 3)
    jacobians = finite_difference_jacobians(operator, points, step=1e-6)
    assert float(np.linalg.det(jacobians).min()) > 0.7

    full_delta = output - rows
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        payload = operator.to_dict()
        payload["strength"] = strength
        strength_operator = AnalyticChromaSectorCurveOperator.from_dict(payload)
        assert np.max(
            np.abs(strength_operator.apply(rows) - (rows + strength * full_delta))
        ) < 2e-15


def test_numpy_and_torch_candidate_paths_match() -> None:
    operator = _curved_operator()
    rows = _grid(8, interior=True).reshape(-1, 3)
    torch_output = _apply_candidate_torch(
        torch.from_numpy(rows.copy()),
        torch.from_numpy(operator.achromatic_control_values.copy()),
        torch.from_numpy(operator.chromatic_control_values.copy()),
        direction_softening_lab=operator.direction_softening_lab,
        direction_concentration=operator.direction_concentration,
        neutral_chroma_half_activation=operator.neutral_chroma_half_activation,
    ).numpy()
    assert np.max(np.abs(operator.apply(rows) - torch_output)) < 2e-15


def test_deterministic_candidate_and_global_fits_reduce_error() -> None:
    source = _grid(5, interior=True).reshape(-1, 3)
    teacher = _curved_operator()
    target = teacher.apply(source)
    arguments = dict(
        minimum_control_increment=0.04,
        seed=71281,
        restarts=2,
        restart_standard_deviation=0.03,
        steps=80,
        learning_rate=0.03,
        identity_regularization=1e-4,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    first, first_audit = fit_analytic_chroma_sector_curve_operator(
        source,
        target,
        direction_softening_lab=4.0,
        direction_concentration=4.0,
        neutral_chroma_half_activation=0.08,
        **arguments,
    )
    second, second_audit = fit_analytic_chroma_sector_curve_operator(
        source,
        target,
        direction_softening_lab=4.0,
        direction_concentration=4.0,
        neutral_chroma_half_activation=0.08,
        **arguments,
    )
    assert first_audit == second_audit
    assert np.array_equal(
        first.achromatic_control_values, second.achromatic_control_values
    )
    assert np.array_equal(
        first.chromatic_control_values, second.chromatic_control_values
    )
    identity_error = float(np.sqrt(np.mean((source - target) ** 2)))
    candidate_error = float(
        np.sqrt(np.mean((first.apply(source) - target) ** 2))
    )
    assert candidate_error < 0.35 * identity_error

    global_operator, _ = fit_global_bernstein_curve_operator(
        source, target, **arguments
    )
    assert isinstance(global_operator, GlobalBernsteinCurveOperator)
    assert float(
        np.sqrt(np.mean((global_operator.apply(source) - target) ** 2))
    ) < identity_error


def test_validation_rejects_nonmonotone_or_wrong_schema() -> None:
    payload = AnalyticChromaSectorCurveOperator.identity().to_dict()
    payload["achromatic_control_values"][2] = 0.01
    with np.testing.assert_raises(ValueError):
        AnalyticChromaSectorCurveOperator.from_dict(payload)
    with np.testing.assert_raises(ValueError):
        GlobalBernsteinCurveOperator.from_dict(
            {"schema": "wrong", "control_values": np.zeros((3, 7)).tolist()}
        )


def test_frozen_contract_chain_truth_and_confirmation_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    v1, v2, v3, _ = _load_contracts(
        root
        / "configs/u5_r2al1_analytic_chroma_sector_curve_capacity_v3.json",
        expected_v3_sha256=V3_SHA256,
    )
    prerequisites = _truth_prerequisites(v1, v2, v3)
    assert prerequisites["all_checks_passed"]
    assert prerequisites["target_pair_rgb_rmse"] > 0.002
    for row in prerequisites["targets"].values():
        assert row["minimum_jacobian_determinant"] > 0.5
        assert row["maximum_jacobian_spectral_norm"] < 3.0
        assert row["negative_jacobian_fraction"] == 0.0
    confirmation = _confirmation_grid(v3)
    assert len({row.tobytes() for row in confirmation}) == len(confirmation)
    assert np.any(confirmation == 0.0)
    assert np.any(confirmation == 1.0)
