from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.roll2film.so3_coordinate_curves import (
    PositiveMatrixBernsteinCurveOperator,
    SO3CoordinateCurveOperator,
    _apply_so3_torch,
    cube_projection_bounds,
    fit_positive_matrix_bernstein_curve_operator,
    fit_so3_coordinate_curve_operator,
    rotation_matrix_from_vector,
)
from scripts.run_u5_r2am1_so3_coordinate_curve_capacity import (
    V3_SHA256,
    _canonical_json,
    _confirmation_points,
    _hue_pairs,
    _load_contracts,
    _target_functions,
    _truth_prerequisites,
)


def _controls() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.14, 0.30, 0.48, 0.67, 0.84, 1.0],
            [0.0, 0.18, 0.35, 0.52, 0.68, 0.83, 1.0],
            [0.0, 0.12, 0.27, 0.45, 0.65, 0.82, 1.0],
        ],
        dtype=np.float64,
    )


def _grid(axis_size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def test_rotation_and_projection_contract() -> None:
    vector = np.asarray([0.12, -0.08, 0.05], dtype=np.float64)
    matrix = rotation_matrix_from_vector(vector)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=2e-16)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=2e-16)
    lower, upper = cube_projection_bounds(matrix)
    projected = _grid(2) @ matrix
    np.testing.assert_array_equal(lower, np.min(projected, axis=0))
    np.testing.assert_array_equal(upper, np.max(projected, axis=0))


def test_identity_is_exact_and_serialization_replays() -> None:
    rows = _grid(9)
    operator = SO3CoordinateCurveOperator.identity()
    np.testing.assert_array_equal(operator.apply(rows), rows)
    replay = SO3CoordinateCurveOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    np.testing.assert_array_equal(replay.apply(rows), rows)
    np.testing.assert_array_equal(replay.inverse(rows), rows)
    assert operator.raw_parameter_count == 21
    assert operator.effective_parameter_count == 18


def test_nontrivial_operator_matches_torch_and_inverts() -> None:
    rows = np.random.default_rng(72832).uniform(0.02, 0.98, size=(128, 3))
    operator = SO3CoordinateCurveOperator(
        rotation_vector=np.asarray([0.12, -0.08, 0.05]),
        control_values=_controls(),
    )
    numpy_output = operator.apply(rows)
    torch_output = _apply_so3_torch(
        torch.from_numpy(rows),
        torch.from_numpy(operator.rotation_vector.copy()),
        torch.from_numpy(operator.control_values.copy()),
    ).detach().numpy()
    np.testing.assert_allclose(numpy_output, torch_output, atol=5e-16)
    restored = operator.inverse(numpy_output)
    np.testing.assert_allclose(restored, rows, atol=2e-15)
    replay = SO3CoordinateCurveOperator.from_dict(operator.to_dict())
    np.testing.assert_array_equal(replay.apply(rows), numpy_output)


def test_strength_is_collinear_and_partial_inverse_is_rejected() -> None:
    rows = np.random.default_rng(12).uniform(0.1, 0.9, size=(32, 3))
    full = SO3CoordinateCurveOperator(
        rotation_vector=np.asarray([0.1, -0.06, 0.03]),
        control_values=_controls(),
    ).apply(rows)
    half_operator = SO3CoordinateCurveOperator(
        rotation_vector=np.asarray([0.1, -0.06, 0.03]),
        control_values=_controls(),
        strength=0.5,
    )
    np.testing.assert_allclose(
        half_operator.apply(rows), 0.5 * rows + 0.5 * full, atol=2e-16
    )
    with pytest.raises(ValueError, match="partial-strength"):
        half_operator.inverse(half_operator.apply(rows))


def test_positive_matrix_control_is_bounded_and_replays() -> None:
    matrix = np.asarray(
        [[0.9, 0.06, 0.04], [0.05, 0.9, 0.05], [0.04, 0.06, 0.9]],
        dtype=np.float64,
    )
    operator = PositiveMatrixBernsteinCurveOperator(
        control_values=_controls(),
        matrix=matrix,
    )
    output = operator.apply(_grid(11))
    assert float(np.min(output)) >= 0.0
    assert float(np.max(output)) <= 1.0
    replay = PositiveMatrixBernsteinCurveOperator.from_dict(operator.to_dict())
    np.testing.assert_array_equal(replay.apply(_grid(7)), operator.apply(_grid(7)))


def test_reduced_fit_is_deterministic_and_valid() -> None:
    source = _grid(5)
    truth = PositiveMatrixBernsteinCurveOperator(
        control_values=_controls(),
        matrix=np.asarray(
            [[0.92, 0.05, 0.03], [0.04, 0.92, 0.04], [0.03, 0.05, 0.92]],
            dtype=np.float64,
        ),
    )
    target = truth.apply(source)
    kwargs = {
        "maximum_rotation_angle_radians": 0.45,
        "minimum_control_increment": 0.02,
        "seed": 72831,
        "restarts": 2,
        "restart_standard_deviation": 0.05,
        "steps": 40,
        "learning_rate": 0.03,
        "rotation_l2": 1e-6,
        "curve_l2_to_identity": 1e-6,
        "gradient_clip_norm": 10.0,
        "thread_count": 1,
    }
    first, first_fit = fit_so3_coordinate_curve_operator(
        source, target, **kwargs
    )
    second, second_fit = fit_so3_coordinate_curve_operator(
        source, target, **kwargs
    )
    assert first.to_dict() == second.to_dict()
    assert first_fit == second_fit
    assert np.sqrt(np.mean((first.apply(source) - target) ** 2)) < 0.04

    positive, history = fit_positive_matrix_bernstein_curve_operator(
        source,
        target,
        minimum_control_increment=0.02,
        maximum_matrix_mix=0.35,
        seed=72831,
        restarts=2,
        restart_standard_deviation=0.05,
        steps=20,
        learning_rate=0.03,
        curve_l2_to_identity=1e-6,
        matrix_l2_to_identity=1e-6,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    assert history["selected_restart"] in (0, 1)
    assert np.all(np.isfinite(positive.apply(source)))


def test_invalid_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="rotation vector"):
        SO3CoordinateCurveOperator(
            rotation_vector=np.asarray([1.0, 0.0, 0.0]),
            control_values=_controls(),
        )
    invalid = _controls()
    invalid[0, 2] = invalid[0, 1]
    with pytest.raises(ValueError, match="positive increments"):
        SO3CoordinateCurveOperator(
            rotation_vector=np.zeros(3),
            control_values=invalid,
        )
    with pytest.raises(ValueError, match="orientation"):
        PositiveMatrixBernsteinCurveOperator(
            control_values=_controls(),
            matrix=np.asarray(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )


def test_frozen_contract_chain_truth_and_audit_geometry() -> None:
    root = Path(__file__).resolve().parents[1]
    v1, _v2, v3, density, positive, _ = _load_contracts(
        root / "configs/u5_r2am1_so3_coordinate_curve_capacity_v3.json",
        expected_v3_sha256=V3_SHA256,
    )
    functions = _target_functions(v1, density, positive)
    prerequisites = _truth_prerequisites(v1, v3, functions)
    assert prerequisites["all_checks_passed"]
    assert list(prerequisites["targets"]) == list(v1["targets"])
    for row in prerequisites["targets"].values():
        assert row["finite"]
        assert row["output_minimum"] >= -1e-12
        assert row["output_maximum"] <= 1.0 + 1e-12
        assert row["minimum_jacobian_determinant"] > 0.0
        assert row["maximum_jacobian_spectral_norm"] <= 8.0

    confirmation = _confirmation_points(v1)
    assert len(confirmation) == 3063
    assert len({row.tobytes(order="C") for row in confirmation}) == len(
        confirmation
    )
    hue_minus, hue_plus = _hue_pairs(v1)
    assert len(hue_minus) == len(hue_plus) == 86
    assert np.all(np.isfinite(hue_minus))
    assert np.all(np.isfinite(hue_plus))


def test_canonical_report_target_order_matches_config_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    v1, _v2, _v3, _density, _positive, _ = _load_contracts(
        root / "configs/u5_r2am1_so3_coordinate_curve_capacity_v3.json",
        expected_v3_sha256=V3_SHA256,
    )
    persisted = json.loads(
        _canonical_json({"targets": {name: {} for name in v1["targets"]}})
    )
    assert list(persisted["targets"]) != list(v1["targets"])
    assert sorted(persisted["targets"]) == sorted(v1["targets"])
