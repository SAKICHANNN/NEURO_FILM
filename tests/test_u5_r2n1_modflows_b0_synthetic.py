from __future__ import annotations

import numpy as np
import pytest

from src.eval.modflows_b0_synthetic import (
    FLOW_PARAMETER_COUNT,
    ModFlowVelocity,
    integrate_modflow_rk4,
    transfer_modflows,
)


def _constant_velocity(red: float, green: float, blue: float) -> np.ndarray:
    parameters = np.zeros(FLOW_PARAMETER_COUNT, dtype=np.float64)
    parameters[-3:] = (red, green, blue)
    return parameters


def test_parameter_layout_and_zero_velocity_are_exact() -> None:
    parameters = np.zeros(FLOW_PARAMETER_COUNT, dtype=np.float64)
    velocity = ModFlowVelocity.from_vector(parameters)
    points = np.asarray([[0.1, 0.2, 0.3], [0.8, 0.5, 0.2]])
    assert np.array_equal(velocity(points, 0.4), np.zeros_like(points))
    output = integrate_modflow_rk4(
        points,
        velocity,
        start_time=0.0,
        end_time=1.0,
        steps=8,
    )
    assert np.array_equal(output, points)


def test_constant_velocity_and_reverse_have_expected_displacement() -> None:
    parameters = _constant_velocity(0.1, -0.2, 0.3)
    velocity = ModFlowVelocity.from_vector(parameters)
    points = np.asarray([[0.4, 0.5, 0.2]])
    forward = integrate_modflow_rk4(
        points,
        velocity,
        start_time=0.0,
        end_time=1.0,
        steps=8,
    )
    reverse = integrate_modflow_rk4(
        forward,
        velocity,
        start_time=1.0,
        end_time=0.0,
        steps=8,
    )
    np.testing.assert_allclose(forward, points + (0.1, -0.2, 0.3), atol=1e-15)
    np.testing.assert_allclose(reverse, points, atol=1e-15)


def test_same_zero_embedding_transfer_is_identity_and_repeatable() -> None:
    parameters = np.zeros(FLOW_PARAMETER_COUNT, dtype=np.float64)
    points = np.linspace(0.0, 1.0, 81, dtype=np.float64).reshape(27, 3)
    first = transfer_modflows(
        points,
        content_parameters=parameters,
        style_parameters=parameters,
        steps_per_leg=8,
    )
    second = transfer_modflows(
        points,
        content_parameters=parameters,
        style_parameters=parameters,
        steps_per_leg=8,
    )
    assert np.array_equal(first, points)
    assert np.array_equal(second, first)


def test_invalid_parameter_and_integration_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        ModFlowVelocity.from_vector(np.zeros(FLOW_PARAMETER_COUNT - 1))
    velocity = ModFlowVelocity.from_vector(np.zeros(FLOW_PARAMETER_COUNT))
    with pytest.raises(ValueError):
        velocity(np.zeros((2, 4)), 0.0)
    with pytest.raises(ValueError):
        integrate_modflow_rk4(
            np.zeros((2, 3)),
            velocity,
            start_time=0.0,
            end_time=1.0,
            steps=0,
        )
