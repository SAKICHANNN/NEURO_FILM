from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_u5_r2ak1_time_dependent_cube_flow_capacity import (
    _fit_stationary,
    _source_fields,
    _target,
)
from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
    _sample_grid_numpy,
    _sample_grid_torch,
    finite_difference_jacobians,
    fit_cube_diffeomorphic_colour_flow,
)
from src.roll2film.time_dependent_cube_flow import (
    TimeDependentCubeColourFlow,
    _bernstein_weights_numpy,
    _sample_grid_torch_vectorized,
    _velocity_numpy,
    fit_time_dependent_cube_colour_flow,
)


def _rgb_grid(axis_size: int, *, interior: bool = False) -> np.ndarray:
    if interior:
        axis = np.linspace(0.1, 0.9, axis_size, dtype=np.float64)
    else:
        axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _test_operator() -> TimeDependentCubeColourFlow:
    axis = np.linspace(0.0, 1.0, 3, dtype=np.float64)
    rgb = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    )
    red, green, blue = np.moveaxis(rgb, -1, 0)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    controls = np.zeros((3, 3, 3, 3, 3), dtype=np.float64)
    controls[0, ..., 0] = 0.8 * (0.5 - luma)
    controls[0, ..., 1] = 0.3 * (red - blue)
    controls[1, ..., 0] = 0.4 * (green - blue)
    controls[1, ..., 2] = 0.5 * (red - green)
    controls[2, ..., 1] = -0.6 * (red - blue)
    controls[2, ..., 2] = 0.7 * (0.5 - luma)
    return TimeDependentCubeColourFlow(controls, integration_steps=32)


def test_identity_is_exact_and_shape_stable() -> None:
    operator = TimeDependentCubeColourFlow.identity(
        axis_size=3, integration_steps=8
    )
    rgb = _rgb_grid(7)
    assert np.array_equal(operator.apply(rgb), rgb)
    assert np.array_equal(operator.inverse(rgb), rgb)


def test_replay_partition_inverse_and_orientation() -> None:
    operator = _test_operator()
    rgb = _rgb_grid(7)
    output = operator.apply(rgb)
    partitioned = np.concatenate(
        [
            operator.apply(rgb.reshape(-1, 3)[:171]),
            operator.apply(rgb.reshape(-1, 3)[171:]),
        ]
    ).reshape(rgb.shape)
    replay = TimeDependentCubeColourFlow.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(output, partitioned)
    assert np.array_equal(output, replay.apply(rgb))
    assert np.max(np.abs(replay.inverse(output) - rgb)) < 1e-10
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0

    points = _rgb_grid(5, interior=True).reshape(-1, 3)
    jacobians = finite_difference_jacobians(
        operator, points, step=1e-6
    )
    determinants = np.linalg.det(jacobians)
    spectral_norms = np.linalg.svd(
        jacobians, compute_uv=False
    )[:, 0]
    assert float(determinants.min()) > 0.5
    assert float(spectral_norms.max()) < 2.0


def test_all_cube_boundary_faces_are_invariant() -> None:
    operator = _test_operator()
    rgb = _rgb_grid(9).reshape(-1, 3)
    output = operator.apply(rgb)
    for channel in range(3):
        for boundary in (0.0, 1.0):
            mask = rgb[:, channel] == boundary
            assert np.array_equal(
                output[mask, channel], rgb[mask, channel]
            )


def test_blend_before_sampling_matches_explicit_temporal_sum() -> None:
    operator = _test_operator()
    rgb = _rgb_grid(5, interior=True).reshape(-1, 3)
    time = 0.37
    weights = _bernstein_weights_numpy(time)
    explicit = np.zeros_like(rgb)
    for index in range(3):
        explicit += weights[index] * _sample_grid_numpy(
            rgb, operator.control_grids[index]
        )
    explicit *= rgb * (1.0 - rgb)
    assert np.max(
        np.abs(
            _velocity_numpy(
                rgb, time, operator.control_grids
            )
            - explicit
        )
    ) < 1e-15


def test_vectorized_torch_sampler_matches_reference_and_gradients() -> None:
    generator = torch.Generator().manual_seed(280729)
    rgb = torch.rand(
        (37, 3),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    grid = torch.rand(
        (4, 4, 4, 3),
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    reference = _sample_grid_torch(rgb, grid)
    vectorized = _sample_grid_torch_vectorized(rgb, grid)
    assert torch.max(torch.abs(reference - vectorized)).item() < 1e-15
    reference.sum().backward(retain_graph=True)
    reference_rgb_gradient = rgb.grad.detach().clone()
    reference_grid_gradient = grid.grad.detach().clone()
    rgb.grad.zero_()
    grid.grad.zero_()
    vectorized.sum().backward()
    assert torch.max(
        torch.abs(rgb.grad - reference_rgb_gradient)
    ).item() < 1e-14
    assert torch.max(
        torch.abs(grid.grad - reference_grid_gradient)
    ).item() < 1e-14


def test_vectorized_stationary_control_matches_reference_fitter() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/u5_r2ak1_time_dependent_cube_flow_capacity_v1.json"
        ).read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["fit"]["steps"] = 8
    config["stationary_controls"]["lower_parameter_control"][
        "integration_steps"
    ] = 6
    source = _rgb_grid(3, interior=True).reshape(-1, 3)
    truth_grid = np.zeros((2, 2, 2, 3), dtype=np.float64)
    truth_grid[..., 0] = 0.3
    truth_grid[..., 1] = -0.2
    truth_grid[..., 2] = 0.1
    target = CubeDiffeomorphicColourFlow(
        truth_grid, integration_steps=6
    ).apply(source)
    vectorized = _fit_stationary(
        source,
        target,
        config,
        control_name="lower_parameter_control",
    )
    fit = config["fit"]
    reference = fit_cube_diffeomorphic_colour_flow(
        source,
        target,
        axis_size=4,
        integration_steps=6,
        maximum_absolute_coefficient=6.0,
        seed=int(fit["seed"]),
        steps=int(fit["steps"]),
        learning_rate=float(fit["learning_rate"]),
        coefficient_l2=float(fit["coefficient_l2"]),
        velocity_smoothness_l2=float(
            fit["spatial_smoothness_l2"]
        ),
        gradient_clip_norm=float(fit["gradient_clip_norm"]),
        thread_count=int(fit["thread_count"]),
    )
    assert np.max(
        np.abs(vectorized.velocity_grid - reference.velocity_grid)
    ) < 1e-14


def test_deterministic_fit_recovers_time_varying_teacher() -> None:
    source = _rgb_grid(5)
    teacher = _test_operator()
    target = teacher.apply(source)
    arguments = dict(
        axis_size=3,
        integration_steps=12,
        maximum_absolute_coefficient=3.0,
        seed=280728,
        steps=100,
        learning_rate=0.04,
        coefficient_l2=1e-5,
        spatial_smoothness_l2=1e-4,
        temporal_smoothness_l2=5e-5,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    first = fit_time_dependent_cube_colour_flow(
        source, target, **arguments
    )
    second = fit_time_dependent_cube_colour_flow(
        source, target, **arguments
    )
    assert np.array_equal(first.control_grids, second.control_grids)
    error = np.sqrt(
        np.mean((first.apply(source) - target) ** 2)
    )
    assert float(error) < 0.004


def test_validation_rejects_wrong_temporal_contract() -> None:
    with np.testing.assert_raises(ValueError):
        TimeDependentCubeColourFlow(
            np.zeros((2, 3, 3, 3, 3), dtype=np.float64)
        )
    payload = TimeDependentCubeColourFlow.identity(
        axis_size=3
    ).to_dict()
    payload["temporal_basis"] = "linear"
    with np.testing.assert_raises(ValueError):
        TimeDependentCubeColourFlow.from_dict(payload)


def test_frozen_staged_truth_is_genuinely_noncommuting() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/u5_r2ak1_time_dependent_cube_flow_capacity_v1.json"
        ).read_text(encoding="utf-8")
    )
    field_a, field_b = _source_fields(config)
    rgb = _rgb_grid(
        int(config["confirmation_grid_axis_size"]), interior=True
    ).reshape(-1, 3)
    a_then_b = _target(
        "field_a_then_field_b", rgb, field_a, field_b
    )
    b_then_a = _target(
        "field_b_then_field_a", rgb, field_a, field_b
    )
    rmse = float(np.sqrt(np.mean((a_then_b - b_then_a) ** 2)))
    assert (
        rmse
        >= config["synthetic_truth"]["minimum_required_target_pair_rmse"]
    )
