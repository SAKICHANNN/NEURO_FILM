from __future__ import annotations

import numpy as np

from src.roll2film.parallel_projection_curves import (
    ParallelProjectionCurveOperator,
    fit_projection_curve_coefficients,
    jacobian_diagnostics,
    select_safe_dose,
)


DIRECTIONS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ],
    dtype=np.float64,
)
EPSILON = 1.0 / 510.0


def test_identity_roundtrip_and_boundary_class_are_exact() -> None:
    rng = np.random.default_rng(2026080101)
    source = rng.random((67, 71, 3), dtype=np.float64)
    source[0, 0] = [0.0, EPSILON / 2.0, EPSILON]
    source[0, 1] = [1.0, 1.0 - EPSILON / 2.0, 1.0 - EPSILON]
    identity = ParallelProjectionCurveOperator.identity(
        DIRECTIONS,
        control_point_count=9,
        boundary_epsilon=EPSILON,
    )
    assert np.array_equal(identity.apply(source), source)
    assert np.array_equal(
        ParallelProjectionCurveOperator.from_dict(identity.to_dict()).apply(
            source
        ),
        source,
    )

    extreme = ParallelProjectionCurveOperator(
        DIRECTIONS,
        np.full((len(DIRECTIONS), 9, 3), 50.0),
        EPSILON,
    )
    output = extreme.apply(source)
    source_boundary = (source <= EPSILON) | (source >= 1.0 - EPSILON)
    output_boundary = (output <= EPSILON) | (
        output >= 1.0 - EPSILON
    )
    assert not np.any(output_boundary & ~source_boundary)
    assert np.array_equal(output[source_boundary], source[source_boundary])


def test_fit_recovers_a_known_projection_curve_transform() -> None:
    rng = np.random.default_rng(2026080102)
    source = rng.uniform(0.03, 0.97, size=(96, 80, 3))
    control_axis = np.linspace(-0.35, 0.35, 9)
    coefficients = np.zeros((len(DIRECTIONS), 9, 3), dtype=np.float64)
    coefficients[0, :, 0] = control_axis
    coefficients[1, :, 1] = -0.6 * control_axis
    coefficients[2, :, 2] = 0.4 * np.sin(np.linspace(0.0, np.pi, 9))
    target = ParallelProjectionCurveOperator(
        DIRECTIONS, coefficients, EPSILON, 0.8
    ).apply(source)
    fitted = fit_projection_curve_coefficients(
        source,
        target,
        directions=DIRECTIONS,
        control_point_count=9,
        boundary_epsilon=EPSILON,
        sample_stride=2,
        target_activation_limit=0.95,
        identity_shrinkage=0.001,
        first_difference_smoothness=0.001,
        coefficient_absolute_limit=3.0,
    )
    rendered = ParallelProjectionCurveOperator(
        DIRECTIONS, fitted, EPSILON
    ).apply(source)
    before = float(np.sqrt(np.mean((source - target) ** 2)))
    after = float(np.sqrt(np.mean((rendered - target) ** 2)))
    assert after < before * 0.08


def test_safe_dose_rejects_folded_full_strength_without_pixel_clipping() -> None:
    coefficients = np.zeros((len(DIRECTIONS), 9, 3), dtype=np.float64)
    coefficients[0, :, 0] = np.asarray(
        [3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0]
    )
    full = ParallelProjectionCurveOperator(
        DIRECTIONS, coefficients, EPSILON
    )
    full_diagnostics = jacobian_diagnostics(
        full, grid_size=7, finite_difference=1e-5
    )
    assert (
        full_diagnostics["nonpositive_determinant_count"] > 0
        or full_diagnostics["minimum_determinant"] < 0.02
        or full_diagnostics["maximum_condition"] > 30.0
    )
    safe, diagnostics = select_safe_dose(
        directions=DIRECTIONS,
        coefficients=coefficients,
        boundary_epsilon=EPSILON,
        grid_size=7,
        finite_difference=1e-5,
        minimum_jacobian_determinant=0.02,
        maximum_jacobian_condition=30.0,
        bisection_iterations=24,
    )
    assert 0.0 < safe.dose < 1.0
    assert diagnostics["nonpositive_determinant_count"] == 0
    assert diagnostics["minimum_determinant"] >= 0.02
    assert diagnostics["maximum_condition"] <= 30.0
    cube = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, 17),
            np.linspace(0.0, 1.0, 17),
            np.linspace(0.0, 1.0, 17),
            indexing="ij",
        ),
        axis=-1,
    )
    output = safe.apply(cube)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
