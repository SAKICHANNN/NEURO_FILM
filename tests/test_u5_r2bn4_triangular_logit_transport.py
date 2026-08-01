from __future__ import annotations

import numpy as np

from src.roll2film.triangular_logit_transport import (
    TriangularLogitTransport,
    fit_triangular_logit_transport,
    select_safe_transport,
    transport_diagnostics,
)


def test_identity_and_endpoints_are_exact() -> None:
    source = np.array(
        [[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]], dtype=np.float64
    )
    transport = TriangularLogitTransport(np.zeros(14))
    assert np.array_equal(transport.apply(source), source)
    assert np.array_equal(transport.inverse(source), source)


def test_nontrivial_transport_is_cube_safe_and_invertible() -> None:
    parameters = np.array(
        [0.2, -0.1, 0.1, 0.15, 0.2, -0.1, -0.1, 0.1, 0.2, -0.1, 0.05, 0.1, -0.15, 0.2]
    )
    rng = np.random.default_rng(421)
    source = rng.uniform(0.001, 0.999, size=(41, 37, 3))
    transport = TriangularLogitTransport(parameters)
    output = transport.apply(source)
    restored = transport.inverse(output)
    assert np.min(output) > 0.0
    assert np.max(output) < 1.0
    assert np.max(np.abs(restored - source)) < 1.0e-12


def test_diagnostics_prove_positive_triangular_jacobian() -> None:
    transport = TriangularLogitTransport(
        np.array([0.3, 0.1, 0.2, -0.2, 0.1, 0.2, -0.2, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1, 0.1])
    )
    diagnostics = transport_diagnostics(
        transport, grid_size=7, finite_difference=1.0e-5
    )
    assert diagnostics["minimum_determinant"] > 0.0
    assert diagnostics["nonpositive_determinant_count"] == 0
    assert diagnostics["maximum_inverse_roundtrip_error"] < 1.0e-10


def test_sequential_fit_recovers_synthetic_transport() -> None:
    rng = np.random.default_rng(422)
    source = rng.uniform(0.02, 0.98, size=(96, 80, 3))
    truth = np.array(
        [0.15, -0.08, 0.12, 0.08, 0.18, -0.10, -0.08, 0.06, 0.10, -0.04, 0.12, 0.08, -0.06, 0.05]
    )
    target = TriangularLogitTransport(truth).apply(source)
    fitted, success = fit_triangular_logit_transport(
        source,
        target,
        lower_bounds=np.full(14, -1.0),
        upper_bounds=np.full(14, 1.0),
        sample_stride=2,
        identity_shrinkage=1.0e-6,
        maximum_evaluations=100,
    )
    assert success
    predicted = TriangularLogitTransport(fitted).apply(source)
    assert np.sqrt(np.mean((predicted - target) ** 2)) < 1.0e-5


def test_safety_selection_shrinks_extreme_conditioning() -> None:
    parameters = np.array(
        [0.7, 1.0, 0.7, 0.7, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0]
    )
    transport, diagnostics = select_safe_transport(
        parameters=parameters,
        grid_size=7,
        finite_difference=1.0e-5,
        minimum_jacobian_determinant=0.02,
        maximum_jacobian_condition=30.0,
        bisection_iterations=24,
    )
    assert 0.0 <= transport.dose <= 1.0
    assert diagnostics["minimum_determinant"] >= 0.02
    assert diagnostics["maximum_condition"] <= 30.0
