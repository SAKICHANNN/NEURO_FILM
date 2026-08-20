from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.canoncgt_extended_triangular_transport import (
    ExtendedTriangularTransport,
    fit_extended_triangular_transport,
    jacobian_diagnostics,
)
from src.roll2film.triangular_logit_transport import TriangularLogitTransport

LOWER = np.asarray(
    [-0.7, -1.0, -0.7, -0.7, -1.0, -1.0, -0.7, -0.7, -0.7, -0.7, -1.0, -1.0, -1.0, -1.0]
)
UPPER = -LOWER


def test_extended_coordinate_and_inverse_are_exact_enough() -> None:
    operator = ExtendedTriangularTransport(
        TriangularLogitTransport(np.zeros(14)), 2.0, -0.1, 1.1
    )
    values = np.asarray(((-0.1, 0.2, 1.1), (0.0, 0.5, 1.0)))
    restored = operator.inverse(operator.apply(values))
    assert np.max(np.abs(restored - values)) < 1.0e-14
    assert np.array_equal(
        operator.coordinate(values)[0],
        np.asarray((0.0, operator.coordinate(values)[0, 1], 1.0)),
    )


def test_fit_recovers_synthetic_extended_transport() -> None:
    rng = np.random.default_rng(111)
    source = rng.uniform(-0.08, 1.08, size=(96, 96, 3)).astype(np.float32)
    parameters = np.asarray(
        (
            0.08,
            -0.12,
            0.05,
            0.02,
            0.1,
            -0.04,
            0.04,
            0.01,
            -0.02,
            0.01,
            -0.08,
            0.03,
            0.02,
            -0.01,
        )
    )
    truth = ExtendedTriangularTransport(
        TriangularLogitTransport(parameters), 2.0, -0.1, 1.1
    )
    target = truth.apply(source).reshape(source.shape)
    fitted, metrics, _, _, _ = fit_extended_triangular_transport(
        source,
        target,
        maximum_rows=8000,
        asinh_scale=2.0,
        input_minimum=-0.1,
        input_maximum=1.1,
        lower_bounds=LOWER,
        upper_bounds=UPPER,
        identity_shrinkage=0.0,
        maximum_evaluations=300,
    )
    assert metrics["converged"] is True
    assert metrics["holdout"]["p95"] < 1.0e-5
    assert np.max(np.abs(fitted.transport.parameters)) < 1.0


def test_extended_transport_has_positive_jacobian_and_inverse() -> None:
    operator = ExtendedTriangularTransport(
        TriangularLogitTransport(np.zeros(14)), 2.0, -0.1, 1.1
    )
    diagnostics = jacobian_diagnostics(operator, grid_size=5, finite_difference=1.0e-6)
    assert diagnostics["minimum_determinant"] > 0.0
    assert diagnostics["minimum_singular_value"] > 0.0
    assert diagnostics["maximum_inverse_roundtrip_error"] < 1.0e-12


def test_runner_has_no_drive_literal_and_requires_output() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2repid11_canoncgt_extended_triangular_transport.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source
