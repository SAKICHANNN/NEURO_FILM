from __future__ import annotations

import numpy as np

from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    dose_operator,
    fit_operator,
    fit_operator_rows,
    matrix_diagnostics,
    mean_oklab_error,
    new_exact_boundary_fraction,
    sample_indexes,
)


def test_sample_indexes_are_exact_and_unique() -> None:
    first = sample_indexes("I0001", 5000, 4096)
    second = sample_indexes("I0001", 5000, 4096)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4096


def test_fit_recovers_bounded_logit_affine() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform(0.02, 0.98, size=(65, 65, 3)).astype(np.float32)
    truth = LogitAffineOperator(
        matrix=np.asarray(((1.05, 0.02, 0.0), (0.0, 0.97, 0.01), (0.01, 0.0, 1.02))),
        bias=np.asarray((0.04, -0.02, 0.03)),
        dose=1.0,
    )
    target = apply_operator(source, truth)
    matrix, bias = fit_operator(
        [source], [target], ["scene"], pixels_per_scene=4096, ridge_alpha=1.0e-4
    )
    fitted = LogitAffineOperator(matrix=matrix, bias=bias, dose=1.0)
    assert mean_oklab_error(apply_operator(source, fitted), target) < 1.0e-6
    assert new_exact_boundary_fraction(source, apply_operator(source, fitted)) == 0.0


def test_dose_and_matrix_diagnostics() -> None:
    operator = dose_operator(np.eye(3) * 1.2, np.ones(3) * 0.1, 0.5)
    diagnostics = matrix_diagnostics(operator)
    assert np.allclose(operator.matrix, np.eye(3) * 1.1)
    assert np.allclose(operator.bias, np.ones(3) * 0.05)
    assert diagnostics["determinant"] > 0.05


def test_fit_rows_matches_image_fit() -> None:
    rng = np.random.default_rng(11)
    source = rng.uniform(0.05, 0.95, size=(70, 70, 3)).astype(np.float32)
    target = np.clip(source * 0.92 + 0.03, 0.0, 1.0).astype(np.float32)
    indexes = sample_indexes("same", source.shape[0] * source.shape[1], 4096)
    expected = fit_operator(
        [source], [target], ["same"], pixels_per_scene=4096, ridge_alpha=1.0e-4
    )
    actual = fit_operator_rows(
        source.reshape(-1, 3)[indexes],
        target.reshape(-1, 3)[indexes],
        ridge_alpha=1.0e-4,
    )
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
