from __future__ import annotations

import numpy as np

from src.real_film.negicc_stock_k1 import _basis, _fit, _predict, _relative_improvement


def test_quadratic_basis_shape_and_finite_values() -> None:
    xyz = np.asarray([[1.0, 2.0, 3.0], [20.0, 30.0, 40.0]], dtype=np.float64)
    basis = _basis(xyz, 1.0)
    assert basis.shape == (2, 10)
    assert np.isfinite(basis).all()
    assert np.array_equal(basis[:, 0], np.ones(2))


def test_fit_predict_recovers_exact_quadratic_mapping() -> None:
    rng = np.random.default_rng(20260827)
    xyz = rng.uniform(1.0, 80.0, size=(100, 3)).astype(np.float64)
    features = _basis(xyz, 0.0)
    coefficients = np.asarray(
        [[-4.0, -4.2, -4.4], [0.1, 0.2, 0.3]] + [[0.0, 0.0, 0.0]] * 8,
        dtype=np.float64,
    )
    target = features @ coefficients
    fit = _fit(features, target, 1e-12)
    predicted = _predict(fit, xyz, 0.0)
    expected = np.exp2(target) * 65535.0
    assert np.allclose(predicted, expected, rtol=1e-7, atol=1e-7)


def test_relative_improvement_is_positive_when_candidate_is_better() -> None:
    candidate = np.asarray([1.0, 2.0])
    control = np.asarray([2.0, 4.0])
    assert np.array_equal(_relative_improvement(candidate, control), [0.5, 0.5])
