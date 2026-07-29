from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.colorreference_bounded_bernstein import (
    BoundedBernsteinModel,
    apply_bounded_bernstein,
    bernstein_tensor_basis,
    fit_bounded_bernstein,
)


def test_basis_is_nonnegative_partition_of_unity() -> None:
    rng = np.random.default_rng(7)
    source = rng.random((128, 3))
    for degree in (1, 2, 3):
        basis = bernstein_tensor_basis(source, degree=degree)
        assert basis.shape == (128, (degree + 1) ** 3)
        assert np.all(basis >= 0.0)
        np.testing.assert_allclose(
            basis.sum(axis=1), 1.0, rtol=0.0, atol=2e-15
        )


def test_bounded_fit_and_apply_need_no_clipping() -> None:
    rng = np.random.default_rng(9)
    source = rng.random((256, 3))
    target = np.column_stack(
        (
            0.1 + 0.5 * source[:, 0],
            0.2 + 0.4 * source[:, 1],
            0.3 + 0.3 * source[:, 2],
        )
    )
    model = fit_bounded_bernstein(
        source,
        target,
        degree=2,
        lower_bound=0.0,
        upper_bound=1.2,
        tolerance=1e-12,
        maximum_iterations=2000,
    )
    output = apply_bounded_bernstein(model, source)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.2
    np.testing.assert_allclose(output, target, atol=1e-10, rtol=0.0)


def test_invalid_model_and_source_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid"):
        BoundedBernsteinModel(
            degree=2,
            coefficients=np.full((27, 3), 1.3),
            lower_bound=0.0,
            upper_bound=1.2,
        )
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        bernstein_tensor_basis(
            np.asarray([[0.0, 0.5, 1.1]]), degree=2
        )
