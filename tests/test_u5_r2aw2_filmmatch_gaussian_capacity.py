from __future__ import annotations

import json

import numpy as np
import pytest

from src.eval.filmmatch_gaussian_residual_capacity import structural_strength
from src.roll2film.code_domain_gaussian_residual import (
    CodeDomainGaussianResidualOperator,
    fit_code_domain_gaussian_residual,
)


def test_code_domain_operator_is_bounded_and_replayable() -> None:
    source = np.random.default_rng(4).uniform(size=(128, 3))
    target = np.column_stack(
        (source[:, 0] ** 0.8, source[:, 1], source[:, 2] ** 1.2)
    )
    operator = fit_code_domain_gaussian_residual(
        source,
        target,
        axis_size=2,
        sigma=0.32,
        epsilon=1e-12,
        ridge=1e-3,
    )
    points = np.random.default_rng(5).uniform(size=(257, 3))
    output = operator.apply(points)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    replay = CodeDomainGaussianResidualOperator.from_dict(
        json.loads(json.dumps(operator.to_dict()))
    )
    assert np.array_equal(replay.apply(points), output)


def test_structural_strength_falls_back_to_identity() -> None:
    centers = np.array(
        [[r, g, b] for r in (0.0, 1.0) for g in (0.0, 1.0) for b in (0.0, 1.0)]
    )
    coefficients = np.full((36, 3), 1e6)
    operator = CodeDomainGaussianResidualOperator(
        centers=centers,
        sigma=0.32,
        coefficients=coefficients,
    )
    config = {
        "candidate": {
            "structural_audit": {
                "cube_size": 3,
                "jacobian_size": 3,
                "jacobian_margin": 0.01,
                "jacobian_step": 1e-5,
                "strength_schedule": [1.0, 0.0],
                "minimum_jacobian_determinant": 1e-8,
                "maximum_jacobian_spectral_norm": 8.0,
            }
        }
    }
    strength, audit = structural_strength(operator, config)
    assert strength == 0.0
    assert audit["candidates"][-1]["safe"]


def test_invalid_code_domain_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_code_domain_gaussian_residual(
            np.zeros((3, 3)),
            np.zeros((2, 3)),
            axis_size=2,
            sigma=0.3,
            epsilon=1e-12,
            ridge=1e-3,
        )
