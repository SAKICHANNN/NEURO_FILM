from __future__ import annotations

import json

import numpy as np

from src.roll2film.factorized_code_domain_residual import (
    FactorizedCodeDomainResidualOperator,
    fit_factorized_code_domain_residual,
)
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurvePositiveMatrixOperator,
)


def _identity_base() -> MonotoneCurvePositiveMatrixOperator:
    return MonotoneCurvePositiveMatrixOperator(
        curve_segment_weights=np.full((3, 3), 1.0 / 3.0),
        matrix=np.eye(3),
    )


def test_factorized_residual_fits_and_replays() -> None:
    source = np.random.default_rng(11).uniform(size=(256, 3))
    base = _identity_base()
    target = np.column_stack(
        (
            source[:, 0] ** 0.9,
            source[:, 1],
            source[:, 2] ** 1.1,
        )
    )
    operator = fit_factorized_code_domain_residual(
        base,
        source,
        target,
        axis_size=2,
        sigma=0.32,
        epsilon=1e-12,
        ridge=1e-3,
    )
    assert np.sqrt(np.mean((operator.apply(source) - target) ** 2)) < 0.02
    replay = FactorizedCodeDomainResidualOperator.from_dict(
        json.loads(json.dumps(operator.to_dict()))
    )
    assert np.array_equal(replay.apply(source), operator.apply(source))
    assert np.array_equal(
        operator.apply(source, strength=0.0), base.apply(source)
    )


def test_factorized_residual_preserves_cube() -> None:
    source = np.random.default_rng(12).uniform(size=(128, 3))
    operator = fit_factorized_code_domain_residual(
        _identity_base(),
        source,
        1.0 - source,
        axis_size=2,
        sigma=0.32,
        epsilon=1e-12,
        ridge=1e-3,
    )
    output = operator.apply(source)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
