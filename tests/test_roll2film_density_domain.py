from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.density_domain import (
    DENSITY_OPERATOR_SCHEMA,
    DensityDomainNegativePrintOperator,
    finite_difference_jacobians,
    logistic_density,
    operator_from_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2e0_density_domain_operator_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _operator(name: str = "neutral_density_reference") -> DensityDomainNegativePrintOperator:
    return operator_from_config(_config()["witnesses"][name])


def test_logistic_density_matches_independent_scalar_reference() -> None:
    exposure = np.array([[-4.0, -3.0, -2.0]])
    midpoint = np.array([-3.2, -3.0, -2.8])
    slope = np.array([0.8, 1.1, 1.4])
    maximum = np.array([1.7, 2.0, 2.3])
    expected = np.array(
        [
            maximum[channel]
            / (
                1.0
                + np.exp(
                    -slope[channel]
                    * (exposure[0, channel] - midpoint[channel])
                )
            )
            for channel in range(3)
        ]
    )
    assert np.allclose(
        logistic_density(exposure, midpoint, slope, maximum)[0],
        expected,
        atol=1e-15,
    )


def test_endpoints_strength_source_nonmutation_and_partition_parity() -> None:
    operator = _operator("warm_dense_like")
    endpoints = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.max(np.abs(operator.apply(endpoints) - endpoints)) <= 1e-12

    rng = np.random.default_rng(83)
    source = rng.random((101, 3))
    before = source.copy()
    full = operator.apply(source)
    partitioned = np.concatenate(
        [operator.apply(source[:37]), operator.apply(source[37:79]), operator.apply(source[79:])]
    )
    assert np.array_equal(full, partitioned)
    assert np.array_equal(operator.apply(source, strength=0.0), source)
    assert np.array_equal(operator.apply(source, strength=1.0), full)
    assert np.array_equal(source, before)


def test_serialization_replay_and_immutable_parameters() -> None:
    operator = _operator("cross_processed_like")
    payload = json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    replay = DensityDomainNegativePrintOperator.from_dict(payload)
    source = np.random.default_rng(84).random((31, 47, 3))

    assert payload["schema"] == DENSITY_OPERATOR_SCHEMA
    assert np.array_equal(operator.apply(source), replay.apply(source))
    with pytest.raises(ValueError):
        operator.negative_midpoints[0] = -5.0


def test_all_witnesses_bounded_monotone_and_positive_jacobian() -> None:
    config = _config()
    axis = np.linspace(0.0, 1.0, 17)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    interior = np.stack(
        np.meshgrid(axis[1:-1], axis[1:-1], axis[1:-1], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    for payload in config["witnesses"].values():
        operator = operator_from_config(payload)
        output = operator.apply(grid)
        jacobians = finite_difference_jacobians(
            operator,
            interior,
            step=config["finite_difference_step"],
        )
        assert np.min(output) >= 0.0
        assert np.max(output) <= 1.0
        assert np.min(jacobians) >= 0.0
        assert np.min(np.linalg.det(jacobians)) > 0.0


def test_invalid_parameters_and_inputs_fail_closed() -> None:
    payload = _config()["witnesses"]["neutral_density_reference"]
    invalid_matrix = json.loads(json.dumps(payload))
    invalid_matrix["capture_matrix"][0] = [0.9, 0.2, 0.1]
    with pytest.raises(ValueError, match="row-stochastic"):
        operator_from_config(invalid_matrix)

    invalid_slope = json.loads(json.dumps(payload))
    invalid_slope["paper_slopes"][0] = 0.0
    with pytest.raises(ValueError, match="slopes"):
        operator_from_config(invalid_slope)

    operator = operator_from_config(payload)
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        operator.apply(np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        operator.apply(np.array([[0.1, 0.2, 0.3]]), strength=1.1)
