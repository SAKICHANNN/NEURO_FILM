from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.positive_film import (
    POSITIVE_FILM_RESPONSE_SCHEMA,
    PositiveFilmResponseOperator,
    finite_difference_jacobians,
    positive_film_operator_from_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2j0_positive_film_response_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _operator(
    name: str = "neutral_positive_reference",
) -> PositiveFilmResponseOperator:
    return positive_film_operator_from_config(_config()["witnesses"][name])


def test_scalar_reference_matches_vector_operator() -> None:
    operator = _operator("warm_highlight_like")
    source = np.asarray([[0.17, 0.43, 0.79]])
    layer = operator.capture_matrix @ source[0]
    log_exposure = np.log2(layer + operator.exposure_floor)
    response = np.asarray(
        [
            operator.maximum_responses[index]
            / (
                1.0
                + np.exp(
                    -operator.response_slopes[index]
                    * (
                        log_exposure[index]
                        - operator.response_midpoints[index]
                    )
                )
            )
            for index in range(3)
        ]
    )
    raw = operator.scan_matrix @ response
    black, white = operator._raw_endpoints()
    expected = (raw - black) / (white - black)
    assert np.allclose(operator.apply(source)[0], expected, atol=1e-15)


def test_endpoints_strength_partition_and_source_nonmutation() -> None:
    operator = _operator("cyan_shadow_warm_highlight_like")
    endpoints = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.max(np.abs(operator.apply(endpoints) - endpoints)) <= 1e-12
    source = np.random.default_rng(119).random((107, 3))
    before = source.copy()
    full = operator.apply(source)
    partitioned = np.concatenate(
        (
            operator.apply(source[:29]),
            operator.apply(source[29:71]),
            operator.apply(source[71:]),
        )
    )
    assert np.array_equal(full, partitioned)
    assert np.array_equal(operator.apply(source, strength=0.0), source)
    assert np.array_equal(operator.apply(source, strength=1.0), full)
    assert np.array_equal(source, before)


def test_serialization_and_readonly_parameters() -> None:
    operator = _operator("cross_bias_like")
    payload = json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    replay = PositiveFilmResponseOperator.from_dict(payload)
    source = np.random.default_rng(120).random((23, 31, 3))
    assert payload["schema"] == POSITIVE_FILM_RESPONSE_SCHEMA
    assert np.array_equal(operator.apply(source), replay.apply(source))
    with pytest.raises(ValueError):
        operator.response_midpoints[0] = -4.0


def test_all_witnesses_are_bounded_monotone_and_orientation_preserving() -> None:
    config = _config()
    axis = np.linspace(0.0, 1.0, config["audit_grid_size"])
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    interior = np.stack(
        np.meshgrid(axis[1:-1], axis[1:-1], axis[1:-1], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    for payload in config["witnesses"].values():
        operator = positive_film_operator_from_config(payload)
        output = operator.apply(grid)
        jacobians = finite_difference_jacobians(
            operator, interior, step=config["finite_difference_step"]
        )
        assert np.min(output) >= 0.0
        assert np.max(output) <= 1.0
        assert np.min(jacobians) >= 0.0
        assert np.min(np.linalg.det(jacobians)) > 0.0


def test_invalid_parameters_and_inputs_fail_closed() -> None:
    payload = _config()["witnesses"]["neutral_positive_reference"]
    invalid_matrix = json.loads(json.dumps(payload))
    invalid_matrix["scan_matrix"][0] = [0.9, 0.2, 0.1]
    with pytest.raises(ValueError, match="row-stochastic"):
        positive_film_operator_from_config(invalid_matrix)
    invalid_slope = json.loads(json.dumps(payload))
    invalid_slope["response_slopes"][0] = 0.0
    with pytest.raises(ValueError, match="slopes"):
        positive_film_operator_from_config(invalid_slope)
    operator = positive_film_operator_from_config(payload)
    with pytest.raises(ValueError):
        operator.apply(np.asarray([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        operator.apply(np.asarray([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        operator.apply(np.asarray([[0.1, 0.2, 0.3]]), strength=1.1)
