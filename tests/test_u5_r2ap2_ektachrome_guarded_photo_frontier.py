from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ap2_ektachrome_guarded_photo_frontier import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import new_hard_clipping_fraction
from src.eval.positive_film_frontier import candidate_bank, validate_contract
from src.roll2film.factorized_boundary_guard import (
    apply_residual_boundary_guard,
)
from src.roll2film.positive_film import positive_film_operator_from_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ap2_ektachrome_guarded_photo_frontier_v1.json"
)


def _operator_and_guard():
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    payload = json.loads((ROOT / config["operator_config"]).read_bytes())
    witness = payload["witnesses"][config["witness_ids"][0]]
    operator = positive_film_operator_from_config(
        witness,
        exposure_floor=float(payload["exposure_floor"]),
        matrix_minimum_determinant=float(
            payload["parameter_bounds"]["matrix_minimum_determinant"]
        ),
        minimum_endpoint_span=float(
            payload["parameter_bounds"]["minimum_endpoint_span"]
        ),
    )
    return operator, config["rendering"]["residual_guard"]


def test_ap2_contract_lineage_and_candidates() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    validated = validate_contract(ROOT, config)
    assert validated["execution_policy"] == (
        "source_inclusive_residual_guard_v1"
    )
    assert len(validated["samples"]) == 41
    assert [row["strength"] for row in candidate_bank(config)] == [0.75, 1.0]
    assert config["rendering"]["hard_clipping_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_residual_guard_is_exact_and_creates_no_new_hard_boundary() -> None:
    operator, guard = _operator_and_guard()
    encoded = np.random.default_rng(20260730).random((96, 64, 3))
    linear = encoded_srgb_to_linear(encoded)
    arguments = {
        "strength": 1.0,
        "hard_boundary_epsilon_encoded_srgb": guard[
            "hard_boundary_epsilon_encoded_srgb"
        ],
        "guard_boundary_epsilon_encoded_srgb": guard[
            "guard_boundary_epsilon_encoded_srgb"
        ],
    }
    first = apply_residual_boundary_guard(operator, linear, **arguments)
    second = apply_residual_boundary_guard(operator, linear, **arguments)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.residual_scale, second.residual_scale)
    output = linear_srgb_to_encoded(first.output)
    assert (
        new_hard_clipping_fraction(
            encoded,
            output,
            guard["hard_boundary_epsilon_encoded_srgb"],
        )
        == 0.0
    )
    assert np.any(first.residual_scale < 1.0)


def test_residual_guard_zero_strength_is_identity() -> None:
    operator, guard = _operator_and_guard()
    source = np.random.default_rng(11).random((32, 3))
    result = apply_residual_boundary_guard(
        operator,
        source,
        strength=0.0,
        hard_boundary_epsilon_encoded_srgb=guard[
            "hard_boundary_epsilon_encoded_srgb"
        ],
        guard_boundary_epsilon_encoded_srgb=guard[
            "guard_boundary_epsilon_encoded_srgb"
        ],
    )
    np.testing.assert_array_equal(result.output, source)
    np.testing.assert_array_equal(result.residual_scale, np.ones(32))


def test_ap2_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)
