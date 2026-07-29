from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.recorder_proxy_code_reuse_frontier import (
    RecorderProxyReuseError,
    recorder_target_linear_srgb,
    validate_contract,
)
from src.roll2film.colorreference_bounded_bernstein import (
    BoundedBernsteinModel,
)
from src.roll2film.factorized_boundary_guard import (
    apply_target_residual_boundary_guard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq4c_recorder_proxy_code_reuse_frontier_v1.json"
)


def test_contract_is_unidentified_look_approximation_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert config["operator_contract"]["input_semantics_identified"] is False
    assert config["operator_refit_allowed"] is False
    assert config["production_integration_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_explicit_target_guard_is_bounded_and_deterministic() -> None:
    source = np.asarray(
        [[[0.0, 0.25, 1.0], [0.4, 0.5, 0.6]]], dtype=np.float64
    )
    target = np.asarray(
        [[[-2.0, 2.0, 3.0], [0.9, -0.5, 1.4]]],
        dtype=np.float64,
    )
    arguments = {
        "strength": 1.0,
        "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
        "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
    }
    first = apply_target_residual_boundary_guard(
        source, target, **arguments
    )
    second = apply_target_residual_boundary_guard(
        source, target, **arguments
    )
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(
        first.residual_scale, second.residual_scale
    )
    assert np.min(first.output) >= 0.0
    assert np.max(first.output) <= 1.0


def test_recorder_target_rejects_out_of_range_code_values() -> None:
    model = BoundedBernsteinModel(
        degree=1,
        coefficients=np.full((8, 3), 0.5),
        lower_bound=0.0,
        upper_bound=1.2,
    )
    with pytest.raises(RecorderProxyReuseError, match="encoded"):
        recorder_target_linear_srgb(
            model, np.asarray([[0.0, 0.5, 1.1]])
        )


def test_contract_tamper_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator_contract"]["input_semantics_identified"] = True
    with pytest.raises(RecorderProxyReuseError, match="contract"):
        validate_contract(ROOT, config)
