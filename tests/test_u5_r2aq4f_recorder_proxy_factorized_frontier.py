from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq4f_recorder_proxy_factorized_frontier import (
    CONFIG_SHA256,
)
from src.eval.recorder_proxy_factorized_frontier import (
    RecorderProxyFactorizedError,
    validate_contract,
)
from src.roll2film.factorized_boundary_guard import (
    apply_target_factorized_boundary_guard,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq4f_recorder_proxy_factorized_frontier_v1.json"
)


def test_frozen_contract_and_direct_control_are_exact() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    validated = validate_contract(ROOT, config)
    assert len(validated["samples"]) == 41
    assert len(validated["direct_records"]) == 41
    assert config["operator_refit_allowed"] is False
    assert config["production_integration_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_explicit_target_factorization_is_repeat_exact_and_bounded() -> None:
    source = np.asarray(
        [[[0.0, 0.25, 1.0], [0.4, 0.5, 0.6]]], dtype=np.float64
    )
    target = np.asarray(
        [[[-2.0, 2.0, 3.0], [0.9, -0.5, 1.4]]],
        dtype=np.float64,
    )
    arguments = {
        "tone_strength": 0.5,
        "chroma_strength": 2.0,
        "luma_weights": np.asarray([0.2126, 0.7152, 0.0722]),
        "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
        "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
    }
    first = apply_target_factorized_boundary_guard(
        source, target, **arguments
    )
    second = apply_target_factorized_boundary_guard(
        source, target, **arguments
    )
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.tone_scale, second.tone_scale)
    np.testing.assert_array_equal(first.chroma_scale, second.chroma_scale)
    assert np.min(first.output) >= 0.0
    assert np.max(first.output) <= 1.0


def test_target_shape_or_contract_tamper_fails_closed() -> None:
    source = np.full((2, 3), 0.5)
    with pytest.raises(ValueError, match="invalid"):
        apply_target_factorized_boundary_guard(
            source,
            np.full((1, 3), 0.5),
            tone_strength=0.5,
            chroma_strength=1.25,
            luma_weights=np.asarray([0.2126, 0.7152, 0.0722]),
            hard_boundary_epsilon_encoded_srgb=0.5 / 255.0,
            guard_boundary_epsilon_encoded_srgb=1.0 / 255.0,
        )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["factorization"]["post_operator_clipping_allowed"] = True
    with pytest.raises(RecorderProxyFactorizedError, match="contract"):
        validate_contract(ROOT, config)
