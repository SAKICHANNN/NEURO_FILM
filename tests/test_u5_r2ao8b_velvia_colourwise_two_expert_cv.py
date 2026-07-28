from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao8b_velvia_colourwise_two_expert_cv import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.velvia_colourwise_two_expert_cv import (
    VelviaTwoExpertCVError,
    deterministic_domain_folds,
    support_router_weights,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/u5_r2ao8b_velvia_colourwise_two_expert_cv_v1.json"
)


def _config() -> dict:
    return load_config(CONFIG_PATH, expected_sha256=CONFIG_SHA256)


def test_contract_validates_exact_proxy_lineage() -> None:
    ao5 = validate_contract(ROOT, _config())
    assert ao5["inputs"]["combined_rows"] == 71


def test_contract_rejects_post_freeze_gate_change() -> None:
    changed = copy.deepcopy(_config())
    changed["gates"]["minimum_selected_rgb_rmse_gain_over_combined"] = 0.0
    with pytest.raises(VelviaTwoExpertCVError, match="contract"):
        validate_contract(ROOT, changed)


def test_config_hash_fails_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG_PATH, expected_sha256="0" * 64)


def test_domain_folds_are_exact_repeat_and_complete() -> None:
    first = deterministic_domain_folds(24, 5, 2026072801)
    second = deterministic_domain_folds(24, 5, 2026072801)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == set(range(5))
    assert np.max(np.bincount(first)) - np.min(np.bincount(first)) <= 1


def test_support_router_uses_only_query_and_development_sources() -> None:
    chart = np.asarray(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1]]
    )
    palette = np.asarray(
        [[0.8, 0.8, 0.8], [0.7, 0.8, 0.8], [0.8, 0.7, 0.8]]
    )
    query = np.asarray([[0.12, 0.11, 0.1], [0.78, 0.79, 0.8]])
    hard, _ = support_router_weights(
        query,
        chart,
        palette,
        kind="hard",
        temperature=None,
        scale_floor=1e-6,
    )
    np.testing.assert_array_equal(hard, np.asarray([1.0, 0.0]))
    soft, _ = support_router_weights(
        query,
        chart,
        palette,
        kind="softmax",
        temperature=0.5,
        scale_floor=1e-6,
    )
    assert soft[0] > 0.5
    assert soft[1] < 0.5
