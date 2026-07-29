from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.velvia_group_balanced_operator import (
    VelviaGroupBalancedOperatorError,
    _balanced_rows,
    evaluate_velvia_group_balanced_operator,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ar1_velvia_group_balanced_operator_v1.json"


def test_contract_accepts_exact_frozen_inputs() -> None:
    validate_contract(json.loads(CONFIG.read_bytes()), ROOT)


def test_contract_rejects_weighting_drift() -> None:
    config = json.loads(CONFIG.read_bytes())
    changed = copy.deepcopy(config)
    changed["candidate"]["weighting"] = "row_proportional"
    with pytest.raises(VelviaGroupBalancedOperatorError, match="candidate drift"):
        validate_contract(changed, ROOT)


def test_contract_rejects_removed_exploration_disclosure() -> None:
    config = json.loads(CONFIG.read_bytes())
    changed = copy.deepcopy(config)
    changed["development_disclosure"][
        "full_row_equal_group_weight_probe_seen_before_freeze"
    ] = False
    with pytest.raises(
        VelviaGroupBalancedOperatorError, match="disclosure drift"
    ):
        validate_contract(changed, ROOT)


def test_cross_repetition_gives_equal_domain_weight() -> None:
    chart_source = np.arange(12, dtype=np.float64).reshape(4, 3)
    palette_source = np.arange(21, dtype=np.float64).reshape(7, 3)
    source, target = _balanced_rows(
        chart_source, chart_source + 1, palette_source, palette_source + 1
    )
    assert source.shape == target.shape == (56, 3)
    assert len(source[: 4 * 7]) == len(source[4 * 7 :])


def test_formal_group_balanced_result_is_structurally_rejected() -> None:
    report = evaluate_velvia_group_balanced_operator(
        json.loads(CONFIG.read_bytes()), ROOT
    )
    assert report["stable_evidence_id"] == (
        "5d39094660fae664a9d98b715e72ce6e19f8b32f2438d59c8269df88dfb3f81c"
    )
    assert report["summary"]["full_worst_domain_improvement"] > 0.09
    assert report["summary"]["crossfit_worst_domain_improvement"] > 0.04
    assert report["checks"]["positive_jacobian"] is False
    assert report["automatic_pass"] is False
    assert report["photographic_render_allowed"] is False
