from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.orthogonal_residual_blind_preference import ARMS, _validate_inputs


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk15_orthogonal_residual_blind_preference_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bk15_blind_contract_is_frozen() -> None:
    config = _config()
    assert config["status"] == "contract_frozen_build_ready"
    assert config["population"]["expected_rows"] == 12
    assert tuple(config["arms"]) == ARMS
    gate = config["blind_gate"]
    assert gate["rounds"] == 3
    assert gate["minimum_bk10_overall_round_wins"] == 2
    assert gate["minimum_bk10_total_choice_share"] == 0.35
    assert gate["minimum_bk10_round_wins_vs_each_control"] == 2
    assert gate["minimum_bk10_choice_share_vs_each_control"] == 0.5
    assert not gate["confirmed_severe_artifact_allowed"]
    assert not config["gates"]["threshold_relaxation_allowed"]
    assert not config["gates"]["operator_refit_allowed"]
    assert not config["gates"]["strength_retuning_allowed"]
    assert not config["gates"]["rerender_allowed"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]


def test_bk15_inputs_bind_exact_fixed_inventory() -> None:
    manifest, report = _validate_inputs(ROOT, _config())
    assert len(manifest) == 12
    assert len(report["records"]) == 48


def test_bk15_contract_mutation_fails_closed() -> None:
    config = _config()
    config["arms"].reverse()
    with pytest.raises(ValueError):
        _validate_inputs(ROOT, config)
