from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.smooth_perceptual_blind_preference import (
    _validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk9_smooth_perceptual_blind_preference_v1.json"
)
DECISION = (
    ROOT
    / "configs/u5_r2bk9_smooth_perceptual_blind_preference_decision_v1.json"
)


def test_bk9_blind_contract_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "contract_frozen_build_ready"
    assert config["population"]["expected_rows"] == 11
    assert config["arms"] == [
        "fixed_bk7_smooth_perceptual_hue_density",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich_velvia_50",
    ]
    gate = config["blind_gate"]
    assert gate["rounds"] == 3
    assert gate["minimum_bk7_overall_round_wins"] == 2
    assert gate["minimum_bk7_total_choice_share"] == 0.4
    assert gate["minimum_bk7_round_wins_vs_ao6"] == 2
    assert gate["minimum_bk7_choice_share_vs_ao6"] == 0.5
    assert not gate["confirmed_severe_artifact_allowed"]
    assert not config["gates"]["threshold_relaxation_allowed"]
    assert not config["gates"]["operator_refit_allowed"]
    assert not config["gates"]["strength_retuning_allowed"]
    assert not config["gates"]["rerender_allowed"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]


def test_bk9_inputs_bind_exact_fixed_inventory() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest, report = _validate_inputs(ROOT, config)
    assert len(manifest) == 11
    assert len(report["records"]) == 33


def test_bk9_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["arms"].reverse()
    with pytest.raises(ValueError):
        _validate_inputs(ROOT, config)


def test_bk9_decision_closes_bk7_without_relaxing_gate() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == (
        "close_bk7_as_generalized_preference_challenger"
    )
    assert not result["blind_gate_passed"]
    assert result["confirmed_severe_artifact_count"] == 0
    assert result["counts"] == {
        "fixed_bk7_smooth_perceptual_hue_density": 16,
        "fixed_ao6_colour_only_t15_c35": 2,
        "safe_rich_velvia_50": 15,
        "tie": 0,
    }
    assert result["round_wins"][
        "fixed_bk7_smooth_perceptual_hue_density"
    ] == 1
    assert result["failed_gate"] == {
        "name": "minimum_bk7_overall_round_wins",
        "observed": 1,
        "required": 2,
    }
    assert result["bk7_vs_ao6_round_wins"][
        "fixed_bk7_smooth_perceptual_hue_density"
    ] == 3
    assert result["bk7_total_choice_share"] >= 0.4
    assert result["bk7_choice_share_vs_ao6"] >= 0.5
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
    assert not decision["production_default_changed"]
