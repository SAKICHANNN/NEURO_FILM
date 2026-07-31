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
