from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.fixed_gaussian_log_odds_capacity import (
    evaluate_fixed_gaussian_log_odds_capacity,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/u5_r2as0_fixed_gaussian_log_odds_capacity_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_contract_rejects_real_pair_rescue_or_clipping() -> None:
    config = _config()
    config["frozen_boundary"]["real_film_or_display_proxy_rows_accessed"] = True
    with pytest.raises(ValueError):
        validate_contract(config, ROOT)
    config = _config()
    config["candidate"]["hard_output_clipping"] = True
    with pytest.raises(ValueError):
        validate_contract(config, ROOT)


def test_small_audit_is_repeat_exact_and_structurally_safe() -> None:
    config = _config()
    config["fit"]["development_rows"] = 128
    config["fit"]["confirmation_rows"] = 96
    config["fit"]["base_maximum_function_evaluations"] = 500
    config["audit"]["cube_axis_size"] = 9
    config["audit"]["jacobian_axis_size"] = 5
    first = evaluate_fixed_gaussian_log_odds_capacity(
        copy.deepcopy(config), ROOT
    )
    second = evaluate_fixed_gaussian_log_odds_capacity(
        copy.deepcopy(config), ROOT
    )
    assert first == second
    assert len(first["rows"]) == 4
    assert all(row["output_minimum"] >= 0.0 for row in first["rows"])
    assert all(row["output_maximum"] <= 1.0 for row in first["rows"])
    assert all(
        row["maximum_neutral_axis_residual"] == 0.0
        for row in first["rows"]
    )
    assert all(
        row["serialization_replay_maximum_absolute_error"] == 0.0
        for row in first["rows"]
    )
