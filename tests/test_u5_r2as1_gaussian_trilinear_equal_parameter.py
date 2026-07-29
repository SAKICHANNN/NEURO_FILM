from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.gaussian_trilinear_equal_parameter import (
    evaluate_gaussian_trilinear_equal_parameter,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2as1_gaussian_trilinear_equal_parameter_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_requires_honest_post_result_disclosure() -> None:
    config = _config()
    config["disclosure"]["exploratory_trilinear_results_seen_before_freeze"] = (
        False
    )
    with pytest.raises(ValueError):
        validate_contract(config, ROOT)


def test_small_diagnostic_is_repeat_exact_and_equal_parameter() -> None:
    config = _config()
    first = evaluate_gaussian_trilinear_equal_parameter(
        copy.deepcopy(config), ROOT
    )
    second = evaluate_gaussian_trilinear_equal_parameter(
        copy.deepcopy(config), ROOT
    )
    assert first == second
    assert len(first["rows"]) == 4
    assert all(
        row["gaussian"]["maximum_neutral_axis_residual"] == 0.0
        and row["trilinear"]["maximum_neutral_axis_residual"] == 0.0
        for row in first["rows"]
    )
