from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.bounded_opponent_regression import (
    BoundedOpponentRegressionError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk5_bounded_opponent_response_v1.json"
DECISION = (
    ROOT / "configs/u5_r2bk5_bounded_opponent_response_decision_v1.json"
)


def test_bk5_regression_contract_binds_bk2_severe_source() -> None:
    validated = validate_contract(
        ROOT, json.loads(CONFIG.read_text(encoding="utf-8"))
    )
    assert validated["source_path"].name == "phaseone_p25plus.png"
    assert validated["source_sha256"] == (
        "a81bd65ce6b88f50d26862ae6f38a03109a7ca02d782eb87ffc3368d657afe9a"
    )


def test_bk5_regression_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["strength"] = 0.75
    with pytest.raises(BoundedOpponentRegressionError):
        validate_contract(ROOT, config)


def test_bk5_decision_opens_only_third_fresh_confirmation() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == (
        "retain_primitive_open_third_fresh_confirmation"
    )
    assert decision["status"] == "primitive_and_failure_regression_pass"
    assert result["automatic_pass"]
    assert result["maximum_zero_to_one_red_code_delta_e76"] <= 1.0
    assert result["bk5_style_delta_e76_on_regression"] >= 5.0
    assert result["bk5_new_boundary_fraction"] == 0.0
    assert result["confirmed_severe_artifact_count"] == 0
    assert result["bk2_failure_reproduced"]
    assert result["bk5_failure_regression_pass"]
    assert decision["evidence"]["repeat_file_hash_differences"] == 0
    assert not result["thresholds_or_strengths_changed"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
    assert not decision["production_default_changed"]
