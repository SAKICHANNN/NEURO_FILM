from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.log_chroma_fresh_comparison import (
    LogChromaFreshComparisonError,
)
from src.eval.smooth_perceptual_fresh_confirmation import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk8_smooth_perceptual_fresh_confirmation_v1.json"
)


def test_bk8_comparison_contract_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "contract_frozen_implementation_ready"
    assert [row["arm_id"] for row in config["fixed_arms"]] == [
        "fixed_bk7_smooth_perceptual_hue_density",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich_velvia_50",
    ]
    assert config["source_preflight"]["expected_eligible_rows"] == 11
    assert config["automatic_gate"]["expected_outputs"] == 33
    assert (
        config["automatic_gate"][
            "minimum_bk7_population_median_style_delta_e76"
        ]
        == 5.0
    )
    rendering = config["rendering"]
    assert not rendering["operator_refit_allowed"]
    assert not rendering["strength_retuning_allowed"]
    assert not rendering["routing_allowed"]
    assert not rendering["dense_blending_allowed"]
    assert not rendering["hard_clipping_allowed"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]


def test_bk8_contract_binds_eligible_source_and_fixed_operator() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["source_rows"]) == 11
    assert validated["bk7_strength"] == 1.0


def test_bk8_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fixed_arms"][0]["strength"] = 0.75
    with pytest.raises(LogChromaFreshComparisonError):
        validate_contract(ROOT, config)
