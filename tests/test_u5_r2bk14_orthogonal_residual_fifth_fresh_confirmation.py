from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.orthogonal_residual_fifth_fresh_confirmation import (
    ARMS,
    render_fixed_arms,
    validate_contract,
)
from src.eval.log_chroma_fresh_comparison import LogChromaFreshComparisonError
from src.film_physics.profile_consumer import compile_standalone_profile_artifact


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bk14_orthogonal_residual_fifth_fresh_confirmation_v1.json"
)
DECISION = (
    ROOT
    / "configs/u5_r2bk14_orthogonal_residual_fifth_fresh_confirmation_decision_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bk14_contract_binds_fixed_arms_and_fifth_source() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    assert tuple(row["arm_id"] for row in config["fixed_arms"]) == ARMS
    assert len(validated["source_rows"]) == 12
    assert len({row["make"] for row in validated["source_rows"]}) == 12
    assert config["automatic_gate"]["expected_outputs"] == 48
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]
    assert not config["rendering"]["strength_retuning_allowed"]
    assert not config["rendering"]["routing_allowed"]
    assert not config["rendering"]["dense_blending_allowed"]


def test_bk14_render_is_finite_bounded_and_repeat_exact() -> None:
    validated = validate_contract(ROOT, _config())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=validated["ao6_config"]
    )
    rng = np.random.default_rng(20260731)
    source = rng.uniform(0.02, 0.98, size=(9, 13, 3)).astype(np.float32)
    whole = render_fixed_arms(
        source, validated=validated, ao6_artifact=artifact
    )
    repeated = render_fixed_arms(
        source, validated=validated, ao6_artifact=artifact
    )
    assert set(whole) == set(ARMS)
    for arm in ARMS:
        assert whole[arm].shape == source.shape
        assert np.all(np.isfinite(whole[arm]))
        assert np.all((whole[arm] >= 0.0) & (whole[arm] <= 1.0))
        assert np.array_equal(whole[arm], repeated[arm])


def test_bk14_rejects_contract_drift() -> None:
    config = _config()
    config["rendering"]["strength_retuning_allowed"] = True
    with pytest.raises(LogChromaFreshComparisonError):
        validate_contract(ROOT, config)


def test_bk14_decision_opens_blind_comparison_without_promotion() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == (
        "retain_bk10_open_independent_blind_comparison"
    )
    assert result["automatic_pass"]
    assert result["visual_pass"]
    assert result["confirmed_severe_artifact_count"] == 0
    assert result["outputs"] == 48
    assert result["maximum_new_code_boundary_fraction_vs_source"] == 0.0
    assert result["bk10_population_median_increment_vs_safe_delta_e76"] >= 2.0
    assert decision["evidence"][
        "repeat_report_and_all_output_hashes_exact"
    ]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
    assert not decision["selector_training_allowed"]
    assert not decision["strength_retuning_allowed"]
    assert not decision["production_default_changed"]
