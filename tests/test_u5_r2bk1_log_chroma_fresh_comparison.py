from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.log_chroma_fresh_comparison import (
    ARMS,
    LogChromaFreshComparisonError,
    render_fixed_arms,
    validate_contract,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk1_log_chroma_fresh_comparison_v1.json"
DECISION = (
    ROOT / "configs/u5_r2bk1_log_chroma_fresh_comparison_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bk1_three_arm_contract_is_frozen_and_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "contract_frozen_implementation_ready"
    assert [row["arm_id"] for row in config["fixed_arms"]] == [
        "fixed_bk0_log_chroma",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich_velvia_50",
    ]
    source = config["source_preflight"]
    assert _sha256(ROOT / source["decision"]) == source["decision_sha256"]
    assert _sha256(ROOT / source["manifest"]) == source["manifest_sha256"]
    for arm in config["fixed_arms"]:
        for key in (
            "config",
            "profile_compiler_config",
            "profile",
            "style_statistics",
            "guardrails",
        ):
            if key in arm:
                assert _sha256(ROOT / arm[key]) == arm[f"{key}_sha256"]


def test_bk1_contract_forbids_fit_retune_routing_and_clipping() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rendering = config["rendering"]
    assert rendering["create_only"]
    assert not rendering["per_image_fit_allowed"]
    assert not rendering["operator_refit_allowed"]
    assert not rendering["strength_retuning_allowed"]
    assert not rendering["routing_allowed"]
    assert not rendering["dense_blending_allowed"]
    assert not rendering["hard_clipping_allowed"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]
    assert not config["production_default_changed"]
    assert not config["stock_or_authenticity_claim_allowed"]


def test_bk1_salience_and_safety_gates_are_fixed() -> None:
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["automatic_gate"]
    assert gate["expected_outputs"] == 30
    assert gate["minimum_bk0_population_median_style_delta_e76"] == 5.0
    assert gate["maximum_per_output_new_code_boundary_fraction_vs_source"] == 0.005
    assert gate["maximum_confirmed_severe_artifact_count"] == 0


def test_bk1_contract_validates_and_fixed_arms_render_safely() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=validated["ao6_config"]
    )
    y, x = np.mgrid[0:19, 0:23]
    source = np.stack(
        (
            x / 22.0,
            y / 18.0,
            (x + y) / 40.0,
        ),
        axis=-1,
    ).astype(np.float32)
    outputs = render_fixed_arms(
        source, validated=validated, ao6_artifact=artifact
    )
    assert tuple(outputs) == ARMS
    for output in outputs.values():
        assert output.shape == source.shape
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)


def test_bk1_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["rendering"]["strength_retuning_allowed"] = True
    with pytest.raises(LogChromaFreshComparisonError):
        validate_contract(ROOT, config)


def test_bk1_decision_rejects_v1_without_visual_or_threshold_relaxation() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == "reject_fixed_bk0_log_chroma_v1"
    assert not result["automatic_pass"]
    assert not result["visual_review_allowed"]
    assert result["population_median_style_delta_e76"][
        "fixed_bk0_log_chroma"
    ] < result["bk0_minimum_required_style_delta_e76"]
    assert result["maximum_new_code_boundary_fraction_vs_source"] > result[
        "maximum_allowed_new_code_boundary_fraction_vs_source"
    ]
    assert result["boundary_failure_arm"] == "fixed_bk0_log_chroma"
    assert decision["evidence"]["repeat_report_byte_exact"]
    assert decision["evidence"]["repeat_output_hash_differences"] == 0
    assert not decision["production_default_changed"]
