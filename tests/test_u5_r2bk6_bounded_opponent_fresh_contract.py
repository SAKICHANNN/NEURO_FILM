from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk6_bounded_opponent_fresh_confirmation_v1.json"
)
ARMS = (
    "fixed_bk5_bounded_opponent_response",
    "fixed_ao6_colour_only_t15_c35",
    "safe_rich_velvia_50",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bk6_three_arm_contract_is_frozen_and_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "contract_frozen_implementation_ready"
    assert tuple(row["arm_id"] for row in config["fixed_arms"]) == ARMS
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


def test_bk6_contract_forbids_fit_retune_routing_and_clipping() -> None:
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


def test_bk6_salience_and_safety_gates_are_fixed() -> None:
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["automatic_gate"]
    assert gate["expected_outputs"] == 36
    assert gate["minimum_bk5_population_median_style_delta_e76"] == 5.0
    assert gate["maximum_per_output_new_code_boundary_fraction_vs_source"] == 0.005
    assert gate["maximum_confirmed_severe_artifact_count"] == 0
