from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bh0_fixed_bank_complete_oracle_v1.json"


def test_bh0_contract_freezes_five_distinct_existing_arms() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    arms = config["fixed_arms"]
    assert len(arms) == 5
    assert len({row["arm_id"] for row in arms}) == 5
    assert config["automatic_gate"]["expected_outputs"] == 50
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["selector_training_allowed"]
    assert not config["production_default_changed"]


def test_bh0_contract_hash_binds_source_and_every_arm_family() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = config["source_preflight"]
    assert sha256_file(ROOT / source["decision"]) == source["decision_sha256"]
    assert sha256_file(ROOT / source["manifest"]) == source["manifest_sha256"]
    assert (
        sha256_file(ROOT / source["visual_review"])
        == source["visual_review_sha256"]
    )
    for row in config["arm_provenance"].values():
        for path_key, hash_key in (
            ("config", "sha256"),
            ("config", "config_sha256"),
            ("decision", "decision_sha256"),
            ("comparison_config", "comparison_config_sha256"),
            ("build_config", "build_config_sha256"),
        ):
            if path_key in row and hash_key in row:
                assert sha256_file(ROOT / row[path_key]) == row[hash_key]


def test_bh0_primary_gate_is_complete_and_cross_round() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))[
        "ranking_protocol"
    ]
    assert protocol["rounds"] == 3
    assert protocol["sources_per_round"] == 10
    assert protocol["arms_per_source"] == 5
    assert protocol["strict_complete_ranking_required"]
    assert not protocol["ties_allowed"]
    assert protocol["global_tie_break_order"] == [
        "fixed_ao6_colour_only_t15_c35",
        "fixed_b0",
        "fixed_ap3_ektachrome_composition",
        "fixed_az0_optical_density_residual",
        "fixed_native_standard_full_strength_1_0",
    ]
    assert (
        protocol["per_source_tie_break"]
        == "fall_back_to_the_selected_global_arm_for_that_leave_one_round_out_fold"
    )
    assert "leave-one-round-out" in protocol["primary_product_value_estimator"]
    assert protocol["minimum_improved_leave_one_round_out_folds"] == 2
    assert protocol["minimum_sources_with_stable_non_global_choice"] == 4
