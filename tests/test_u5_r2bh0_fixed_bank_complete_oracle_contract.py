from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bh0_fixed_bank_complete_oracle_v1.json"
OBSERVATIONS = (
    ROOT / "configs/u5_r2bh0_fixed_bank_complete_oracle_observations_v1.json"
)
MAPPING_RECEIPT = (
    ROOT
    / "configs/u5_r2bh0_fixed_bank_complete_oracle_mapping_receipt_v1.json"
)


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


def test_bh0_blind_observations_are_complete_and_mapping_sealed() -> None:
    observations = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    expected_sources = {
        "sony_ilme_fx2",
        "panasonic_dc_s1m2es",
        "nikon_d5300",
        "dji_fc4382",
        "om_system_om_3",
        "minolta_dimage_5",
        "phase_one_p45_plus",
        "fujifilm_x_e5",
        "samsung_ek_gn120",
        "google_pixel_7_pro",
    }
    assert observations["status"] == (
        "blind_complete_rankings_frozen_before_mapping_reveal"
    )
    assert not observations["mapping_files_read"]
    assert observations["ties_allowed"] is False
    assert len(observations["blind_sheets"]) == 6
    assert len(observations["rounds"]) == 3
    for round_row in observations["rounds"]:
        assert set(round_row["rankings"]) == expected_sources
        for ranking in round_row["rankings"].values():
            assert len(ranking) == 5
            assert set(ranking) == {"A", "B", "C", "D", "E"}
    assert observations["severe_artifact_review"]["confirmed_severe_count"] == 0


def test_bh0_mapping_receipt_binds_frozen_observations() -> None:
    receipt = json.loads(MAPPING_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == (
        "mapping_identities_bound_after_rankings_commit"
    )
    assert receipt["mapping_revealed"]
    assert receipt["rankings_commit"] == (
        "e1ea9e44164f399de8396aade57041e2d9d36e21"
    )
    assert receipt["observations_sha256"] == sha256_file(OBSERVATIONS)
    assert len(receipt["mapping_sha256_by_round"]) == 3
