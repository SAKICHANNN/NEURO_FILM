from __future__ import annotations

import json
from pathlib import Path

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bl8s_filmmatch_replacement_source_preflight_v1.json"
BL7_MANIFEST = ROOT / "outputs/u5_r2bl7s_filmmatch_fresh_source_preflight_v1/run_a/manifest.json"
DECISION = ROOT / "configs/u5_r2bl8s_filmmatch_replacement_source_preflight_decision_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bl8s_replacement_contract_is_frozen_and_parent_bound() -> None:
    config = _config()
    validate_contract(ROOT, config)
    candidates = config["candidates"]
    assert len(candidates) == 18
    assert len({row["make"] for row in candidates}) == 18
    assert config["preflight"]["minimum_hash_clean_decoded_rows"] == 14
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False


def test_bl8s_retains_only_clean_bl7_rows_and_adds_exact_replacements() -> None:
    config = _config()
    current = {row["id"] for row in config["candidates"]}
    bl7_rows = json.loads(BL7_MANIFEST.read_text(encoding="utf-8"))
    excluded = {
        "samsung_galaxy_s23",
        "sony_ilce_7m5",
        "pentax_k_r",
        "om_system_om_5_mark_ii",
        "canon_eos_r6_mark_iii",
    }
    retained = {row["id"] for row in bl7_rows} - excluded
    replacements = {
        "leaf_credo_40",
        "motorola_moto_g_5s_plus",
        "epson_r_d1s",
        "apple_iphone_8",
        "kodak_easyshare_z981",
    }
    assert len(retained) == 13
    assert current == retained | replacements
    assert current.isdisjoint(excluded)


def test_bl8s_decision_binds_repeat_exact_outputs_and_excludes_chart() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["automatic_pass"] is True
    assert decision["two_run_report_manifest_and_sheet_identity"] is True
    assert decision["visual_pass"] is True
    assert decision["eligible_ordinary_photo_rows"] == 17
    assert decision["content_exclusions"] == ["apple_iphone_8"]
    assert (
        decision["decision"]
        == "pass_replacement_fresh_population_open_fixed_bl5_vs_ao6_confirmation"
    )
