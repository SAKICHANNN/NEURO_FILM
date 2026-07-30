from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_filmmatch_acquisition_is_exact_and_bounded() -> None:
    acquisition = _config()["acquisition"]
    folders = acquisition["folders"]
    assert [row["expected_files"] for row in folders] == [33, 35, 33, 35, 4]
    assert sum(row["expected_files"] for row in folders) == 140
    assert sum(row["expected_bytes"] for row in folders) == 4_255_132_151
    assert acquisition["expected_chart_tiffs"] == 136
    assert acquisition["expected_candidate_pairs"] == 68
    assert acquisition["expected_chart_bytes"] == 4_230_171_722
    assert acquisition["maximum_download_bytes"] >= acquisition["expected_total_bytes"]
    assert acquisition["bounded_folder_recursion_only"] is True
    assert len({row["folder_id"] for row in folders}) == len(folders)


def test_filmmatch_rights_and_claim_ceiling_fail_closed() -> None:
    config = _config()
    assert config["source"]["explicit_licence_observed"] is False
    assert "internal non-redistributed" in config["source"]["allowed_use"]
    assert "redistribute source pixels" in config["source"]["forbidden_use"]
    assert config["gates"]["operator_fitting_allowed"] is False
    assert config["gates"]["training_allowed"] is False
    assert config["gates"]["production_integration_allowed"] is False
    assert "no identified scene-to-stock response" in config["claim_ceiling"]


def test_filmmatch_pilot_hashes_and_unknowns_are_frozen() -> None:
    pilot = _config()["observed_pilot"]
    assert pilot["film_file"]["sha256"] == (
        "100102db3c8fca8d02a988f0717ef2ce4e5c423dc88c0f793264de4fedabb663"
    )
    assert pilot["digital_file"]["sha256"] == (
        "f25acad0e9a8fd1a51ab32661f38e403954cd9331f886588231b4bff1fc6ea8b"
    )
    assert pilot["film_file"]["embedded_icc"] is False
    assert pilot["digital_file"]["embedded_icc"] is False
    assert "complete pair mapping" in pilot["unresolved"]
    assert "Ektachrome scan encoding and scanner interpretation" in pilot["unresolved"]
