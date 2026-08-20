from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp0_pairwise_preference_source_lock_v1.json"


def test_spcp0_contract_freezes_metadata_only_source() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP0"
    assert payload["source"]["revision"] == (
        "068af97eed82969f15278db3af4bd450176cf6f3"
    )
    assert payload["range_protocol"]["full_zip_download_forbidden"] is True
    assert payload["range_protocol"]["image_member_payload_read_forbidden"] is True
    assert payload["frozen_gates"]["image_payload_bytes_read_exact"] == 0


def test_spcp0_contract_freezes_complete_preference_graph() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gates = payload["frozen_gates"]
    assert gates["scene_count_exact"] == 1000
    assert gates["canonical_image_count_exact"] == 12000
    assert gates["pair_rows_exact"] == 45000
    assert gates["pairs_per_scene_exact"] == 45
    assert gates["score_rows_exact"] == 12000
    assert gates["subject_columns_exact"] == 20
    assert payload["rights"]["image_source_and_redistribution_rights_inferred"] is False


def test_spcp0_successor_is_one_explicit_source_free_operator() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    successor = payload["prospective_successor"]
    assert "one shared source-free bounded" in successor["mechanism"]
    assert "per-image or per-scene routing" in successor["forbidden"]
    assert "direct RGB generation" in successor["forbidden"]
