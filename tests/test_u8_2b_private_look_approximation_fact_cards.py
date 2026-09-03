from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u8_2b_private_look_approximation_fact_cards_v1.json"


def test_u8_2b_contract_freezes_private_claim_and_current_runtime() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    claim = config["claim_ceiling"]
    assert config["input"]["runtime_receipt_sha256"] == (
        "548a80583811329260083bd37d0abd122b033141b1c358ef700f07ee044ea756"
    )
    assert config["input"]["profile_sha256"] == (
        "09722cb48c55d1c4c460220c9f59b9ced81801c1b6fe856708a27f5cc99d39fe"
    )
    assert config["input"]["root_license_expected"] == "absent"
    assert config["required_card_order"] == [
        "product",
        "data",
        "model",
        "profiles",
        "release",
    ]
    assert config["required_profile_order"] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
        "generic_bw",
    ]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["public_release"] is False
    assert claim["legal_clearance"] is False


def test_u8_2b_builder_surface_is_reserved() -> None:
    expected = ROOT / "src/inference/product_fact_cards.py"
    audit = ROOT / "scripts/audit_u8_2b_private_look_approximation_fact_cards.py"
    assert expected.name == "product_fact_cards.py"
    assert audit.name == "audit_u8_2b_private_look_approximation_fact_cards.py"
