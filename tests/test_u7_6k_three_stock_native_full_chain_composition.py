from __future__ import annotations

import json

from scripts.audit_u7_6k_three_stock_native_full_chain_composition import (
    CONTRACT_SCHEMA,
    ROOT,
    _load_contract,
)


def test_u7_6k_contract_is_frozen_and_bound() -> None:
    path = ROOT / "configs/u7_6k_three_stock_native_full_chain_composition_v1.json"
    contract, digest = _load_contract(path)
    assert contract["schema"] == CONTRACT_SCHEMA
    assert contract["experiment_id"] == "U7.6K"
    assert len(digest) == 64
    assert contract["gates"]["maximum_rgb_absolute_error"] == 0.0
    assert contract["execution"]["production_integration_allowed"] is False


def test_u7_6k_parent_is_exact_pointwise_pass() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u7_6k_three_stock_native_full_chain_composition_v1.json"
        ).read_text("utf-8")
    )
    parent = json.loads(
        (ROOT / contract["parent_positive"]["evidence_path"]).read_text("utf-8")
    )
    assert parent["status"] == "PASS"
    assert parent["observations"]["maximum_lab_absolute_error"] == 0.0
