from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p275_luckyhdr_source_feasibility import P275Error, _canonical

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p275_luckyhdr_source_feasibility_v1.json"
EVIDENCE = ROOT / "docs/evidence/P275_LUCKYHDR_SOURCE_FEASIBILITY_RESULT.json"


def test_p275_contract_is_frozen_and_bounded() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FORMAL_GIT_OBJECT_AUDIT"
    assert config["limits"]["binary_body_reads"] == 0
    assert config["limits"]["model_loads"] == 0
    assert config["limits"]["pixel_decodes"] == 0
    assert config["frozen_interpretation"]["product_rights_clear"] is False
    assert config["frozen_interpretation"]["candidate3_consumed"] is False
    assert len(config["binary_objects"]) == 4


def test_p275_canonical_json_is_stable() -> None:
    assert _canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


def test_p275_invalid_order_rejects_before_network() -> None:
    from scripts.audit_p275_luckyhdr_source_feasibility import execute

    with pytest.raises(P275Error, match="order"):
        execute(CONFIG, "invalid")


def test_p275_evidence_preserves_zero_read_and_rights_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    result = evidence["result"]
    assert evidence["status"] == "PASS_PRIVATE_BOUNDED_SOURCE_FEASIBILITY"
    assert result["forward_reverse_scientific_exact"] is True
    assert result["binary_body_reads"] == 0
    assert result["pixel_decodes"] == 0
    assert result["model_loads"] == 0
    assert result["candidate3_consumed"] is False
    assert result["product_rights_clear"] is False
    assert evidence["rights_and_product"]["product_admission"] is False
