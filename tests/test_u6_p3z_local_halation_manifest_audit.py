from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3z_local_halation_manifest_audit_v1.json"


def test_p3z_contract_is_bounded_and_metadata_only() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    scope = contract["scope"]
    assert contract["status"] == "contract_frozen_before_content_scan"
    assert scope["extensions"] == [".json", ".jsonl"]
    assert scope["maximum_files"] == 256
    assert scope["maximum_total_bytes"] == 64 * 1024 * 1024
    assert scope["pixel_reads_allowed"] is False


def test_p3z_contract_requires_all_protocol_fields() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "contains every P3Y required_row_field" in contract["candidate_rule"]
    forbidden = " ".join(contract["forbidden"])
    assert "partial key overlap" in forbidden
    assert "image, PDF, archive or media" in forbidden
