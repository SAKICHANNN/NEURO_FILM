from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vision3_16mm_contact_sheet_source import (
    Vision3ContactSheetSourceError,
    canonical_json,
    evaluate_contact_sheet_source,
    load_contract,
    load_observations,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu6_vision3_16mm_contact_sheet_source_v1.json"
OBSERVATIONS = (
    ROOT / "configs/u5_r2bu6_vision3_16mm_contact_sheet_observations_v1.json"
)


def test_contract_rejects_rights_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["rights"]["explicit_reuse_license_found"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Vision3ContactSheetSourceError, match="contract drift"):
        load_contract(changed)


def test_observations_reject_contact_sheet_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    payload["contact_sheet"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Vision3ContactSheetSourceError, match="observation binding"):
        load_observations(changed)


def test_contact_sheet_audit_is_exact() -> None:
    contract = load_contract(CONTRACT)
    observations = load_observations(OBSERVATIONS)
    first = evaluate_contact_sheet_source(ROOT, contract, observations)
    second = evaluate_contact_sheet_source(ROOT, contract, observations)
    assert canonical_json(first) == canonical_json(second)


def test_contact_sheet_opens_only_bounded_tiff_integrity_audit() -> None:
    report = evaluate_contact_sheet_source(
        ROOT, load_contract(CONTRACT), load_observations(OBSERVATIONS)
    )
    assert report["gate_pass"] is True
    assert report["stock_ids"] == ["7207", "7219"]
    assert report["pdf_facts"]["page_image_counts"] == [15, 15, 15, 7]
    assert report["pdf_facts"]["embedded_image_count"] == 52
    assert len(report["distinct_condition_labels"]) >= 4
    assert len(report["paired_roles"]) == 3
    assert all(row["explicit_label_match"] for row in report["paired_roles"])
    assert report["external_image_payload_downloads"] == 0
    assert report["external_video_payload_downloads"] == 0
    assert report["decision"] == (
        "open_minimum_sufficient_tiff_integrity_and_role_audit"
    )
