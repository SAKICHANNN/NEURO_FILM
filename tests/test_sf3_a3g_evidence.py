from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    ROOT
    / "docs/evidence/SF3_A3G_RICHARD_PHOTO_LAB_STOCK_EXPOSURE_SOURCE_LOCK_RESULT.json"
)


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_sf3_a3g_evidence_closes_before_pixels() -> None:
    evidence = _evidence()
    assert evidence["status"] == "FAIL_CLOSED_DYNAMIC_PAGE_IDENTITY"
    assert evidence["decision"] == "FAIL_CLOSED_RPL_STOCK_SOURCE_STRUCTURE"
    assert evidence["formal_replay"]["reports_byte_exact"] is False
    assert evidence["observed_source_facts"]["member_payload_reads"] == 0
    assert evidence["observed_source_facts"]["image_member_reads"] == 0
    assert evidence["observed_source_facts"]["pixel_decodes"] == 0
    assert evidence["gate_outcomes"]["page_identity_exact"] is False
    assert evidence["gate_outcomes"]["central_identity_exact"] is True
    assert set(
        evidence["observed_source_facts"]["stock_ladders_structurally_present"]
    ) == {
        "kodak_portra_400",
        "kodak_ektar_100",
        "kodak_tri_x_400",
        "ilford_hp5_plus",
    }


def test_sf3_a3g_evidence_binds_exact_implementation() -> None:
    evidence = _evidence()
    implementation = evidence["implementation"]
    assert implementation["commit"] == "191209a83b98f7c5100646af66dd2083482d7917"
    expected = {
        "config_sha256": "configs/sf3_a3g_richard_photo_lab_stock_exposure_source_lock_v1.json",
        "source_lock_sha256": "src/real_film/richard_photo_lab_stock_source.py",
        "runner_sha256": "scripts/audit_sf3_a3g_richard_photo_lab_stock_exposure_source.py",
        "test_sha256": "tests/test_sf3_a3g_richard_photo_lab_stock_exposure_source.py",
    }
    for key, relative in expected.items():
        assert (
            implementation[key]
            == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        )
