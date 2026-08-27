from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sf3_a3k_evidence_binds_formal_reports_and_sources() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/SF3_A3K_ONLANDSCAPE_THREE_STOCK_SOURCE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    formal = evidence["formal_replay"]
    forward = ROOT / formal["forward_report"]
    reverse = ROOT / formal["reverse_report"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert _sha256(forward) == formal["report_sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["decision"] == evidence["decision"]
    assert report["stable_evidence_id"] == formal["stable_evidence_id"]
    assert report["image_requests"] == 0
    assert report["pixel_decodes"] == 0
    assert not report["gates"][
        "explicit_fitting_release_commercial_permissions_present"
    ]


def test_sf3_a3k_evidence_binds_implementation() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/SF3_A3K_ONLANDSCAPE_THREE_STOCK_SOURCE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    implementation = evidence["implementation"]
    expected = {
        "config_sha256": "configs/sf3_a3k_onlandscape_three_stock_source_v1.json",
        "source_audit_sha256": "src/real_film/onlandscape_three_stock_source.py",
        "runner_sha256": "scripts/audit_sf3_a3k_onlandscape_three_stock_source.py",
        "test_sha256": "tests/test_sf3_a3k_onlandscape_three_stock_source.py",
    }
    for key, relative_path in expected.items():
        assert _sha256(ROOT / relative_path) == implementation[key]
