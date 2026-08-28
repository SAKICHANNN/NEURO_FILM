from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/SF3_A3N_PARVEC_CONTROLLED_FILM_SOURCE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sf3_a3n_evidence_binds_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    formal = evidence["formal_replay"]
    forward = ROOT / formal["forward_report"]
    reverse = ROOT / formal["reverse_report"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == formal["report_bytes"]
    assert _sha256(forward) == formal["report_sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["decision"] == evidence["decision"]
    assert report["stable_evidence_id"] == formal["stable_evidence_id"]
    assert not report["gates"]["complete_three_stock_coverage_present"]
    assert not report["gates"][
        "public_machine_readable_measurements_and_rights_present"
    ]
    assert report["operation_counts"]["preset_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_sf3_a3n_evidence_binds_implementation() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": "configs/sf3_a3n_parvec_controlled_film_source_v1.json",
        "source_audit_sha256": "src/real_film/parvec_controlled_film_source.py",
        "runner_sha256": "scripts/audit_sf3_a3n_parvec_controlled_film_source.py",
        "test_sha256": "tests/test_sf3_a3n_parvec_controlled_film_source.py",
    }
    for key, relative_path in expected.items():
        assert _sha256(ROOT / relative_path) == evidence["implementation"][key]


def test_sf3_a3n_claim_stays_below_stock_admission() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_SOURCE_RIGHTS_OR_PAYLOAD_GAP"
    assert evidence["observed_source_facts"]["target_stock_chapters_present"] == [
        "kodak_ektar_100",
        "kodak_portra_400",
    ]
    assert evidence["observed_source_facts"]["fujifilm_velvia_50_chapter_present"] is False
    assert "no complete Velvia 50/Portra 400/Ektar 100 dataset" in evidence[
        "claim_ceiling"
    ]
