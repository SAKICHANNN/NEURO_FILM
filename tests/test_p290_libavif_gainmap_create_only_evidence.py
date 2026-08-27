from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P290_LIBAVIF_GAINMAP_CREATE_ONLY_API_RESULT.json"


def test_p290_evidence_matches_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = ROOT / evidence["formal"]["forward_report_path"]
    reverse = ROOT / evidence["formal"]["reverse_report_path"]
    payload = forward.read_bytes()

    assert evidence["status"] == "PASS_PRIVATE_LIBAVIF_GAINMAP_CREATE_ONLY_API"
    assert payload == reverse.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == evidence["formal"]["report_sha256"]
    report = json.loads(payload)
    assert report["stable_identity"] == evidence["formal"]["scientific_identity"]
    assert report["scientific"]["receipt_sha256"] == evidence["formal"][
        "receipt_sha256"
    ]
    assert all(report["scientific"]["gates"].values())


def test_p290_is_private_terminal_wrapper_only() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["consumer_mapping"] is False
    assert evidence["adjacent_wrapper_expansion_closed"] is True
    assert evidence["candidate_count_after_result"] == "2/3"
    assert evidence["interface"]["sdr_tone_policy_included"] is False
