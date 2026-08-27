from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p264_evidence_matches_exact_formal_reports() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/P264_REAL_DNG_AP0_PQ_INTERCHANGE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    forward = ROOT / evidence["formal"]["forward_report_path"]
    reverse = ROOT / evidence["formal"]["reverse_report_path"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert forward.read_bytes() == reverse.read_bytes()
    assert _sha256(forward) == evidence["formal"]["report_sha256"]
    assert report["decision"] == evidence["status"]
    assert report["result"]["comparison"]["different_component_count"] == 7442
    assert report["result"]["comparison"]["maximum_code_difference"] == 1
    assert report["gates"]["direct_interchange_rgb16_exact"] is False
    assert report["gates"]["direct_interchange_png_exact"] is False
    assert evidence["candidate_count_after_result"] == "2/3"
