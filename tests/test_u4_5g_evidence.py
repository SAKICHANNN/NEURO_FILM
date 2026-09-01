from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5G_PREVIEW_PNG_COMPRESSION_BINDING_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_5g_evidence_binds_accepted_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U4_5G_PREVIEW_PNG_COMPRESSION_BINDING"
    reports = []
    for order in ("forward", "reverse"):
        binding = evidence["formal_reports"][order]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == evidence["status"]
        assert report["scientific_identity"] == evidence["formal_reports"][
            "scientific_identity"
        ]
        assert all(report["gates"].values())
        reports.append(report)
    assert reports[0]["scientific_identity"] == reports[1]["scientific_identity"]


def test_u4_5g_evidence_preserves_claim_and_failed_attempt_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["excluded_attempt"]["accepted_report_count_before_correction"] == 0
    assert evidence["accepted_results"]["all_levels_decoded_rgb8_exact"] is True
    assert evidence["accepted_results"]["all_levels_icc_exact"] is True
    assert evidence["accepted_results"]["all_formal_gates_pass"] is True
    assert "Look Approximations" in evidence["claim_ceiling"]
    assert "No calibrated or physical film-stock response" in evidence["claim_ceiling"]
