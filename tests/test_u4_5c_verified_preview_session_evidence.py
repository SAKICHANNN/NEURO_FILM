from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5C_VERIFIED_PREVIEW_SESSION_RESULT.json"
REPORT_ROOT = ROOT / "outputs/eval/u4_5c_verified_preview_session_v1"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_u4_5c_evidence_binds_two_passing_scientifically_exact_reports() -> None:
    evidence = _load(EVIDENCE)
    formal = _load(REPORT_ROOT / "formal.json")
    replay = _load(REPORT_ROOT / "formal_replay.json")
    assert evidence["status"] == "PASS_PRIVATE_VERIFIED_PREVIEW_SESSION"
    assert formal["status"] == replay["status"] == evidence["status"]
    assert formal["scientific_stable_id"] == replay["scientific_stable_id"]
    assert formal["scientific_stable_id"] == evidence["identities"]["scientific_stable_id"]
    assert formal["scientific"] == replay["scientific"]
    assert all(formal["scientific"]["gates"].values())


def test_u4_5c_evidence_preserves_warm_and_claim_boundaries() -> None:
    evidence = _load(EVIDENCE)
    results = evidence["results"]
    assert results["maximum_warm_lookup_wall_seconds"] <= 0.3
    assert results["warm_filesystem_reads"] == 0
    assert results["warm_pixel_decodes"] == 0
    assert results["warm_renders"] == 0
    assert results["warm_writes"] == 0
    assert results["new_admission_rejects_disk_mutation"] is True
    assert results["post_admission_disk_mutation_changes_snapshot"] is False
    claim = evidence["claim_ceiling"].lower()
    assert "non-calibrated" in claim
    assert "no render acceleration" in claim
    assert "no" in claim and "multi-stock scientific completion" in claim
