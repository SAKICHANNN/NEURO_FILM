from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_3D_RECIPE_HISTORY_EXPORT_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_3d_evidence_binds_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_EXACT_RECIPE_HISTORY_EXPORT"
    formal = evidence["formal_execution"]
    for prefix in ("forward", "reverse"):
        report = ROOT / formal[f"{prefix}_report"]
        assert report.stat().st_size == formal[f"{prefix}_report_bytes"]
        assert _sha256(report) == formal[f"{prefix}_report_sha256"]
        decoded = json.loads(report.read_text(encoding="utf-8"))
        assert decoded["decision"] == "PASS"
        assert decoded["stable_identity"] == formal["stable_identity"]
        assert all(decoded["gates"].values())
    assert formal["scientific_rows_exact_across_orders"] is True
    assert formal["gates_exact_across_orders"] is True


def test_u7_3d_evidence_preserves_three_look_claim_ceiling() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert {row["style"] for row in evidence["rows"]} == {
        "ektar_100",
        "portra_400",
        "velvia_50",
    }
    assert evidence["information_flow"]["existing_output_reads_per_order"] == 0
    assert evidence["information_flow"]["exports_are_recipe_replays_not_output_copies"]
    assert all(evidence["gates"].values())
    assert "No GUI shell" in evidence["claim_ceiling"]
    assert "calibrated stock" in evidence["claim_ceiling"]
