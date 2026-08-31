from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_2L_PRODUCT_CREATE_ONLY_RENDER_TRANSACTION_RESULT.json"
)


def test_retained_formal_report_passes_all_frozen_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report_path = ROOT / evidence["formal_reports"]["forward_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
