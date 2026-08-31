from __future__ import annotations

from pathlib import Path

from scripts.audit_u4_5e_receipt_bound_preview_session import build_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5e_receipt_bound_preview_session_v1.json"


def test_formal_report_passes_all_frozen_gates() -> None:
    report = build_report(config_path=CONFIG, order="forward")
    assert report["status"] == "PASS_PRIVATE_RECEIPT_BOUND_PREVIEW_SESSION"
    assert all(report["scientific"]["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
