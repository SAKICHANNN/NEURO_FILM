from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_6L_THREE_STOCK_NATIVE_PREVIEW_BACKEND_RESULT.json"


def test_u7_6l_evidence_records_exact_closed_result() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    report_path = ROOT / payload["formal_report"]["path"]
    report = json.loads(report_path.read_text("utf-8"))
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == payload["formal_report"]["sha256"]
    assert report["stable_id"] == payload["formal_report"]["stable_id"]
    assert report["status"] == payload["status"] == "CLOSE_NATIVE_THREE_STOCK_PREVIEW_BACKEND"
    assert report["gates"]["native_wall_seconds"] is False
    assert report["gates"]["native_to_python_wall_ratio"] is True
    assert report["maximum_native_transaction_wall_seconds"] > 5.0
    assert report["gates"]["exact_native_vs_python_png_sha256"] is True
    assert report["gates"]["exact_native_vs_python_rgb_sha256"] is True
