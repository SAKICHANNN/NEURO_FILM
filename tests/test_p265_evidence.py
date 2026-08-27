from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p265_evidence_binds_exact_fail_closed_reports() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs"
            / "evidence"
            / "P265_ACES2065_ACES2_PQ_LINUX_RUNTIME_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "FAIL_CLOSED_ACES2065_ACES2_PQ_LINUX_RUNTIME"
    assert evidence["formal"]["reports_byte_exact"] is True
    assert evidence["result"]["decoded_working_f32le_sha256"] == (
        "1f1e83dea1805af54d098af1752ba3a25d92b2887e22a6baa11e9e8af662c2c3"
    )
    assert evidence["result"]["windows_linux_encoded_exact"] is False
    assert evidence["result"]["windows_linux_rgb16_exact"] is False
    assert evidence["result"]["windows_linux_png_exact"] is False
    expected = evidence["formal"]["report_sha256"]
    for name in ("formal_forward.json", "formal_reverse.json"):
        path = ROOT / "outputs" / "eval" / "p265_aces2065_aces2_pq_linux_runtime" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
