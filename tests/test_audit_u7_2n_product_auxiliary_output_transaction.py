from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2N_PRODUCT_AUXILIARY_OUTPUT_TRANSACTION_RESULT.json"


def test_recorded_formal_report_passes_all_frozen_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    formal = evidence["formal_reports"]
    forward = ROOT / formal["forward_path"]
    reverse = ROOT / formal["reverse_path"]
    payload = forward.read_bytes()
    assert payload == reverse.read_bytes()
    assert len(payload) == formal["bytes_each"]
    assert hashlib.sha256(payload).hexdigest() == formal["sha256"]
    report = json.loads(payload)
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
