from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_2E_HIGH_PRECISION_ADOBE_RGB_TIFF_RESULT.json"


def test_u1_2e_evidence_is_exact_and_passes_all_gates() -> None:
    payload = EVIDENCE.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "cf259280d304f4e2540653817d3f28bc49b3c94dc4b859aa1ebb74a4f4e13134"
    )
    assert report["status"] == "PASS_RGB16_ADOBE_RGB_TIFF_INGRESS"
    assert all(report["gates"].values())
    assert len(report["profile_rows"]) == 2
    assert all(row["pixel_exact"] for row in report["profile_rows"])
