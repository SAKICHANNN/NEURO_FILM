from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_2D_HIGH_PRECISION_TIFF_ORIENTATION_RESULT.json"


def test_u1_2d_evidence_is_exact_and_passes_all_gates() -> None:
    payload = EVIDENCE.read_bytes()
    report = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == (
        "e72c3dae6c2150374f84c3705dacc4a832fcf1748380fdd948fed85490ea1fe1"
    )
    assert report["status"] == "PASS_RGB16_TIFF_ORIENTATION_INGRESS"
    assert all(report["gates"].values())
    assert len(report["orientation_rows"]) == 8
    assert report["renderer"]["output_dimensions"] == [3, 5]
