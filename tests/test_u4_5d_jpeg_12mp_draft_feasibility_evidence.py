from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5D_JPEG_12MP_DRAFT_FEASIBILITY_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_5d_evidence_binds_exact_negative_result() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_JPEG_12MP_NATIVE_DRAFT_UNAVAILABLE"
    for key in ("config", "contract", "preview_core", "runner", "test", "source"):
        binding = evidence["bindings"][key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
    reports = [
        ROOT / evidence["execution"]["forward_report_path"],
        ROOT / evidence["execution"]["reverse_report_path"],
    ]
    assert all(
        path.stat().st_size == evidence["execution"]["report_bytes"]
        for path in reports
    )
    assert {_sha256(path) for path in reports} == {
        evidence["execution"]["report_sha256"]
    }
    assert evidence["gates"]["pixel_reads_zero"]
    assert evidence["gates"]["forward_reverse_exact"]
    assert not evidence["gates"]["draft_strictly_smaller_than_source"]
    assert not evidence["gates"]["draft_pixels_at_most_target_ceiling"]
