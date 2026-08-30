from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2I_PRODUCT_LOOK_CATALOG_ENFORCEMENT_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2i_evidence_binds_sources_and_exact_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for key in ("contract", "config", "cli", "formal_runner"):
        binding = evidence["bindings"][key]
        assert_historical_evidence_binding(ROOT, binding)

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == reports["bytes_each"]
    assert _sha(forward) == reports["sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
    assert all(
        row["input_decode_attempted"] is False and row["residue_count"] == 0
        for row in report["rejection_results"].values()
    )
