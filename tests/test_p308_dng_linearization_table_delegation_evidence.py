from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/P308_DNG_LINEARIZATION_TABLE_DELEGATION_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p308_evidence_binds_current_inputs_and_exact_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_DNG_LINEARIZATION_TABLE_DELEGATION"
    for key in ("config", "contract", "implementation", "runner", "test"):
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
    assert all(evidence["gates"].values())
