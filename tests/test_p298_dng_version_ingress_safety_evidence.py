from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P298_DNG_VERSION_INGRESS_SAFETY_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p298_evidence_binds_current_inputs_and_exact_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_DNG_VERSION_INGRESS_SAFETY"
    for key in ("config", "implementation", "runner", "test"):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    reports = [
        ROOT / evidence["execution"]["forward_report_path"],
        ROOT / evidence["execution"]["reverse_report_path"],
    ]
    assert all(
        path.stat().st_size == evidence["execution"]["report_bytes"] for path in reports
    )
    assert {_sha256(path) for path in reports} == {
        evidence["execution"]["report_sha256"]
    }
    assert all(evidence["gates"].values())
