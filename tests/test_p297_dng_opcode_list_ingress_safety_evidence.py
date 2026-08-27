from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P297_DNG_OPCODE_LIST_INGRESS_SAFETY_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p297_evidence_binds_current_implementation_and_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_DNG_OPCODE_LIST_INGRESS_SAFETY"
    for key in ("config", "implementation", "runner", "test"):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]
    reports = [
        ROOT / "outputs/eval/p297_dng_opcode_list_ingress_safety/formal_forward.json",
        ROOT / "outputs/eval/p297_dng_opcode_list_ingress_safety/formal_reverse.json",
    ]
    assert all(
        path.stat().st_size == evidence["execution"]["report_bytes"] for path in reports
    )
    assert {_sha256(path) for path in reports} == {
        evidence["execution"]["report_sha256"]
    }
    assert all(evidence["gates"].values())
