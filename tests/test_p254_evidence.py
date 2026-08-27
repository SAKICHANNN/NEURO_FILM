from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads(
    (
        ROOT
        / "docs/evidence/P254_R1EC_DNG_PROFILE_HUESAT_CALLABLE_INTAKE_RESULT.json"
    ).read_text(encoding="utf-8")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_reports_are_exact_and_pass() -> None:
    execution = EVIDENCE["execution"]
    forward = ROOT / execution["forward_report_path"]
    reverse = ROOT / execution["reverse_report_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == execution["report_bytes_each"]
    assert _sha256(forward) == execution["report_sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["status"] == EVIDENCE["status"]
    assert report["stable_identity"] == execution["stable_identity"]
    assert all(report["scientific"]["controls"].values())
    assert all(report["scientific"]["gates"].values())


def test_evidence_binds_committed_consumer_sources() -> None:
    consumer = EVIDENCE["consumer"]
    for prefix in ("runner", "config", "source_lock"):
        path = ROOT / consumer[f"{prefix}_path"]
        assert path.stat().st_size == consumer[f"{prefix}_bytes"]
        assert _sha256(path) == consumer[f"{prefix}_sha256"]
    assert EVIDENCE["verification"]["p244_p245_guards_changed"] is False
    assert EVIDENCE["verification"]["candidate_counter_changed"] is False
