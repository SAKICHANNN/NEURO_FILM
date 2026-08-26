from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P249_ACES2065_OPENEXR_EXACT_CONSUMER_INTAKE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p249_evidence_binds_exact_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = []
    for binding in evidence["formal_reports"]:
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        reports.append(path.read_bytes())
    assert reports[0] == reports[1]


def test_p249_pass_remains_private_and_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_EXACT_R1DT_ACES2065_CONSUMER_INTAKE"
    assert all(evidence["gates"].values())
    result = evidence["result"]
    assert result["decoded_maximum_absolute_error"] == 0.0
    assert not result["writer_source_copied_to_repository"]
    assert result["network_reads"] == 0
    assert result["project_pixel_reads"] == 0
    assert result["owned_temporary_residue"] == 0
    assert "No neuro_film public API" in evidence["claim_ceiling"]
