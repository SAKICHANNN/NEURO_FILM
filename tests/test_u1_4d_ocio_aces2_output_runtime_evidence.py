from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_4D_OCIO_ACES2_OUTPUT_RUNTIME_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_4d_evidence_binds_runtime_and_dependency_files() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_OCIO_ACES2_OUTPUT_RUNTIME"
    assert evidence["gates"]["all"] is True
    for key in (
        "contract",
        "implementation",
        "config",
        "runner",
        "requirements_windows",
        "requirements_macos_arm",
    ):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_u1_4d_evidence_does_not_promote_hdr_or_renderer() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    excluded = " ".join(evidence["claim_ceiling"]["not_established"])
    assert "HDR file encoding" in excluded
    assert "WorkingImage or default renderer integration" in excluded
    assert evidence["metrics"]["sdr_rec709"]["maximum_scalar_packed_absolute_error"] == 0.0
    assert evidence["metrics"]["hdr_rec2020_pq"]["maximum_scalar_packed_absolute_error"] == 0.0

