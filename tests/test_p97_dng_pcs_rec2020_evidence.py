from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p97_evidence_binds_exact_passing_report() -> None:
    evidence = json.loads(
        Path("docs/evidence/P97_DNG_PCS_REC2020_COMPOSITION_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    report = evidence["bindings"]["raw_report"]
    assert _sha256(Path(report["path"])) == report["sha256"]
    assert evidence["status"] == "PASS_PRIVATE_DNG_PCS_REC2020_COMPOSITION"
    assert all(evidence["gates"].values())
    assert evidence["execution"]["reports_byte_exact"]


def test_p97_claim_remains_raster_and_loader_closed() -> None:
    evidence = json.loads(
        Path("docs/evidence/P97_DNG_PCS_REC2020_COMPOSITION_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["execution"]["raster_sample_rgb_reads"] == 0
    assert evidence["metrics"]["composition_max_abs_error"] <= 5e-15
    assert evidence["metrics"]["white_to_unit_max_abs_error"] <= 0.001
    assert "loader integration" in evidence["claim_ceiling"]
