from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p98_evidence_binds_exact_passing_report() -> None:
    evidence = json.loads(
        Path("docs/evidence/P98_DNG_FORWARD_RASTER_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    report = evidence["bindings"]["raw_report"]
    assert _sha256(Path(report["path"])) == report["sha256"]
    assert evidence["status"] == "PASS_PRIVATE_DNG_FORWARD_RASTER"
    assert all(evidence["gates"].values())
    assert evidence["execution"]["reports_byte_exact"]


def test_p98_claim_remains_opt_in_and_product_closed() -> None:
    evidence = json.loads(
        Path("docs/evidence/P98_DNG_FORWARD_RASTER_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["execution"]["default_loader_changes"] == 0
    assert evidence["execution"]["decoded_image_artifacts_written"] == 0
    assert evidence["metrics"]["maximum_composition_error"] <= 5e-15
    assert "no arbitrary DNG" in evidence["claim_ceiling"]
    assert "product" in evidence["claim_ceiling"]
