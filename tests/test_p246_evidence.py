from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P246_ACESCG_OPENEXR_EXACT_CONSUMER_INTAKE_RESULT.json"
REPORT = (
    ROOT
    / "outputs/eval/p246_acescg_openexr_exact_consumer_intake_v1/formal_forward.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p246_evidence_binds_exact_passing_report() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_EXACT_R1DO_OPENEXR_CONSUMER_INTAKE"
    assert report["decision"] == evidence["status"]
    assert _sha256(REPORT) == evidence["formal_execution"]["report_sha256"]
    assert REPORT.stat().st_size == evidence["formal_execution"]["report_bytes"]
    assert report["scientific_identity"] == evidence["formal_execution"][
        "scientific_identity"
    ]
    assert all(report["gates"].values())


def test_p246_evidence_preserves_private_claim_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["producer_identity"]["writer_copied_to_consumer_repository"] is False
    assert evidence["formal_execution"]["network_reads"] == 0
    assert evidence["formal_execution"]["producer_real_photo_reads"] == 0
    assert evidence["formal_execution"]["project_pixel_reads"] == 0
    assert evidence["rights_and_product"] == {
        "new_data_dependency": False,
        "new_model_dependency": False,
        "public_api_or_dependency": False,
        "package_or_schema": False,
        "capability_or_product_mapping": False,
    }
