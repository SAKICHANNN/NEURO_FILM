from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U8_2A_PRIVATE_PRODUCT_RUNTIME_SBOM_RESULT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_object(commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    ).stdout.strip()


def test_u8_2a_formal_reports_and_sources_are_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    formal = evidence["formal_execution"]
    payloads = [(ROOT / path).read_bytes() for path in formal["reports"]]
    assert payloads[0] == payloads[1]
    assert len(payloads[0]) == formal["report_bytes"]
    assert hashlib.sha256(payloads[0]).hexdigest() == formal["report_sha256"]
    report = json.loads(payloads[0])
    assert report["status"] == evidence["status"]
    assert all(report["gates"].values())
    assert report["inventory"]["count"] == 13
    for path, blob in formal["source_git_objects"].items():
        assert _git_object(formal["source_commit"], path) == blob


def test_u8_2a_published_boms_match_formal_documents() -> None:
    from src.inference.product_runtime_sbom import validate_sbom_documents

    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    published = evidence["published_artifacts"]
    cdx_path = ROOT / published["cyclonedx"]["path"]
    spdx_path = ROOT / published["spdx"]["path"]
    assert cdx_path.stat().st_size == published["cyclonedx"]["bytes"]
    assert spdx_path.stat().st_size == published["spdx"]["bytes"]
    assert _sha256(cdx_path) == published["cyclonedx"]["sha256"]
    assert _sha256(spdx_path) == published["spdx"]["sha256"]
    validate_sbom_documents(cdx_path.read_bytes(), spdx_path.read_bytes())
    assert published["repo_relative"] is True
    assert published["p_backed_parent_junction"] is True
    assert published["stage_residue_zero"] is True


def test_u8_2a_evidence_preserves_release_and_stock_claim_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert all(evidence["gates"].values())
    assert evidence["claim"] == {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
        "public_release": False,
        "legal_clearance": False,
        "vulnerability_analysis": False,
    }
