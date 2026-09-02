from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/U7_9D_PRIVATE_RUNTIME_SOURCE_SCOPE_BINDING_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_9d_evidence_binds_final_reports_runtime_and_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_U7_9D_RUNTIME_SOURCE_SCOPE_BINDING"
    assert evidence["automatic_pass"] is True
    assert evidence["formal_execution"]["byte_exact"] is True
    assert all(evidence["gates"].values())
    for row in evidence["formal_execution"]["reports"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
    receipt = ROOT / evidence["runtime"]["receipt_path"]
    assert receipt.stat().st_size == evidence["runtime"]["receipt_bytes"]
    assert _sha256(receipt) == evidence["runtime"]["receipt_sha256"]
    commit = evidence["formal_execution"]["source_commit"]
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(
            ROOT, {"path": relative, "commit": commit, **binding}
        )


def test_u7_9d_evidence_preserves_claim_and_cleanup_truth() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    claim = evidence["claim_ceiling"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["standalone_runtime"] is False
    assert claim["public_release"] is False
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    cleanup = evidence["retained_cleanup_facts"]
    assert cleanup["deletion_blocked_before_execution_by_platform_policy"] is True
    assert cleanup["files_deleted"] == 0
    assert cleanup["future_scientific_or_product_authority"] is False


def test_u7_9d_readme_points_to_verified_versioned_runtime_and_policy() -> None:
    readme = (ROOT / "README.md").read_text("utf-8")
    assert "private-product-runtime-u7-9d-78d4931" in readme
    assert "current `HEAD` to descend from the installed commit" in readme
    assert "Committed documentation, evidence and test-only descendants" in readme
    assert "private-product-runtime\\kmcfm-look.exe" not in readme
