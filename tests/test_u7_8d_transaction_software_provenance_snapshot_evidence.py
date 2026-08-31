from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT_RESULT.json"
)


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git_blob(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def test_evidence_binds_exact_formal_commit_sources_and_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert (
        evidence["decision"]
        == "PASS_PRIVATE_U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT"
    )
    commit = evidence["bindings"]["formal_execution_commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=True
    )

    rows = [
        evidence["bindings"]["contract"],
        evidence["bindings"]["config"],
        evidence["bindings"]["runner"],
        *({"path": path, **row} for path, row in evidence["bindings"]["core"].items()),
        *({"path": path, **row} for path, row in evidence["bindings"]["tests"].items()),
    ]
    for row in rows:
        payload = _git_bytes(commit, row["path"])
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert _git_blob(commit, row["path"]) == row["git_blob"]

    reports = evidence["formal_reports"]
    assert reports["reports_byte_exact"] is True
    assert reports["scientific_identity_exact"] is True
    report_bytes: list[bytes] = []
    for key in ("forward", "reverse"):
        row = reports[key]
        path = ROOT / row["path"]
        payload = path.read_bytes()
        report = json.loads(payload)
        report_bytes.append(payload)
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert report["implementation_commit"] == commit
        assert report["scientific_identity"] == row["scientific_identity"]
        assert report["decision"] == evidence["decision"]
        assert report["stable"]["transaction_start_snapshot_resolution_count"] == 1
        assert report["stable"]["prepublication_recheck_count"] == 1
        assert report["stable"]["software_commit_resolution_count_total"] == 2
        assert report["targeted_test_count"] == 14
    assert report_bytes[0] == report_bytes[1]


def test_parent_negative_and_historical_bindings_remain_immutable() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    parents = evidence["bindings"]["parents"]
    assert (
        sha256_file(ROOT / "docs/evidence/U7_8A_THREE_STOCK_INPUT_BATCH_RESULT.json")
        == parents["u7_8a_evidence_sha256"]
    )
    assert (
        sha256_file(
            ROOT / "docs/evidence/U7_8B_RESUMABLE_THREE_STOCK_BATCH_RESULT.json"
        )
        == parents["u7_8b_evidence_sha256"]
    )
    u7_8c = json.loads(
        (
            ROOT / "docs/evidence/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        sha256_file(
            ROOT / "docs/evidence/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_RESULT.json"
        )
        == parents["u7_8c_evidence_sha256"]
    )
    assert u7_8c["decision"] == parents["u7_8c_decision"]
    assert evidence["pre_evidence_attempts_excluded"][1]["implementation_commit"] == (
        "6e7ea734dfb24e7a28bd2a68d7f95303b6f46f92"
    )


def test_all_frozen_gates_pass_without_claim_expansion() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert all(
        value is True
        for key, value in evidence["gate_results"].items()
        if key not in {"network_requests", "owned_scratch_residue_count"}
    )
    assert evidence["gate_results"]["network_requests"] == 0
    assert evidence["gate_results"]["owned_scratch_residue_count"] == 0
    assert evidence["metrics"]["transaction_start_snapshot_resolution_count"] == 1
    assert evidence["metrics"]["prepublication_recheck_count"] == 1
    assert evidence["metrics"]["software_commit_resolution_count_total"] == 2
    assert evidence["metrics"]["targeted_test_count_per_outer_run"] == 14
    assert evidence["stop_rule"] == {
        "u7_8c_rewritten_or_rerun": False,
        "provenance_normalized_or_stripped": False,
        "cross_commit_resume_opened": False,
        "recipe_schema_or_pixels_changed": False,
        "cache_packaging_raw_hdr_or_stock_scope_opened": False,
        "promotion_opened": False,
    }
    assert "calibrated stock response" in evidence["claim_ceiling"]
