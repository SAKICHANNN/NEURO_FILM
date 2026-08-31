from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_u7_8c_evidence_binds_frozen_sources_and_reports() -> None:
    evidence = _payload()
    commit = evidence["bindings"]["implementation_commit"]
    for key in (
        "config",
        "contract",
        "runner",
        "u7_8a_transaction_core",
        "three_look_child_core",
    ):
        binding = evidence["bindings"][key]
        resolved = subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{binding['path']}"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        assert resolved == binding["git_blob"]

    lock = evidence["bindings"]["execution_lock_v2"]
    assert _sha256(ROOT / lock["path"]) == lock["sha256"]
    for binding in evidence["formal_reports"].values():
        report = ROOT / binding["path"]
        assert report.stat().st_size == binding["bytes"]
        assert _sha256(report) == binding["sha256"]


def test_u7_8c_formal_pair_fails_only_the_frozen_cross_run_identity_family() -> None:
    evidence = _payload()
    forward = json.loads(
        (ROOT / evidence["formal_reports"]["forward"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    reverse = json.loads(
        (ROOT / evidence["formal_reports"]["reverse"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["decision"] == (
        "FAIL_CLOSED_U7_8C_FORWARD_REVERSE_RECIPE_IDENTITY_DRIFT"
    )
    assert (
        forward["decision"]
        == reverse["decision"]
        == ("PASS_PRIVATE_U7_8C_CANON_REAL_SCALE_INPUT_BATCH")
    )
    assert forward["source_identity"] == reverse["source_identity"]
    assert forward["decode_counts"] == reverse["decode_counts"]
    assert forward["controls"] == reverse["controls"]
    assert forward["gate_results"] == reverse["gate_results"]

    forward_jobs = {row["job_id"]: row for row in forward["transaction"]["rows"]}
    reverse_jobs = {row["job_id"]: row for row in reverse["transaction"]["rows"]}
    output_matches: list[bool] = []
    recipe_matches: list[bool] = []
    child_matches: list[bool] = []
    for job_id in sorted(forward_jobs):
        first = forward_jobs[job_id]
        second = reverse_jobs[job_id]
        child_matches.append(
            first["child_manifest_sha256"] == second["child_manifest_sha256"]
        )
        for first_row, second_row in zip(first["rows"], second["rows"], strict=True):
            assert first_row["style_id"] == second_row["style_id"]
            output_matches.append(
                first_row["output_sha256"] == second_row["output_sha256"]
            )
            recipe_matches.append(
                first_row["recipe_sha256"] == second_row["recipe_sha256"]
            )

    assert sum(output_matches) == 18
    assert sum(recipe_matches) == 0
    assert sum(child_matches) == 0
    assert (
        forward["transaction"]["receipt_sha256"]
        != reverse["transaction"]["receipt_sha256"]
    )
    assert forward["transaction"]["batch_id"] != reverse["transaction"]["batch_id"]
    assert forward["scientific_identity"] != reverse["scientific_identity"]
    assert (
        evidence["cross_run_gates"]["forward_reverse_transaction_identities_exact"]
        is False
    )


def test_u7_8c_preserves_resource_and_claim_boundaries() -> None:
    evidence = _payload()
    assert evidence["resource_result"]["both_runs_pass_resource_gates"] is True
    assert evidence["successful_mechanics"]["network_requests"] == 0
    assert evidence["successful_mechanics"]["owned_scratch_residue_count"] == 0
    assert evidence["diagnosis"]["third_run_or_hash_normalization_used"] is False
    assert evidence["stop_rule"] == {
        "exact_family_closed": True,
        "sources_or_order_changed": False,
        "thresholds_or_renderer_changed": False,
        "rerun_to_obtain_favourable_identity": False,
        "promotion_opened": False,
    }
    assert "film-inspired" in evidence["claim_ceiling"]
    assert "calibrated stock response" in evidence["claim_ceiling"]
