from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_8G_INPUT_BATCH_STAGE_OWNERSHIP_REPAIR_RESULT.json"
)


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def test_u7_8g_evidence_binds_exact_commits_reports_and_pass() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert (
        evidence["decision"]
        == "PASS_PRIVATE_U7_8G_INPUT_BATCH_STAGE_OWNERSHIP_REPAIR"
    )
    assert evidence["outer_replay"] == {
        "forward_report_bytes": 2291,
        "forward_report_sha256": (
            "0b7e2e6a43e6cec38fad542fdfdbed31840518750e3515d179f91b90d0da8691"
        ),
        "reports_byte_exact": True,
        "reverse_report_bytes": 2291,
        "reverse_report_sha256": (
            "0b7e2e6a43e6cec38fad542fdfdbed31840518750e3515d179f91b90d0da8691"
        ),
        "stable_identity": (
            "f23a17dbcaceb428830b4c997d38dd59e69e66fd220e17fcc0ee6eee44e29508"
        ),
    }
    assert all(
        value is True
        for name, value in evidence["gate_results"].items()
        if name not in {"network_requests", "ordinary_owned_stage_residue_count"}
    )
    assert evidence["gate_results"]["network_requests"] == 0
    assert evidence["gate_results"]["ordinary_owned_stage_residue_count"] == 0
    assert evidence["controls"]["replacement_failure"] == {
        "displaced_owned_stage_preserved": True,
        "failure_rejected": True,
        "final_destination_absent": True,
        "foreign_payload_preserved": True,
        "foreign_stage_preserved": True,
    }


def test_u7_8g_source_objects_and_lf_hashes_are_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    bindings = evidence["bindings"]
    object_rows = (
        (
            bindings["contract_commit"],
            "docs/planning/U7_8G_INPUT_BATCH_STAGE_OWNERSHIP_REPAIR_CONTRACT.md",
            bindings["contract_blob"],
            bindings["contract_git_lf_sha256"],
        ),
        (
            bindings["execution_commit"],
            "configs/u7_8g_input_batch_stage_ownership_repair_v1.json",
            bindings["config_blob"],
            bindings["config_sha256"],
        ),
        (
            bindings["execution_commit"],
            "scripts/audit_u7_8g_input_batch_stage_ownership_repair.py",
            bindings["runner_blob"],
            bindings["runner_git_lf_sha256"],
        ),
    )
    for commit, path, expected_blob, expected_sha256 in object_rows:
        actual_blob = subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        payload = _git_bytes(commit, path)
        assert actual_blob == expected_blob
        assert hashlib.sha256(payload).hexdigest() == expected_sha256

    for path, expected_sha256 in bindings[
        "implementation_git_lf_sha256"
    ].items():
        payload = _git_bytes(bindings["implementation_commit"], path)
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
