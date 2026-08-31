from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_8B_RESUMABLE_THREE_STOCK_BATCH_RESULT.json"
CONFIG = ROOT / "configs/u7_8b_resumable_three_stock_batch_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object(revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def test_u7_8b_evidence_binds_formal_commit_and_all_frozen_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_U7_8B_RESUMABLE_THREE_STOCK_BATCH"
    assert evidence["config_sha256"] == _sha256(CONFIG)
    assert set(evidence["gate_results"]) == set(config["gates"]) | {
        "all_100_jobs_published",
        "all_300_outputs_and_recipes_verified",
        "canonical_job_order_exact",
        "manifest_enumeration_is_nonsemantic",
        "reused_children_byte_exact",
    }
    integer_gates = {
        "network_requests": 0,
        "owned_transient_residue_count": 0,
        "paused_workspace_checkpoint_count": 37,
    }
    for key, expected in integer_gates.items():
        assert evidence["gate_results"][key] == expected
    assert all(
        value is True
        for key, value in evidence["gate_results"].items()
        if key not in integer_gates
    )
    assert evidence["outer_replay"] == {
        "forward_report_bytes": 4866,
        "forward_report_sha256": "ceb2710c202582c36cc739c478508485552732ef6387d3705ca041a1840a9847",
        "reports_byte_exact": True,
        "reverse_report_bytes": 4866,
        "reverse_report_sha256": "ceb2710c202582c36cc739c478508485552732ef6387d3705ca041a1840a9847",
    }
    revision = evidence["implementation_commit"]
    for path, expected in evidence["source_blobs"].items():
        assert _git_object(revision, path) == expected
    assert evidence["metrics"]["resumed_reused_jobs"] == 37
    assert evidence["metrics"]["resumed_new_jobs"] == 63
    assert evidence["test_evidence"]["adjacent_product_tests_passed"] == 44
