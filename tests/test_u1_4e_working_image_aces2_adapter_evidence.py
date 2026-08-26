from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_4E_WORKING_IMAGE_ACES2_ADAPTER_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha256(commit: str, path: str) -> str:
    value = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(value).hexdigest()


def test_u1_4e_evidence_is_bound_and_passes_only_private_adapter() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_PRIVATE_WORKING_IMAGE_ACES2_ADAPTER"
    assert all(payload["gates"].values())
    assert payload["metrics"]["maximum_adapter_direct_absolute_error"] == 0.0
    assert payload["metrics"]["maximum_source_scalar_packed_absolute_error"] == 0.0
    assert payload["execution"]["all_four_report_sha256_exact"] is True
    formal_lock_commit = payload["bindings"]["formal_lock_commit"]
    for key in ("contract", "implementation", "runner", "config"):
        path = payload["bindings"][f"{key}_path"]
        assert _git_sha256(formal_lock_commit, path) == payload["bindings"][
            f"{key}_sha256"
        ]
    report_path = ROOT / payload["bindings"]["formal_report_path"]
    assert _sha256(report_path) == payload["bindings"]["formal_report_sha256"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stable_evidence_id"] == payload["bindings"]["stable_evidence_id"]
    assert "no ACES certification" in payload["claim_ceiling"]
    assert "no ACES certification" in report["claim_ceiling"]
