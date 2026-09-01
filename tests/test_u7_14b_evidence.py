from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY_RESULT.json"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_14b_evidence_is_an_exact_bounded_pass() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    assert report["status"] == (
        "PASS_PRIVATE_U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY"
    )
    assert report["formal_reports"] == {
        "byte_exact": True,
        "bytes_each": 2803,
        "forward_path": "outputs/eval/u7_14b_desktop_canonical_scratch_boundary/formal_forward.json",
        "order_independent_scientific_payload_exact": True,
        "owned_external_control_residue": 0,
        "owned_tmp_residue": 0,
        "reverse_path": "outputs/eval/u7_14b_desktop_canonical_scratch_boundary/formal_reverse.json",
        "scientific_identity": "e8eb81f238775265776022afe474ae9aab7a4225fd660afd56220d4359cbfee0",
        "sha256": "041390e2c7124c4d6e49c3bb456fed29b221aa20bebac05a9a8522d6e8106150",
    }
    assert all(report["gates"].values())
    assert report["claim"]["mode"] == "film-inspired / Look Approximation"
    assert report["claim"]["canonical_repo_storage_only"] is True
    assert report["claim"]["calibrated_stock_response"] is False
    assert report["claim"]["physical_film_reproduction"] is False
    assert report["science"]["forbidden_pre_rejection_imports"] == []


def test_u7_14b_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["commits"]["formal_execution"]
    for path, expected in report["bindings"].items():
        assert hashlib.sha256(_git_blob(commit, path)).hexdigest() == expected
