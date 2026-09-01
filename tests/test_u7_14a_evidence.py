from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_14A_DESKTOP_INPUT_BASIS_PREVIEW_RESULT.json"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_14a_evidence_is_an_exact_bounded_pass() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    assert report["status"] == "PASS_PRIVATE_U7_14A_DESKTOP_INPUT_BASIS_PREVIEW"
    assert report["scientific_identity"] == (
        "abd40a408426168be0520660d934b97bd01b65528547d2d9f6586f68d3fec9f1"
    )
    assert all(report["gates"].values())
    assert report["formal_reports"] == {
        "forward_bytes": 7292,
        "forward_sha256": "f765c9657f8fac9e97e17862c5bba14c5de3d0376594905566b2fe5c0b35d446",
        "full_reports_byte_exact": True,
        "order_independent_scientific_payload_exact": True,
        "reverse_bytes": 7292,
        "reverse_sha256": "f765c9657f8fac9e97e17862c5bba14c5de3d0376594905566b2fe5c0b35d446",
    }
    assert report["excluded_pre_evidence"]["status"].startswith("FAIL_CLOSED_")
    assert report["claim"]["mode"] == "film-inspired / Look Approximation"
    assert report["claim"]["calibrated_stock_response"] is False
    assert report["claim"]["calibrated_camera_rendering"] is False
    assert report["claim"]["physical_film_reproduction"] is False
    assert all(
        row["input_preview_sha256"] == row["oracle_sha256"]
        and row["candidate_decode_calls"] == 1
        for row in report["sources"]
    )


def test_u7_14a_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    for path, expected in report["bindings"].items():
        assert (
            hashlib.sha256(_git_blob(report["source_commit"], path)).hexdigest()
            == expected
        )
