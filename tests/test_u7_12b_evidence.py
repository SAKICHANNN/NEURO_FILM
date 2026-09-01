from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12B_DESKTOP_SINGLE_PHOTO_OUTPUT_FORMAT_RESULT.json"
EVIDENCE_SHA256 = "281ff125c9f019f83753c0a98ab2d7047232c0d61c7a656b8e1064b704d81488"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_12b_evidence_is_exact_pass() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == (
        "PASS_PRIVATE_U7_12B_DESKTOP_SINGLE_PHOTO_OUTPUT_FORMATS"
    )
    assert len(report["case_results"]) == 17
    assert all(row["passed"] for row in report["case_results"].values())
    assert all(report["gates"].values())
    assert report["claim"] == {
        "batch_format": "png16",
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "formats": ["png16", "tiff16", "jpeg8"],
        "hdr_or_wide_gamut_publication": False,
        "look_math_changed": False,
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "private_native_single_photo_sdr_export_only": True,
        "public_release": False,
        "renderer_changed": False,
        "stock_distinguishability": False,
    }


def test_u7_12b_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["source_commit"]
    for path, expected in report["bindings"].items():
        assert hashlib.sha256(_git_blob(commit, path)).hexdigest() == expected
