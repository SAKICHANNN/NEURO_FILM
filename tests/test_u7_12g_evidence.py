from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12G_DESKTOP_BATCH_OUTPUT_FORMAT_RESULT.json"
EVIDENCE_SHA256 = "59e62b1ce628f4a8999ebddb387d6a73c6bb9cf984cb33dc6a5f9898aed9aa38"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_12g_evidence_is_exact_pass() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == "PASS_PRIVATE_U7_12G_DESKTOP_BATCH_OUTPUT_FORMATS"
    assert len(report["case_results"]) == 9
    assert all(row["passed"] for row in report["case_results"].values())
    assert all(report["gates"].values())
    assert report["claim"] == {
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "formats": ["png16", "tiff16", "jpeg8"],
        "hdr_or_wide_gamut_publication": False,
        "look_math_changed": False,
        "mode": "film-inspired",
        "non_png_receipt_schema": "kmcfm.desktop-single-look-batch.v2",
        "one_format_per_atomic_batch": True,
        "physical_film_reproduction": False,
        "png16_receipt_schema": "kmcfm.desktop-single-look-batch.v1",
        "private_windows_desktop_batch_export_only": True,
        "public_release": False,
        "recovery_format": "png16-only",
        "renderer_changed": False,
        "stock_distinguishability": False,
    }


def test_u7_12g_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["source_commit"]
    for path, expected in report["bindings"].items():
        assert hashlib.sha256(_git_blob(commit, path)).hexdigest() == expected
