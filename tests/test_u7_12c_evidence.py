from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_12C_DESKTOP_DISPLAY_NATIVE_PREVIEW_RESULT.json"
EVIDENCE_SHA256 = "a68f6d0ebba6d74341798a8828fcc10da9c9ed164078ae6f63bf92ec6161c2ef"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_12c_evidence_is_exact_pass() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == "PASS_PRIVATE_U7_12C_DESKTOP_DISPLAY_NATIVE_PREVIEW"
    assert report["scientific_identity"] == (
        "sha256:404aab50eed0aff39571d706e72b2182a57ca5b05cc7fb17eeb048df4374fce4"
    )
    assert all(report["gates"].values())
    assert report["formal_reports"]["order_independent_scientific_payload_exact"]
    assert not report["formal_reports"]["full_reports_byte_exact"]
    assert report["fidelity"]["maximum_rgb_rmse"] <= 0.03
    assert report["fidelity"]["maximum_rgb_absolute_error_p95"] <= 0.08
    assert report["fidelity"]["maximum_new_boundary_fraction"] <= 0.001
    assert report["claim"] == {
        "arbitrary_input_or_device_quality": False,
        "calibrated_stock_response": False,
        "cross_platform_gui": False,
        "evidence_grade": "look-approximation",
        "hdr_or_wide_gamut_publication": False,
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "private_windows_python_preview_mechanics_only": True,
        "public_release_or_installer": False,
        "stock_distinguishability": False,
    }


def test_u7_12c_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    assert report["bindings"]["commit"] == report["source_commit"]
    for path, expected in report["bindings"]["files"].items():
        assert hashlib.sha256(_git_blob(report["source_commit"], path)).hexdigest() == expected
