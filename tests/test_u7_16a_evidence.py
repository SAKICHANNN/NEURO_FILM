from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_16A_DESKTOP_EXACT_DETAIL_INSPECTION_RESULT.json"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def test_u7_16a_evidence_binds_exact_product_result() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    assert payload["status"] == ("PASS_PRIVATE_U7_16A_DESKTOP_EXACT_DETAIL_INSPECTION")
    reports = payload["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"] == 4864
    assert hashlib.sha256(forward.read_bytes()).hexdigest() == reports["sha256"]
    formal = json.loads(forward.read_text("utf-8"))
    assert formal["scientific_identity"] == reports["scientific_identity"]
    assert all(formal["gates"].values())

    for path, identity in payload["bindings"].items():
        value = _git_bytes(path)
        assert len(value) == identity["bytes"]
        assert hashlib.sha256(value).hexdigest() == identity["sha256"]
        assert (
            subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
            ).strip()
            == identity["git_blob"]
        )

    observation = payload["primary_observation"]
    assert (
        observation["temporary_full_output_sha256"]
        == (observation["ordinary_full_output_sha256"])
    )
    assert observation["detail_rgb_sha256"] == observation["oracle_rgb_sha256"]
    assert observation["temporary_workspace_residue"] == 0
    assert observation["scratch_residue"] == 0
    assert payload["claim"]["calibrated_stock_response"] is False
    assert payload["claim"]["physical_film_reproduction"] is False
    assert payload["claim"]["sample_code_or_display_calibrated"] is False
