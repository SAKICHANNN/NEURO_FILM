from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from src.inference import (
    PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID,
    load_render_profile,
    summarize_render_profile_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"
CONFIG = ROOT / "configs" / "u2_6a_profile_evidence_inspection_v1.json"
SCRIPT = ROOT / "scripts" / "inspect_render_profile.py"


def _run(profile: Path = PROFILE) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(profile)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_summary_is_exact_non_mutating_and_matches_frozen_contract() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    profile = load_render_profile(PROFILE, root=ROOT)
    original = copy.deepcopy(profile)
    summary = summarize_render_profile_evidence(profile)

    assert list(summary) == contract["required_keys"]
    assert summary["schema_id"] == PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID
    for key, value in contract["expected_safe_rich"].items():
        assert summary[key] == value
    assert summary["claim_ceiling"] == profile["evidence"]["claim_ceiling"]
    summary["claim_ceiling"] = "mutated caller copy"
    assert profile == original


def test_cli_output_is_byte_deterministic_and_exact() -> None:
    first = _run()
    second = _run()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.encode("utf-8") == second.stdout.encode("utf-8")
    assert first.stderr == second.stderr == ""
    assert json.loads(first.stdout) == summarize_render_profile_evidence(
        load_render_profile(PROFILE, root=ROOT)
    )


def test_cli_fails_closed_on_evidence_escalation(tmp_path: Path) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    profile["evidence"]["calibrated_reference_allowed"] = True
    invalid = tmp_path / "escalated.json"
    invalid.write_text(json.dumps(profile), encoding="utf-8")

    result = _run(invalid)
    assert result.returncode == 2
    assert "held-out S3 evidence" in result.stderr
    assert result.stdout == ""


def test_cli_fails_closed_on_asset_hash_mismatch(tmp_path: Path) -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    profile["assets"][0]["sha256"] = "0" * 64
    invalid = tmp_path / "tampered.json"
    invalid.write_text(json.dumps(profile), encoding="utf-8")

    result = _run(invalid)
    assert result.returncode == 2
    assert "asset hash mismatch" in result.stderr
    assert result.stdout == ""
