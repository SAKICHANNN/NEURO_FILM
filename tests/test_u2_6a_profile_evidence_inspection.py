from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from src.inference import (
    LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID,
    PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID,
    load_render_profile,
    summarize_legacy_style_evidence_inventory,
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


def test_legacy_style_inventory_exposes_names_without_stock_claims() -> None:
    profile = load_render_profile(PROFILE, root=ROOT)
    inventory = summarize_legacy_style_evidence_inventory(profile)

    assert inventory["schema_id"] == LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID
    assert inventory["style_count"] == len(profile["style_parameters"])
    assert [row["style_id"] for row in inventory["styles"]] == sorted(
        profile["style_parameters"]
    )
    assert {row["evidence_role"] for row in inventory["styles"]} == {
        "legacy_named_look_proxy"
    }
    assert {row["film_stock_id"] for row in inventory["styles"]} == {None}
    assert not any(
        row["stock_specific_operator_admitted"]
        or row["target_film_closeness_established"]
        or row["stock_distinguishability_established"]
        or row["calibrated_reference_allowed"]
        for row in inventory["styles"]
    )


def test_cli_legacy_style_inventory_is_exact_and_read_only() -> None:
    command = [sys.executable, str(SCRIPT), "--legacy-style-inventory"]
    first = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    second = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout.encode("utf-8") == second.stdout.encode("utf-8")
    assert json.loads(first.stdout) == summarize_legacy_style_evidence_inventory(
        load_render_profile(PROFILE, root=ROOT)
    )
