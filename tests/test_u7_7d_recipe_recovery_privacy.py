from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_7d_recipe_recovery_privacy import (
    _member_paths_relative,
    privacy_findings,
)


def _contract() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "configs/u7_7d_recipe_recovery_privacy_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_privacy_findings_reject_absolute_input_and_output_paths_without_retaining_values() -> (
    None
):
    recipe = {
        "input": {"path": r"P:\neuro_film_storage\data\private.jpg"},
        "output": {"path": r"C:\Users\example\result.png"},
    }
    findings = privacy_findings(recipe, _contract())
    assert {row["field"] for row in findings} == {"/input/path", "/output/path"}
    assert all("private.jpg" not in json.dumps(row) for row in findings)
    assert {row["kind"] for row in findings} >= {
        "windows_absolute_path",
        "forbidden_local_identifier",
        "input_path_present",
        "output_path_present",
    }


def test_privacy_findings_accept_portable_recipe_fields() -> None:
    recipe = {
        "input": {"sha256": "a" * 64},
        "output": {"sha256": "b" * 64},
        "profile": {"path": "configs/render_profiles/safe_rich_v1.json"},
    }
    assert privacy_findings(recipe, _contract()) == []


def test_bundle_member_paths_must_be_normalized_relative() -> None:
    assert _member_paths_relative(["manifest.json", "payload/configs/a.json"])
    assert not _member_paths_relative(["../escape.json"])
    assert not _member_paths_relative([r"payload\a.json"])
    assert not _member_paths_relative(["/absolute.json"])


def test_contract_binds_three_exact_recipes() -> None:
    contract = _contract()
    assert len(contract["inputs"]["recipes"]) == 3
    assert set(contract["inputs"]["recipe_sha256"]) == {
        "velvia_50",
        "portra_400",
        "ektar_100",
    }
    assert contract["audit"]["maximum_pixel_decodes"] == 0
