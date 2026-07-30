from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v17 import (
    CLAIM_CEILING,
    PAYLOAD_SCOPE,
    REQUIRED_CONTRACT_SCHEMA_PATHS,
    SCHEMA_ID,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v17.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_match_main_integration_manifest_v17.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v16.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_v17_rebuilds_and_preserves_v16() -> None:
    frozen = _manifest()
    rebuilt = build_manifest(
        repo=ROOT,
        payload_commit=frozen["payload_commit"],
        base_commit=frozen["base_commit"],
        main_commit=frozen["main_commit"],
    )
    assert rebuilt == frozen
    assert encode_json(rebuilt) == MANIFEST.read_text(encoding="utf-8")
    assert encode_json(build_schema()) == SCHEMA.read_text(encoding="utf-8")
    validate_manifest(repo=ROOT, manifest=frozen)
    assert frozen["schema_id"] == SCHEMA_ID
    assert frozen["payload_scope"] == PAYLOAD_SCOPE
    assert frozen["claim_ceiling"] == CLAIM_CEILING
    assert frozen["overlap_paths"] == []
    assert frozen["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR.read_bytes()
    ).hexdigest()


def test_v17_binds_capability_neutral_invocation_contract() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    schemas = [row["path"] for row in manifest["contract_schemas"]]
    assert schemas == list(REQUIRED_CONTRACT_SCHEMA_PATHS)
    assert schemas[-1] == (
        "configs/schemas/reference_dpct_invocation_profile_v2.schema.json"
    )
    files = {row["path"] for row in manifest["payload_files"]}
    assert "src/color_match/dpct_invocation_profile.py" in files
    assert "tests/test_color_match_dpct_invocation_v2.py" in files
    assert (
        "configs/reference_match_main_integration_manifest_v16.json"
        in files
    )
    assert (
        "configs/reference_match_main_integration_manifest_v17.json"
        not in files
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P108"),
        lambda value: value.update(claim_ceiling="merged"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["contract_schemas"].pop(),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v17_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


def test_v17_schema_rejects_before_git_access(tmp_path: Path) -> None:
    manifest = deepcopy(_manifest())
    manifest["payload_files"][0]["path"] = "../escape"
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )
