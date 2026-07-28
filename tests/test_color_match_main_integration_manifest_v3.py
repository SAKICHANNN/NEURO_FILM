from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v3 import (
    CLAIM_CEILING,
    PAYLOAD_SCOPE,
    REQUIRED_CONTRACT_SCHEMA_PATHS,
    REQUIRED_PUBLIC_EXPORTS,
    SCHEMA_ID,
    build_manifest,
    encode_manifest,
    validate_manifest,
    validate_manifest_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v3.json"
)
PRIOR_MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v2.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_v3_manifest_rebuilds_byte_identically() -> None:
    frozen = _manifest()
    rebuilt = build_manifest(
        repo=ROOT,
        payload_commit=frozen["payload_commit"],
        base_commit=frozen["base_commit"],
        main_commit=frozen["main_commit"],
    )
    assert rebuilt == frozen
    assert encode_manifest(rebuilt) == MANIFEST.read_text(encoding="utf-8")
    validate_manifest(repo=ROOT, manifest=frozen)


def test_v3_adds_p65_but_preserves_p64_exactly() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    assert manifest["schema_id"] == SCHEMA_ID
    assert manifest["payload_scope"] == PAYLOAD_SCOPE
    assert manifest["claim_ceiling"] == CLAIM_CEILING
    assert manifest["overlap_paths"] == []
    assert manifest["payload_changed_file_count"] == len(
        manifest["payload_files"]
    )
    assert manifest["required_public_exports"] == list(
        REQUIRED_PUBLIC_EXPORTS
    )
    assert [row["path"] for row in manifest["contract_schemas"]] == list(
        REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    assert manifest["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR_MANIFEST.read_bytes()
    ).hexdigest()
    payload_paths = {row["path"] for row in manifest["payload_files"]}
    assert (
        "configs/reference_match_main_integration_manifest_v2.json"
        in payload_paths
    )
    assert (
        "configs/reference_match_main_integration_manifest_v3.json"
        not in payload_paths
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P66"),
        lambda value: value.update(claim_ceiling="merged"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value["contract_schemas"][4].update(
            sha256="0" * 64
        ),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v3_manifest_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("schema_id"),
        lambda value: value["payload_files"][0].update(mode="100600"),
        lambda value: value["payload_files"][0].update(path="../escape"),
        lambda value: value.update(
            overlap_paths=["src/color_match/a.py"]
        ),
        lambda value: value.update(payload_changed_file_count=-1),
        lambda value: value["contract_schemas"].pop(),
        lambda value: value["contract_schemas"].reverse(),
        lambda value: value["supersedes_manifest"].update(
            path="../p64.json"
        ),
    ],
)
def test_v3_schema_rejects_before_git_access(
    tmp_path: Path,
    mutate,
) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )


def test_v3_schema_unavailable_fails_before_git_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        IntegrationManifestError,
        match="schema unavailable",
    ):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=_manifest(),
            schema_path=tmp_path / "missing.schema.json",
        )


def test_v3_wrong_base_or_pre_p65_payload_fails_closed() -> None:
    manifest = _manifest()
    with pytest.raises(IntegrationManifestError, match="merge base"):
        build_manifest(
            repo=ROOT,
            payload_commit=manifest["payload_commit"],
            base_commit=manifest["main_commit"],
            main_commit=manifest["main_commit"],
        )
    with pytest.raises(
        IntegrationManifestError,
        match="missing public|contract schema",
    ):
        build_manifest(
            repo=ROOT,
            payload_commit=manifest["base_commit"],
            base_commit=manifest["base_commit"],
            main_commit=manifest["main_commit"],
        )
