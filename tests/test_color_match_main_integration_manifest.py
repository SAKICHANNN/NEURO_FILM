from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    CLAIM_CEILING,
    REQUIRED_PUBLIC_EXPORTS,
    SCHEMA_ID,
    IntegrationManifestError,
    build_manifest,
    encode_manifest,
    validate_manifest,
    validate_manifest_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v1.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_manifest_rebuilds_byte_identically() -> None:
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


def test_manifest_has_exact_review_ceiling_and_zero_overlap() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    assert manifest["schema_id"] == SCHEMA_ID
    assert manifest["claim_ceiling"] == CLAIM_CEILING
    assert manifest["overlap_paths"] == []
    assert manifest["payload_changed_file_count"] == len(
        manifest["payload_files"]
    )
    assert manifest["required_public_exports"] == list(
        REQUIRED_PUBLIC_EXPORTS
    )
    assert len(manifest["shared_schemas"]) == 9
    assert all(
        row["path"].startswith("configs/schemas/reference_shared")
        for row in manifest["shared_schemas"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(claim_ceiling="merged"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value["shared_schemas"][0].update(sha256="0" * 64),
        lambda value: value.update(unexpected=True),
    ],
)
def test_manifest_tamper_fails_closed(mutate) -> None:
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
        lambda value: value.update(overlap_paths=["src/color_match/a.py"]),
        lambda value: value.update(payload_changed_file_count=-1),
    ],
)
def test_schema_rejects_malformed_manifest_before_git_access(
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


def test_schema_unavailable_fails_before_git_access(tmp_path: Path) -> None:
    with pytest.raises(
        IntegrationManifestError,
        match="schema unavailable",
    ):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=_manifest(),
            schema_path=tmp_path / "missing.schema.json",
        )


def test_wrong_base_or_pre_shared_payload_fails_closed() -> None:
    manifest = _manifest()
    with pytest.raises(IntegrationManifestError, match="merge base"):
        build_manifest(
            repo=ROOT,
            payload_commit=manifest["payload_commit"],
            base_commit=manifest["main_commit"],
            main_commit=manifest["main_commit"],
        )
    with pytest.raises(IntegrationManifestError, match="missing public"):
        build_manifest(
            repo=ROOT,
            payload_commit=manifest["base_commit"],
            base_commit=manifest["base_commit"],
            main_commit=manifest["main_commit"],
        )
