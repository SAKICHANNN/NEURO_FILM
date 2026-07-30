from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v20 import (
    CLAIM_CEILING,
    PAYLOAD_SCOPE,
    REQUIRED_CONTRACT_SCHEMA_PATHS,
    REQUIRED_PUBLIC_EXPORTS,
    SCHEMA_ID,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v20.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_match_main_integration_manifest_v20.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v19.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_v20_rebuilds_and_preserves_v19() -> None:
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


def test_v20_binds_batch_limit_and_full_invocation_commit() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    assert manifest["required_public_exports"] == list(
        REQUIRED_PUBLIC_EXPORTS
    )
    assert [row["path"] for row in manifest["contract_schemas"]] == list(
        REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    files = {row["path"]: row["blob"] for row in manifest["payload_files"]}
    expected_schema_hashes = {
        "configs/schemas/reference_dpct_batch_resolution_v1.schema.json": (
            "ddf448c25b7f4027612bc24d68f8efc2d254368b45c43eed4601c2275a98f655"
        ),
        "configs/schemas/reference_shared_operator_batch_v1.schema.json": (
            "2c2e0e9a9513dd41b0b841426d2548a9589c4de0d657267e4d7b54745618bc21"
        ),
    }
    for path, expected in expected_schema_hashes.items():
        raw = subprocess.check_output(
            ["git", "cat-file", "blob", files[path]],
            cwd=ROOT,
        )
        assert hashlib.sha256(raw).hexdigest() == expected
    assert "src/color_match/batch_limits.py" in files
    assert "configs/reference_match_dpct_invocation_v1.json" in files
    assert (
        "configs/reference_match_main_integration_manifest_v19.json"
        in files
    )
    assert (
        "configs/reference_match_main_integration_manifest_v20.json"
        not in files
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P117"),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value["contract_schemas"].pop(),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v20_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


def test_v20_schema_rejects_before_git_access(tmp_path: Path) -> None:
    manifest = deepcopy(_manifest())
    manifest["payload_files"][0]["path"] = "../escape"
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )
