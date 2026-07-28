from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v24 import (
    CLAIM_CEILING,
    PAYLOAD_SCOPE,
    SCHEMA_ID,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)
from src.color_match.strict_json import strict_json_loads


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v24.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_match_main_integration_manifest_v24.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v23.json"


def _manifest() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def _payload_bytes(path: str) -> bytes:
    files = {
        row["path"]: row["blob"] for row in _manifest()["payload_files"]
    }
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]],
        cwd=ROOT,
    )


def test_frozen_v24_rebuilds_and_supersedes_v23() -> None:
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


def test_v24_binds_original_temp_root_no_follow_policy() -> None:
    validate_manifest_schema(_manifest())
    runtime_source = _payload_bytes(
        "src/color_match/shared_runtime_staging_transaction.py"
    ).decode("utf-8")
    lock_body = runtime_source.split(
        "def _target_transaction_lock(",
        maxsplit=1,
    )[1].split(
        "def _require_absent_destination(",
        maxsplit=1,
    )[0]
    assert "Path(tempfile.gettempdir()).resolve" not in lock_body
    assert (
        "_reject_reparse_components(\n"
        "        Path(tempfile.gettempdir()),"
    ) in lock_body
    assert "temporary root must be a directory" in lock_body
    payload_paths = {
        row["path"] for row in _manifest()["payload_files"]
    }
    assert (
        "tests/test_color_match_shared_runtime_staging_transaction.py"
        in payload_paths
    )
    assert (
        "configs/reference_match_main_integration_manifest_v23.json"
        in payload_paths
    )
    assert (
        "configs/reference_match_main_integration_manifest_v24.json"
        not in payload_paths
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P129"),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value["contract_schemas"].pop(),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v24_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


def test_v24_schema_rejects_before_git_access(tmp_path: Path) -> None:
    manifest = deepcopy(_manifest())
    manifest["payload_files"][0]["path"] = "../escape"
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )
