from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v22 import (
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
from src.color_match.strict_json import strict_json_loads


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v22.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_match_main_integration_manifest_v22.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v21.json"


def _manifest() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_v22_rebuilds_and_preserves_v21() -> None:
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


def test_v22_binds_all_product_batch_resource_limits() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    assert manifest["required_public_exports"] == list(
        REQUIRED_PUBLIC_EXPORTS
    )
    assert [row["path"] for row in manifest["contract_schemas"]] == list(
        REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    files = {row["path"]: row["blob"] for row in manifest["payload_files"]}
    required_paths = {
        "src/color_match/batch_limits.py",
        "src/color_match/files.py",
        "src/color_match/render.py",
        "src/color_match/safety.py",
        "src/color_match/staging_io.py",
        "src/color_match/shared_runtime_staging_transaction.py",
        "tests/test_color_match_batch_limit_consistency.py",
        "tests/test_color_match_staging_io_limits.py",
        "configs/reference_match_main_integration_manifest_v21.json",
    }
    assert required_paths <= files.keys()
    assert (
        "configs/reference_match_main_integration_manifest_v22.json"
        not in files
    )
    expected_schema_hashes = {
        "configs/schemas/reference_match_report_v1.schema.json": (
            "6fa335df0b1b82a77cc5a59465060fcc3282b1dc9c9d9bb6413baef77028016f"
        ),
        "configs/schemas/reference_match_replay_report_v1.schema.json": (
            "84b54e861d23ea5ad1476017851f8a1434ee32dca467b974dcab3bce4930affe"
        ),
    }
    for path, expected in expected_schema_hashes.items():
        raw = subprocess.check_output(
            ["git", "cat-file", "blob", files[path]],
            cwd=ROOT,
        )
        assert hashlib.sha256(raw).hexdigest() == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P124"),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value["contract_schemas"].pop(),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v22_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


def test_v22_schema_rejects_before_git_access(tmp_path: Path) -> None:
    manifest = deepcopy(_manifest())
    manifest["payload_files"][0]["path"] = "../escape"
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )
