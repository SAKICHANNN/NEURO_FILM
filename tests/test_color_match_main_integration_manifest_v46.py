import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v46 import (
    PAYLOAD_SCOPE,
    SCHEMA_ID,
    V46_REQUIRED_CONTRACT_SCHEMA_PATHS,
    V46_REQUIRED_PUBLIC_EXPORTS,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)
from src.color_match.strict_json import strict_json_loads

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v46.json"
SCHEMA = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v46.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v45.json"


def frozen() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def blob(path: str) -> str:
    files = {row["path"]: row["blob"] for row in frozen()["payload_files"]}
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]],
        cwd=ROOT,
    ).decode()


def test_v46_rebuilds_corrected_post_merge_payload() -> None:
    value = frozen()
    assert build_manifest(
        repo=ROOT,
        payload_commit=value["payload_commit"],
        base_commit=value["base_commit"],
        main_commit=value["main_commit"],
    ) == value
    assert encode_json(build_schema()) == SCHEMA.read_text(encoding="utf-8")
    validate_manifest(repo=ROOT, manifest=value)
    validate_manifest_schema(value)
    assert value["schema_id"] == SCHEMA_ID
    assert value["payload_scope"] == PAYLOAD_SCOPE
    assert not value["overlap_paths"]
    assert value["required_public_exports"] == list(
        V46_REQUIRED_PUBLIC_EXPORTS
    )
    assert [row["path"] for row in value["contract_schemas"]] == list(
        V46_REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    assert value["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR.read_bytes()
    ).hexdigest()
    paths = {row["path"] for row in value["payload_files"]}
    assert "src/color_match/files.py" in paths
    assert "configs/reference_match_main_integration_manifest_v45.json" in paths
    implementation = blob("src/color_match/files.py")
    assert "_validate_run_path_topology" in implementation
    assert "_OwnedDirectory" in implementation
    assert "reference-match-directory-owner.v1" in implementation
    handoff = blob(
        "docs/drpt/REFERENCE_COLOR_MATCH_MAIN_INTEGRATION_HANDOFF.md"
    )
    assert "V45 and payload `d4d817d6` must not be merged" in handoff


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(payload_scope="bad"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value.update(extra=True),
    ],
)
def test_v46_tamper(mutation) -> None:
    value = deepcopy(frozen())
    mutation(value)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=value)
