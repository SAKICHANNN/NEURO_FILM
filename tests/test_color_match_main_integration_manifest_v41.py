import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v41 import (
    PAYLOAD_SCOPE,
    SCHEMA_ID,
    V41_REQUIRED_CONTRACT_SCHEMA_PATHS,
    V41_REQUIRED_PUBLIC_EXPORTS,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)
from src.color_match.strict_json import strict_json_loads


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v41.json"
SCHEMA = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v41.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v40.json"


def frozen() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def blob(path: str) -> str:
    files = {row["path"]: row["blob"] for row in frozen()["payload_files"]}
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]],
        cwd=ROOT,
    ).decode()


def test_v41_rebuilds_and_binds_p168_metadata_enforcement() -> None:
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
        V41_REQUIRED_PUBLIC_EXPORTS
    )
    assert [row["path"] for row in value["contract_schemas"]] == list(
        V41_REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    assert value["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR.read_bytes()
    ).hexdigest()
    assert "attest_reference_file_output_metadata" in blob(
        "src/color_match/files.py"
    )
    assert "source_metadata_copy" in blob(
        "configs/schemas/reference_file_output_metadata_policy_v1.schema.json"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(payload_scope="bad"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["required_public_exports"].pop(),
        lambda value: value.update(extra=True),
    ],
)
def test_v41_tamper(mutation) -> None:
    value = deepcopy(frozen())
    mutation(value)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=value)
