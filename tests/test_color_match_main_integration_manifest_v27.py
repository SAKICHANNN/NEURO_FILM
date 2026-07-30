from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.build_reference_match_integration_manifest import IntegrationManifestError
from scripts.build_reference_match_integration_manifest_v27 import (
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
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v27.json"
SCHEMA = (
    ROOT / "configs/schemas/reference_match_main_integration_manifest_v27.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v26.json"


def frozen():
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def blob(path):
    files = {row["path"]: row["blob"] for row in frozen()["payload_files"]}
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]], cwd=ROOT
    ).decode()


def test_v27_rebuilds_and_binds_alias_coherent_lock():
    value = frozen()
    assert (
        build_manifest(
            repo=ROOT,
            payload_commit=value["payload_commit"],
            base_commit=value["base_commit"],
            main_commit=value["main_commit"],
        )
        == value
    )
    assert encode_json(build_schema()) == SCHEMA.read_text(encoding="utf-8")
    validate_manifest(repo=ROOT, manifest=value)
    validate_manifest_schema(value)
    assert value["schema_id"] == SCHEMA_ID
    assert value["payload_scope"] == PAYLOAD_SCOPE
    assert not value["overlap_paths"]
    assert value["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR.read_bytes()
    ).hexdigest()
    lock = blob("src/color_match/transaction_lock.py")
    assert "os.path.realpath" in lock
    assert "TransactionPathV2" in lock


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(payload_scope="bad"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value.update(extra=True),
    ],
)
def test_v27_tamper(mutation):
    value = deepcopy(frozen())
    mutation(value)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=value)
