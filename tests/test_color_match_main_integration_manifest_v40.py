import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v34 import (
    V34_REQUIRED_CONTRACT_SCHEMA_PATHS,
    V34_REQUIRED_PUBLIC_EXPORTS,
)
from scripts.build_reference_match_integration_manifest_v40 import (
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
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v40.json"
SCHEMA = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v40.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v39.json"


def frozen() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def blob(path: str) -> str:
    files = {row["path"]: row["blob"] for row in frozen()["payload_files"]}
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]],
        cwd=ROOT,
    ).decode()


def test_v40_rebuilds_and_binds_post_v39_product_evidence() -> None:
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
    assert value["required_public_exports"] == list(V34_REQUIRED_PUBLIC_EXPORTS)
    assert [row["path"] for row in value["contract_schemas"]] == list(
        V34_REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    assert value["supersedes_manifest"]["sha256"] == hashlib.sha256(
        PRIOR.read_bytes()
    ).hexdigest()
    for path in (
        "configs/reference_match_target_scale_memory_decision_v1.json",
        "configs/reference_match_sdr_file_format_matrix_decision_v1.json",
        "configs/reference_match_rec2020_sdr_file_matrix_decision_v1.json",
        "configs/reference_match_output_capability_execution_decision_v1.json",
    ):
        assert '"automatic_pass": true' in blob(path)
    assert "del rendered" in blob("src/color_match/files.py")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(payload_scope="bad"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value.update(extra=True),
    ],
)
def test_v40_tamper(mutation) -> None:
    value = deepcopy(frozen())
    mutation(value)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=value)
