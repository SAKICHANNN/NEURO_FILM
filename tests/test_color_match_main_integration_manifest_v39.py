from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v34 import (
    V34_REQUIRED_CONTRACT_SCHEMA_PATHS,
    V34_REQUIRED_PUBLIC_EXPORTS,
)
from scripts.build_reference_match_integration_manifest_v39 import (
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
MANIFEST = ROOT / "configs/reference_match_main_integration_manifest_v39.json"
SCHEMA = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v39.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v38.json"


def frozen() -> dict:
    return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))


def blob(path: str) -> str:
    files = {row["path"]: row["blob"] for row in frozen()["payload_files"]}
    return subprocess.check_output(
        ["git", "cat-file", "blob", files[path]],
        cwd=ROOT,
    ).decode()


def test_v39_rebuilds_and_binds_exact_memory_kernels() -> None:
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
    assert "_REFERENCE_STYLE_VERTICAL_HALO = 5" in blob(
        "src/color_match/render.py"
    )
    assert "REFERENCE_MATCH_ROW_CHUNK = 128" in blob(
        "src/color_match/row_kernels.py"
    )
    assert (
        '"automatic_pass": true'
        in blob("configs/reference_match_halo_style_memory_decision_v1.json")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(payload_scope="bad"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value.update(extra=True),
    ],
)
def test_v39_tamper(mutation) -> None:
    value = deepcopy(frozen())
    mutation(value)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=value)
