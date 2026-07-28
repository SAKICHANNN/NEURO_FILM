from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_reference_match_integration_manifest import (
    IntegrationManifestError,
)
from scripts.build_reference_match_integration_manifest_v16 import (
    CLAIM_CEILING,
    PAYLOAD_SCOPE,
    SCHEMA_ID,
    build_manifest,
    build_schema,
    encode_json,
    validate_manifest,
    validate_manifest_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/reference_match_main_integration_manifest_v16.json"
)
SCHEMA = (
    ROOT
    / "configs/schemas/"
    "reference_match_main_integration_manifest_v16.schema.json"
)
PRIOR = ROOT / "configs/reference_match_main_integration_manifest_v15.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_v16_rebuilds_and_preserves_v15() -> None:
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


def test_v16_binds_android_product_chain_runtime_sources() -> None:
    manifest = _manifest()
    validate_manifest_schema(manifest)
    files = {row["path"]: row["blob"] for row in manifest["payload_files"]}
    assert files[
        "runtime/android_srgb_quantizer_testlab/test/jni/"
        "nf_srgb_quantizer_testlab.c"
    ] == "eeaa08eb93d6b705490900a9760988e64b803159"
    assert files[
        "scripts/run_android_srgb_quantizer_emulator_v1.py"
    ] == "06428696bbbc775e8779cb8cd3aa8c407687023e"
    assert files[
        "native/reference_canonical_core.c"
    ] == "dca602fb50cddaa65bb6ab315599afceb908192b"
    assert (
        "configs/reference_match_main_integration_manifest_v15.json"
        in files
    )
    assert (
        "configs/reference_match_main_integration_manifest_v16.json"
        not in files
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(payload_scope="P1-P105"),
        lambda value: value.update(claim_ceiling="merged"),
        lambda value: value["payload_files"][0].update(blob="0" * 40),
        lambda value: value["supersedes_manifest"].update(
            sha256="0" * 64
        ),
        lambda value: value.update(unexpected=True),
    ],
)
def test_v16_tamper_fails_closed(mutate) -> None:
    manifest = deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(
        IntegrationManifestError,
        match="schema violation|differs from commit-derived",
    ):
        validate_manifest(repo=ROOT, manifest=manifest)


def test_v16_schema_rejects_before_git_access(tmp_path: Path) -> None:
    manifest = deepcopy(_manifest())
    manifest["payload_files"][0]["path"] = "../escape"
    with pytest.raises(IntegrationManifestError, match="schema violation"):
        validate_manifest(
            repo=tmp_path / "not-a-repository",
            manifest=manifest,
        )
