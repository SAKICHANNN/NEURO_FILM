from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from src.color_match import (
    MAX_REFERENCE_MATCH_BATCH_SOURCES,
    REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING,
    REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID,
    reference_file_output_capabilities_payload,
    reference_file_output_metadata_policy_payload,
    reference_file_supported_input_rails,
    reference_match_product_capabilities_payload,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_product_capabilities_v1.schema.json"
)
SCRIPT = ROOT / "scripts" / "match_reference_color.py"


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


def test_product_capabilities_are_strict_and_composed_from_public_contracts() -> None:
    payload = reference_match_product_capabilities_payload()
    _validator().validate(payload)
    assert payload["schema_id"] == REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID
    assert (
        payload["claim_ceiling"]
        == REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING
    )
    assert payload["batch"]["maximum_sources"] == (
        MAX_REFERENCE_MATCH_BATCH_SOURCES
    )
    assert payload["input"]["accepted_decoded_rails"] == list(
        reference_file_supported_input_rails()
    )
    assert payload["output"] == reference_file_output_capabilities_payload()
    assert payload["output_metadata"] == (
        reference_file_output_metadata_policy_payload()
    )
    assert payload["delivery"] == {
        "guard_policy_id": "reference-render-guard.v2",
        "algorithm_status": "not-promoted",
        "default_action": "identity-fallback",
        "research_override": "explicit-opt-in-only-not-product",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("batch", "maximum_sources"), 65),
        (("input", "render_revalidates_inputs"), False),
        (("delivery", "algorithm_status"), "promoted"),
        (("output_metadata", "source_metadata_copy"), "all"),
    ],
)
def test_product_capabilities_schema_rejects_claim_drift(
    path: tuple[str, str],
    value: object,
) -> None:
    payload = reference_match_product_capabilities_payload()
    payload[path[0]][path[1]] = value
    with pytest.raises(jsonschema.ValidationError):
        _validator().validate(payload)


def test_product_capabilities_returns_fresh_nested_payloads() -> None:
    first = reference_match_product_capabilities_payload()
    first["output"]["capabilities"][0]["extensions"].append(".avif")
    first["input"]["accepted_decoded_rails"][0]["working_space"] = "acescg"
    second = reference_match_product_capabilities_payload()
    _validator().validate(second)
    assert ".avif" not in second["output"]["capabilities"][0]["extensions"]
    assert second["input"]["accepted_decoded_rails"][0] == {
        "working_space": "linear_rec2020",
        "transfer_state": "display_linear",
    }


def test_cli_reports_unified_product_capabilities() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--product-capabilities"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload == reference_match_product_capabilities_payload()
    _validator().validate(payload)


def test_cli_product_capabilities_rejects_render_arguments(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--product-capabilities",
            "--output",
            str(tmp_path / "output.png"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "--product-capabilities cannot be combined" in completed.stderr
