from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
import numpy as np
import pytest

from src.color_match import (
    ReferenceMatchContractError,
    canonical_sha256,
    decode_f32be_hex,
    encode_f32be_hex,
    load_portable_conformance_bundle,
    portable_conformance_result_to_json,
    verify_portable_conformance_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "configs" / "reference_match_portable_conformance_v1.json"
SCHEMAS = ROOT / "configs" / "schemas"


def _payload() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _refresh_fixture_id(payload: dict) -> None:
    identity = dict(payload)
    identity.pop("fixture_id")
    payload["fixture_id"] = canonical_sha256(identity)


def test_frozen_portable_bundle_passes_and_repeats_exactly() -> None:
    payload = load_portable_conformance_bundle(BUNDLE)
    first = verify_portable_conformance_bundle(payload)
    second = verify_portable_conformance_bundle(payload)
    assert first.passed is True
    assert len(first.case_results) == 2
    assert all(case.passed for case in first.case_results)
    assert portable_conformance_result_to_json(
        first
    ) == portable_conformance_result_to_json(second)


def test_bundle_and_result_match_language_neutral_schemas() -> None:
    bundle_schema = json.loads(
        (
            SCHEMAS / "reference_match_portable_conformance_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    recipe_schema = json.loads(
        (SCHEMAS / "reference_look_recipe_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    result_schema = json.loads(
        (
            SCHEMAS
            / "reference_match_portable_conformance_result_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(bundle_schema)
    Draft202012Validator.check_schema(result_schema)
    registry = Registry().with_resource(
        recipe_schema["$id"],
        Resource.from_contents(recipe_schema),
    )
    Draft202012Validator(
        bundle_schema,
        registry=registry,
    ).validate(_payload())
    result = verify_portable_conformance_bundle(_payload())
    Draft202012Validator(result_schema).validate(
        json.loads(portable_conformance_result_to_json(result))
    )


def test_ieee754_float_payload_roundtrips_exact_bits() -> None:
    values = np.asarray(
        [[[0.0, -0.0, 1.0], [0.1, 0.5, 0.9]]],
        dtype=np.float32,
    )
    encoded = encode_f32be_hex(values)
    decoded = decode_f32be_hex(encoded, values.shape, label="fixture")
    assert decoded.tobytes(order="C") == values.tobytes(order="C")


def test_bundle_rejects_identity_drift_before_execution() -> None:
    payload = _payload()
    payload["cases"][0]["expected"]["output_f32be_hex"] = (
        "0" + payload["cases"][0]["expected"]["output_f32be_hex"][1:]
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="fixture_id does not match",
    ):
        verify_portable_conformance_bundle(payload)


def test_reidentified_output_drift_fails_numeric_gate() -> None:
    payload = deepcopy(_payload())
    encoded = payload["cases"][0]["expected"]["output_f32be_hex"]
    expected = decode_f32be_hex(
        encoded,
        (3, 4, 3),
        label="expected",
    )
    expected[0, 0, 0] += np.float32(0.05)
    payload["cases"][0]["expected"]["output_f32be_hex"] = encode_f32be_hex(
        expected
    )
    _refresh_fixture_id(payload)
    result = verify_portable_conformance_bundle(payload)
    assert result.passed is False
    assert result.case_results[0].passed is False
    assert result.case_results[0].linear_rgb_max_abs > 0.04
    assert result.case_results[1].passed is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["cases"][0]["reference"].update(
                {"pixels_f32be_hex": "ABCDEF"}
            ),
            "lowercase hexadecimal",
        ),
        (
            lambda payload: payload["cases"][0].update(
                {"unexpected": True}
            ),
            "keys mismatch",
        ),
        (
            lambda payload: payload["tolerances"].update(
                {"linear_rgb_max_abs": 0.0}
            ),
            "positive number",
        ),
    ],
)
def test_bundle_fails_closed_on_invalid_wire_values(
    mutation,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    mutation(payload)
    _refresh_fixture_id(payload)
    with pytest.raises(ReferenceMatchContractError, match=message):
        verify_portable_conformance_bundle(payload)


def test_cli_emits_schema_valid_repeatable_report(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "verify_reference_match_portable_conformance.py"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--bundle",
                str(BUNDLE),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    assert first.read_bytes() == second.read_bytes()
