from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
from referencing import Registry, Resource

from src.color_match.canonical import canonical_sha256
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_conformance import (
    CORE_CONSUMER_CONFORMANCE_RESULT_SCHEMA_ID,
    core_consumer_conformance_result_to_json,
    verify_core_consumer_conformance_bundle,
)
from src.color_match.core_contracts import make_capabilities


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "configs" / "schemas"
FIXTURE = ROOT / "configs" / (
    "reference_match_core_consumer_conformance_v1.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(bundle: dict) -> dict:
    payload = deepcopy(bundle)
    payload.pop("fixture_id")
    bundle["fixture_id"] = canonical_sha256(payload)
    return bundle


def _schema_registry() -> Registry:
    registry = Registry()
    for name in (
        "reference_core_capabilities_v1.schema.json",
        "reference_core_match_view_v1.schema.json",
    ):
        schema = _json(SCHEMAS / name)
        Draft202012Validator.check_schema(schema)
        registry = registry.with_resource(
            schema["$id"],
            Resource.from_contents(schema),
        )
        registry = registry.with_resource(
            (
                "https://neuro-film.local/schemas/"
                f"{name}"
            ),
            Resource.from_contents(schema),
        )
    return registry


def test_frozen_consumer_fixture_passes_and_is_deterministic() -> None:
    first = verify_core_consumer_conformance_bundle(FIXTURE)
    second = verify_core_consumer_conformance_bundle(FIXTURE)
    assert first == second
    assert first.passed
    assert len(first.case_results) == 2
    assert all(case.passed for case in first.case_results)
    assert all(
        case.expected_view_id == case.actual_view_id
        for case in first.case_results
    )
    assert (
        core_consumer_conformance_result_to_json(first)
        == core_consumer_conformance_result_to_json(second)
    )


def test_frozen_bundle_and_result_match_strict_schemas() -> None:
    bundle = _json(FIXTURE)
    bundle_schema = _json(
        SCHEMAS / "reference_core_consumer_conformance_v1.schema.json"
    )
    result_schema = _json(
        SCHEMAS
        / "reference_core_consumer_conformance_result_v1.schema.json"
    )
    Draft202012Validator.check_schema(bundle_schema)
    Draft202012Validator.check_schema(result_schema)
    Draft202012Validator(
        bundle_schema,
        registry=_schema_registry(),
    ).validate(bundle)
    result = verify_core_consumer_conformance_bundle(bundle)
    encoded = core_consumer_conformance_result_to_json(result)
    decoded = json.loads(encoded)
    Draft202012Validator(result_schema).validate(decoded)
    assert decoded["schema_id"] == (
        CORE_CONSUMER_CONFORMANCE_RESULT_SCHEMA_ID
    )


def test_fixture_is_synthetic_and_contains_no_producer_identity() -> None:
    encoded = FIXTURE.read_text(encoding="utf-8").lower()
    assert '"producer_role": "synthetic-consumer-fixture"' in encoded
    assert "dpct" not in encoded
    assert "zhuise" not in encoded


def test_tampered_expected_view_becomes_a_failed_case() -> None:
    bundle = _json(FIXTURE)
    bundle["cases"][0]["expected_match_view"]["primaries"] = "rec2020"
    _rehash(bundle)
    result = verify_core_consumer_conformance_bundle(bundle)
    assert not result.passed
    assert result.case_results[0].reasons == ("match-view-payload",)
    assert result.case_results[1].passed


def test_missing_advertised_profile_fails_only_affected_case() -> None:
    bundle = _json(FIXTURE)
    old = bundle["capabilities"]
    capabilities = make_capabilities(
        producer_id=old["producer_id"],
        producer_version=old["producer_version"],
        producer_build_sha256=old["producer_build_sha256"],
        contract_schema_ids=old["contract_schema_ids"],
        supported_profile_ids=[
            "neuro-film.display-relative-linear-srgb-d65.v1"
        ],
        supported_algorithm_ids=old["supported_algorithm_ids"],
        feature_flags=old["feature_flags"],
        determinism=old["determinism"],
    )
    bundle["capabilities"] = capabilities.to_dict()
    _rehash(bundle)
    result = verify_core_consumer_conformance_bundle(bundle)
    assert result.case_results[0].passed
    assert not result.case_results[1].passed
    assert result.case_results[1].reasons[0].startswith(
        "contract:prepared match profile is not advertised"
    )


def test_tampered_fixture_identity_fails_closed() -> None:
    bundle = _json(FIXTURE)
    bundle["cases"][0]["bit_depth_in"] = 16
    with pytest.raises(
        ReferenceMatchContractError,
        match="fixture_id does not match",
    ):
        verify_core_consumer_conformance_bundle(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda bundle: bundle.update(
                producer_role="external-producer"
            ),
            "producer role must be synthetic",
        ),
        (
            lambda bundle: bundle.update(cases=bundle["cases"][:1]),
            "at least two cases",
        ),
        (
            lambda bundle: bundle.update(surprise=True),
            "keys mismatch",
        ),
    ],
)
def test_structural_mutations_fail_closed(mutation, message) -> None:
    bundle = _json(FIXTURE)
    mutation(bundle)
    _rehash(bundle)
    with pytest.raises(ReferenceMatchContractError, match=message):
        verify_core_consumer_conformance_bundle(bundle)
