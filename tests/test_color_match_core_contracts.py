from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.color_match.core_contracts import (
    CAPABILITIES_SCHEMA_ID,
    DIAGNOSTICS_SCHEMA_ID,
    MATCH_PROFILE_ABSOLUTE_REC2020,
    MATCH_PROFILE_ABSOLUTE_XYZ,
    MATCH_PROFILE_DISPLAY_SRGB,
    MATCH_VIEW_SCHEMA_ID,
    TRANSFORM_BUNDLE_SCHEMA_ID,
    CapabilitiesV1,
    DiagnosticsV1,
    capabilities_from_json,
    capabilities_to_json,
    diagnostics_from_json,
    diagnostics_to_json,
    make_capabilities,
    make_match_view,
    make_transform_bundle,
    match_view_from_json,
    match_view_to_json,
    transform_bundle_from_json,
    transform_bundle_to_json,
    validate_core_binding,
    validate_diagnostics,
    validate_match_view,
    validate_transform_bundle,
)
from src.color_match.contracts import ReferenceMatchContractError


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "configs" / "schemas"
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_D = "4" * 64
CONTRACT_ID = "zhuise.transform-bundle.v1"
ALGORITHM_ID = "zhuise.dpct.v1"


def _schema(name: str) -> dict:
    payload = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(payload)
    return payload


def _view(*, reference: bool = False):
    return make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=SHA_B if reference else SHA_A,
        shape=(5, 7, 3),
        render_bridge_id="neuro-film.working-image-direct.v1",
        provenance_fingerprint=SHA_D if reference else SHA_C,
    )


def _capabilities() -> CapabilitiesV1:
    return make_capabilities(
        producer_id="zhuise-core",
        producer_version="0.1.0",
        producer_build_sha256=SHA_D,
        contract_schema_ids=[CONTRACT_ID],
        supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
        supported_algorithm_ids=[ALGORITHM_ID],
        feature_flags=["opaque-payload", "source-bound-fit"],
    )


def _transform(source, reference):
    return make_transform_bundle(
        producer_id="zhuise-core",
        producer_version="0.1.0",
        producer_build_sha256=SHA_D,
        contract_schema_id=CONTRACT_ID,
        algorithm_id=ALGORITHM_ID,
        algorithm_version="1.0.0",
        intent_id=SHA_C,
        source_view_id=source.view_id,
        reference_view_id=reference.view_id,
        payload_schema_id="zhuise.dpct-transform.v1",
        payload_sha256=SHA_A,
        capability_requirements=["opaque-payload", "source-bound-fit"],
    )


def _diagnostics(transform, *, status: str = "ok") -> DiagnosticsV1:
    return DiagnosticsV1(
        schema_id=DIAGNOSTICS_SCHEMA_ID,
        transform_id=transform.transform_id,
        status=status,
        fallback_reason=None if status == "ok" else "unsupported-rail",
        finite=True,
        out_of_gamut_fraction=0.01,
        clipping_fraction=0.0,
        projected_fraction=0.01,
        confidence=0.8,
        timing_ms=1.25,
        backend_id="cpu-reference",
        backend_fingerprint=SHA_B,
        warnings=(),
    )


@pytest.mark.parametrize(
    ("schema_name", "builder"),
    [
        (
            "reference_core_match_view_v1.schema.json",
            lambda: _view().to_dict(),
        ),
        (
            "reference_core_transform_bundle_v1.schema.json",
            lambda: _transform(_view(), _view(reference=True)).to_dict(),
        ),
        (
            "reference_core_diagnostics_v1.schema.json",
            lambda: _diagnostics(
                _transform(_view(), _view(reference=True))
            ).to_dict(),
        ),
        (
            "reference_core_capabilities_v1.schema.json",
            lambda: _capabilities().to_dict(),
        ),
    ],
)
def test_core_payloads_match_strict_schemas(schema_name, builder) -> None:
    validator = Draft202012Validator(_schema(schema_name))
    payload = builder()
    validator.validate(payload)
    payload["surprise"] = True
    assert any(
        error.validator == "additionalProperties"
        for error in validator.iter_errors(payload)
    )


def test_match_view_roundtrip_and_identity_are_deterministic() -> None:
    first = _view()
    second = _view()
    assert first == second
    assert first.schema_id == MATCH_VIEW_SCHEMA_ID
    assert first.view_id == second.view_id
    assert match_view_from_json(match_view_to_json(first)) == first


def test_match_view_profile_semantics_and_absolute_white_are_strict() -> None:
    relative = _view()
    with pytest.raises(
        ReferenceMatchContractError,
        match="profile colour semantics",
    ):
        validate_match_view(replace(relative, primaries="rec2020"))
    with pytest.raises(
        ReferenceMatchContractError,
        match="reference_white_nits",
    ):
        make_match_view(
            profile_id=MATCH_PROFILE_ABSOLUTE_XYZ,
            pixel_sha256=SHA_A,
            shape=(1, 1, 3),
            render_bridge_id="trusted.absolute-bridge.v1",
            provenance_fingerprint=SHA_B,
        )
    absolute_rec2020 = make_match_view(
        profile_id=MATCH_PROFILE_ABSOLUTE_REC2020,
        pixel_sha256=SHA_A,
        shape=(1, 1, 3),
        render_bridge_id="trusted.absolute-rec2020-bridge.v1",
        provenance_fingerprint=SHA_B,
        reference_white_nits=203.0,
    )
    assert absolute_rec2020.domain == "display-absolute-linear"
    assert absolute_rec2020.primaries == "rec2020"
    assert absolute_rec2020.white_point == "D65"
    assert absolute_rec2020.reference_white_nits == 203.0
    Draft202012Validator(
        _schema("reference_core_match_view_v1.schema.json")
    ).validate(absolute_rec2020.to_dict())


def test_transform_is_source_bound_and_roundtrips() -> None:
    source = _view()
    reference = _view(reference=True)
    transform = _transform(source, reference)
    assert transform.schema_id == TRANSFORM_BUNDLE_SCHEMA_ID
    assert transform.binding_scope == "source-bound"
    assert (
        transform_bundle_from_json(
            transform_bundle_to_json(transform)
        )
        == transform
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="source-bound",
    ):
        validate_transform_bundle(
            replace(transform, binding_scope="shared-operator")
        )


def test_capabilities_are_canonical_and_roundtrip() -> None:
    capabilities = _capabilities()
    assert capabilities.schema_id == CAPABILITIES_SCHEMA_ID
    assert (
        capabilities_from_json(capabilities_to_json(capabilities))
        == capabilities
    )
    assert capabilities.feature_flags == (
        "opaque-payload",
        "source-bound-fit",
    )


def test_diagnostics_status_invariants_and_roundtrip() -> None:
    transform = _transform(_view(), _view(reference=True))
    diagnostics = _diagnostics(transform)
    assert diagnostics_from_json(diagnostics_to_json(diagnostics)) == diagnostics
    with pytest.raises(
        ReferenceMatchContractError,
        match="fallback reason",
    ):
        validate_diagnostics(
            replace(diagnostics, fallback_reason="not-allowed")
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="fallback_reason",
    ):
        validate_diagnostics(
            replace(
                diagnostics,
                status="unsupported",
                fallback_reason=None,
            )
        )


def test_core_binding_accepts_exact_producer_view_and_capabilities() -> None:
    source = _view()
    reference = _view(reference=True)
    transform = _transform(source, reference)
    validate_core_binding(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=_capabilities(),
        diagnostics=_diagnostics(transform),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source", "source_view_id"),
        ("reference", "reference_view_id"),
        ("producer", "producer/build"),
        ("algorithm", "algorithm is not advertised"),
        ("capability", "missing required core capabilities"),
        ("diagnostics", "diagnostics transform_id"),
    ],
)
def test_core_binding_fails_closed_on_every_identity_boundary(
    mutation: str,
    message: str,
) -> None:
    source = _view()
    reference = _view(reference=True)
    transform = _transform(source, reference)
    capabilities = _capabilities()
    diagnostics = _diagnostics(transform)
    if mutation == "source":
        other_source = make_match_view(
            profile_id=MATCH_PROFILE_DISPLAY_SRGB,
            pixel_sha256=SHA_D,
            shape=(5, 7, 3),
            render_bridge_id="neuro-film.working-image-direct.v1",
            provenance_fingerprint=SHA_C,
        )
        transform = _transform(other_source, reference)
        diagnostics = _diagnostics(transform)
    elif mutation == "reference":
        other_reference = make_match_view(
            profile_id=MATCH_PROFILE_DISPLAY_SRGB,
            pixel_sha256=SHA_C,
            shape=(5, 7, 3),
            render_bridge_id="neuro-film.working-image-direct.v1",
            provenance_fingerprint=SHA_D,
        )
        transform = _transform(source, other_reference)
        diagnostics = _diagnostics(transform)
    elif mutation == "producer":
        capabilities = make_capabilities(
            producer_id="zhuise-core",
            producer_version="0.1.0",
            producer_build_sha256=SHA_A,
            contract_schema_ids=[CONTRACT_ID],
            supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
            supported_algorithm_ids=[ALGORITHM_ID],
            feature_flags=["opaque-payload", "source-bound-fit"],
        )
    elif mutation == "algorithm":
        capabilities = make_capabilities(
            producer_id="zhuise-core",
            producer_version="0.1.0",
            producer_build_sha256=SHA_D,
            contract_schema_ids=[CONTRACT_ID],
            supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
            supported_algorithm_ids=["other.algorithm"],
            feature_flags=["opaque-payload", "source-bound-fit"],
        )
    elif mutation == "capability":
        capabilities = make_capabilities(
            producer_id="zhuise-core",
            producer_version="0.1.0",
            producer_build_sha256=SHA_D,
            contract_schema_ids=[CONTRACT_ID],
            supported_profile_ids=[MATCH_PROFILE_DISPLAY_SRGB],
            supported_algorithm_ids=[ALGORITHM_ID],
            feature_flags=["opaque-payload"],
        )
    elif mutation == "diagnostics":
        diagnostics = replace(diagnostics, transform_id=SHA_A)
    with pytest.raises(ReferenceMatchContractError, match=message):
        validate_core_binding(
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
        )


@pytest.mark.parametrize(
    ("encoded", "parser"),
    [
        (lambda: match_view_to_json(_view()), match_view_from_json),
        (
            lambda: transform_bundle_to_json(
                _transform(_view(), _view(reference=True))
            ),
            transform_bundle_from_json,
        ),
        (
            lambda: diagnostics_to_json(
                _diagnostics(
                    _transform(_view(), _view(reference=True))
                )
            ),
            diagnostics_from_json,
        ),
        (
            lambda: capabilities_to_json(_capabilities()),
            capabilities_from_json,
        ),
    ],
)
def test_core_parsers_reject_unknown_fields(encoded, parser) -> None:
    payload = json.loads(encoded())
    payload["surprise"] = True
    with pytest.raises(ReferenceMatchContractError, match="keys mismatch"):
        parser(json.dumps(payload))
