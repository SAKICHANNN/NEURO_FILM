from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from src.color_match import (
    FROZEN_GATE_POLICY_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    SUCCESSOR_DECLARATION_SCHEMA,
    SUCCESSOR_POLICY_ID,
    SUPPORTED_PROFILE_ID,
    PreparedMatchViewV1,
    PromotionDecision,
    ReferenceMatchContractError,
    authorize_shared_product_staging_v1,
    bind_shared_promotion_v1,
    guard_shared_numeric_batch_v1,
    make_match_view,
    make_shared_apply_numeric_facts_v1,
    make_shared_reference_operator_v1,
    prepare_shared_operator_apply_v1,
    resolve_shared_operator_batch_v1,
    shared_product_authorization_from_json,
    shared_product_authorization_to_json,
    successor_declaration_id_v1,
    validate_shared_product_staging_authorization_v1,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_shared_product_staging_authorization_v1.schema.json"
)
CAPABILITY = "zhuise.rgin.cpu-reference.v1"
PRODUCER_COMMIT = "1" * 40
MODEL = "d" * 64
OPTIONS = "e" * 64
EVIDENCE = "f" * 64


def _prepared(seed: float) -> PreparedMatchViewV1:
    pixels = np.full((2, 3, 3), seed, dtype=np.float32)
    wire = pixels.astype(">f4", copy=False).tobytes()
    descriptor = make_match_view(
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        pixel_sha256=hashlib.sha256(wire).hexdigest(),
        shape=pixels.shape,
        render_bridge_id="neuro-film.shared-product-test.v1",
        provenance_fingerprint=hashlib.sha256(
            str(seed).encode()
        ).hexdigest(),
    )
    pixels.flags.writeable = False
    return PreparedMatchViewV1(descriptor=descriptor, pixels=pixels)


def _declaration(*, product: bool = True) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SUCCESSOR_DECLARATION_SCHEMA,
        "policy_id": SUCCESSOR_POLICY_ID,
        "candidate_id": "zhuise.rgin.shared.v1",
        "producer": {
            "stable_commit": PRODUCER_COMMIT,
            "package_source_commit": "2" * 40,
            "fixture_commit": "3" * 40,
            "package_lock_sha256": "a" * 64,
            "conformance_fixture_sha256": "b" * 64,
        },
        "package": {
            "distribution": "zhuise-research",
            "version": "0.4.0",
            "entrypoint": "zhuise-shared-invoke",
            "wheel_filename": "zhuise_research-0.4.0-py3-none-any.whl",
            "wheel_size_bytes": 110000,
            "wheel_sha256": "c" * 64,
        },
        "wire": {
            "capability_id": CAPABILITY,
            "profile_id": SUPPORTED_PROFILE_ID,
            "lower_compatibility_profile_id": (
                "neuro-film.dpct-consumer.v2"
            ),
            "request_schema": "zhuise.shared-request.v1",
            "request_schema_sha256": "a" * 64,
            "response_schema": "zhuise.shared-response.v1",
            "response_schema_sha256": "b" * 64,
        },
        "semantics": {
            "fit_semantics": "reference-only-shared",
            "batch_transform_policy": "shared-bundle",
            "deterministic": True,
            "hidden_state": "none",
        },
        "rights": {
            "evaluation_allowed": True,
            "commercial_use_allowed": product,
            "redistribution_allowed": product,
            "evidence_id": "rights-review-v1",
        },
        "runtime_evidence": {
            "windows_x64": product,
            "macos_arm64": product,
            "ios_arm64": product,
            "android_arm64": product,
        },
        "product_evidence": {
            "gate_policy_id": FROZEN_GATE_POLICY_ID,
            "stable_evidence_id": EVIDENCE,
            "a1_passed": True,
            "a4_passed": True,
            "a5_passed": True,
            "blind_aesthetic_passed": True,
        },
    }
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _resign(value: dict[str, object]) -> dict[str, object]:
    value.pop("declaration_id", None)
    value["declaration_id"] = successor_declaration_id_v1(value)
    return value


def _pipeline(
    *,
    product: bool = True,
    promotion: PromotionDecision | None = None,
    second_clip: float = 0.01,
    model: str = MODEL,
):
    reference = _prepared(0.2)
    sources = (_prepared(0.3), _prepared(0.4))
    declaration = _declaration(product=product)
    operator = make_shared_reference_operator_v1(
        reference=reference,
        compatibility_profile_id="neuro-film.dpct-consumer.v2",
        capability_id=CAPABILITY,
        producer_commit=PRODUCER_COMMIT,
        producer_bundle_id="sha256:" + "9" * 64,
        producer_reference_view_id="sha256:" + "8" * 64,
        model_fingerprint=model,
        options_sha256=OPTIONS,
    )
    applies = tuple(
        prepare_shared_operator_apply_v1(
            operator=operator,
            source_index=index,
            source=source,
            producer_source_view_id=(
                "sha256:" + str(index + 1) * 64
            ),
            producer_apply_result_id=(
                "sha256:" + str(index + 3) * 64
            ),
            diagnostics_id="sha256:" + str(index + 5) * 64,
            output_pixels=np.full(
                source.descriptor.shape,
                0.31 + index * 0.1,
                dtype=np.float32,
            ),
        )
        for index, source in enumerate(sources)
    )
    batch = resolve_shared_operator_batch_v1(
        operator=operator,
        reference=reference,
        sources=sources,
        applies=applies,
    )
    facts = tuple(
        make_shared_apply_numeric_facts_v1(
            prepared=prepared,
            producer_diagnostics_id=prepared.receipt.diagnostics_id,
            all_finite=True,
            output_minimum=float(np.min(prepared.pixels)),
            output_maximum=float(np.max(prepared.pixels)),
            out_of_gamut_fraction=(
                second_clip if index == 1 else 0.01
            ),
            clipping_fraction=second_clip if index == 1 else 0.01,
            projected_fraction=0.0,
        )
        for index, prepared in enumerate(applies)
    )
    numeric = guard_shared_numeric_batch_v1(
        batch=batch,
        sources=sources,
        applies=applies,
        facts=facts,
    )
    promotion = promotion or PromotionDecision("promoted", ())
    binding = bind_shared_promotion_v1(
        operator=operator,
        declaration=declaration,
        stable_evidence_id=EVIDENCE,
        promotion=promotion,
    )
    return declaration, operator, batch, numeric, binding


def test_three_locks_authorize_staging_only() -> None:
    declaration, _operator, batch, numeric, binding = _pipeline()
    first = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    second = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    assert first == second
    assert first.state == "authorized-for-staging"
    assert all(row.action == "authorized-for-staging" for row in first.sources)
    encoded = shared_product_authorization_to_json(first)
    assert "applied" not in encoded
    assert "delivered" not in encoded
    assert "pixels" not in encoded
    assert shared_product_authorization_from_json(encoded) == first
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(encoded))


def test_declaration_pass_flags_without_promotion_cannot_authorize() -> None:
    promotion = PromotionDecision(
        "eligible-for-visual-review",
        ("blind-aesthetic-review-required",),
    )
    declaration, _operator, batch, numeric, binding = _pipeline(
        promotion=promotion
    )
    result = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    assert result.admission_product_ready
    assert result.state == "identity-fallback"
    assert all(
        row.reasons == ("promotion-not-promoted",)
        for row in result.sources
    )


def test_product_admission_and_numeric_fallback_are_atomic() -> None:
    declaration, _operator, batch, numeric, binding = _pipeline(
        product=False
    )
    result = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    assert result.state == "identity-fallback"
    assert "successor-product-not-ready" in result.sources[0].reasons

    declaration, _operator, batch, numeric, binding = _pipeline(
        second_clip=0.06
    )
    result = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    assert result.state == "identity-fallback"
    assert all(row.action == "identity-fallback" for row in result.sources)
    assert result.sources[0].reasons == ("numeric-batch-fallback",)
    assert result.sources[1].reasons == (
        "numeric-batch-fallback",
        "numeric:clipping-fraction",
    )


@pytest.mark.parametrize(
    "mutation",
    ["producer", "capability", "profile", "compatibility", "evidence"],
)
def test_declaration_substitution_fails_closed(mutation: str) -> None:
    declaration, operator, _batch, _numeric, _binding = _pipeline()
    changed = deepcopy(declaration)
    if mutation == "producer":
        changed["producer"]["stable_commit"] = "7" * 40
    elif mutation == "capability":
        changed["wire"]["capability_id"] = "zhuise.foreign.v1"
    elif mutation == "profile":
        changed["wire"]["profile_id"] = "zhuise.foreign-profile.v1"
    elif mutation == "compatibility":
        changed["wire"]["lower_compatibility_profile_id"] = (
            "neuro-film.foreign.v1"
        )
    else:
        changed["product_evidence"]["stable_evidence_id"] = "7" * 64
    _resign(changed)
    with pytest.raises(ReferenceMatchContractError):
        bind_shared_promotion_v1(
            operator=operator,
            declaration=changed,
            stable_evidence_id=EVIDENCE,
            promotion=PromotionDecision("promoted", ()),
        )


def test_foreign_model_binding_and_guard_fail_closed() -> None:
    declaration, _operator, batch, numeric, _binding = _pipeline()
    (
        foreign_declaration,
        _foreign_operator,
        _foreign_batch,
        _foreign_numeric,
        foreign_binding,
    ) = _pipeline(model="7" * 64)
    assert foreign_declaration["declaration_id"] == declaration[
        "declaration_id"
    ]
    with pytest.raises(
        ReferenceMatchContractError,
        match="promotion binding does not bind",
    ):
        authorize_shared_product_staging_v1(
            batch=batch,
            numeric_guard=numeric,
            declaration=declaration,
            promotion_binding=foreign_binding,
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="numeric guard does not bind",
    ):
        authorize_shared_product_staging_v1(
            batch=batch,
            numeric_guard=_foreign_numeric,
            declaration=declaration,
            promotion_binding=foreign_binding,
        )


def test_authorization_identity_and_embedded_state_mutation_fail_closed() -> None:
    declaration, _operator, batch, numeric, binding = _pipeline()
    result = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    with pytest.raises(
        ReferenceMatchContractError, match="identity mismatch"
    ):
        validate_shared_product_staging_authorization_v1(
            replace(result, authorization_id="0" * 64)
        )
    with pytest.raises(ReferenceMatchContractError):
        validate_shared_product_staging_authorization_v1(
            replace(result, admission_product_ready=False)
        )
    with pytest.raises(ReferenceMatchContractError):
        validate_shared_product_staging_authorization_v1(
            replace(
                result,
                sources=(
                    replace(
                        result.sources[0],
                        numeric_accepted_for_transaction=False,
                    ),
                    result.sources[1],
                ),
            )
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="not canonical",
    ):
        validate_shared_product_staging_authorization_v1(
            replace(
                result,
                successor_declaration_json=json.dumps(
                    json.loads(result.successor_declaration_json),
                    indent=2,
                    sort_keys=True,
                ),
            )
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="must be distinct",
    ):
        validate_shared_product_staging_authorization_v1(
            replace(
                result,
                sources=(
                    result.sources[0],
                    replace(
                        result.sources[1],
                        source_view_id=result.sources[0].source_view_id,
                    ),
                ),
            )
        )


def test_unknown_json_fields_and_promotion_evidence_mismatch_fail_closed() -> None:
    declaration, operator, batch, numeric, binding = _pipeline()
    result = authorize_shared_product_staging_v1(
        batch=batch,
        numeric_guard=numeric,
        declaration=declaration,
        promotion_binding=binding,
    )
    payload = json.loads(shared_product_authorization_to_json(result))
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError):
        shared_product_authorization_from_json(json.dumps(payload))
    with pytest.raises(
        ReferenceMatchContractError,
        match="promotion evidence does not bind",
    ):
        bind_shared_promotion_v1(
            operator=operator,
            declaration=declaration,
            stable_evidence_id="7" * 64,
            promotion=PromotionDecision("promoted", ()),
        )
