"""Consumer binding for one reference-only operator applied to N sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_contracts import (
    MatchViewV1,
    make_match_view,
    match_view_from_json,
    validate_match_view,
)


SHARED_OPERATOR_SCHEMA_ID = (
    "neuro-film.reference-shared-operator.v1"
)
SHARED_APPLY_RECEIPT_SCHEMA_ID = (
    "neuro-film.reference-shared-apply-receipt.v1"
)
SHARED_BATCH_SCHEMA_ID = "neuro-film.reference-shared-batch.v1"
SHARED_BATCH_POLICY_ID = "neuro-film.reference-only-shared-batch.v1"
SHARED_OUTPUT_BRIDGE_ID = "neuro-film.shared-operator-output.v1"
SHARED_CLAIM_CEILING = (
    "candidate-only-awaiting-a1-a4-a5-and-numeric-product-guards"
)
_FIT_SEMANTICS = "reference-only-shared"
_BATCH_POLICY = "shared-bundle"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OPERATOR_KEYS = {
    "schema_id",
    "operator_id",
    "fit_semantics",
    "batch_transform_policy",
    "compatibility_profile_id",
    "capability_id",
    "producer_commit",
    "producer_bundle_id",
    "reference_view_id",
    "producer_reference_view_id",
    "profile_id",
    "model_fingerprint",
    "options_sha256",
}
_RECEIPT_KEYS = {
    "schema_id",
    "receipt_id",
    "claim_ceiling",
    "operator_id",
    "source_index",
    "source_view_id",
    "producer_source_view_id",
    "producer_apply_result_id",
    "diagnostics_id",
    "output_view",
}
_BATCH_KEYS = {
    "schema_id",
    "batch_id",
    "policy_id",
    "claim_ceiling",
    "operator",
    "source_count",
    "sources",
}


@dataclass(frozen=True)
class SharedReferenceOperatorV1:
    schema_id: str
    operator_id: str
    fit_semantics: str
    batch_transform_policy: str
    compatibility_profile_id: str
    capability_id: str
    producer_commit: str
    producer_bundle_id: str
    reference_view_id: str
    producer_reference_view_id: str
    profile_id: str
    model_fingerprint: str
    options_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharedOperatorApplyReceiptV1:
    schema_id: str
    receipt_id: str
    claim_ceiling: str
    operator_id: str
    source_index: int
    source_view_id: str
    producer_source_view_id: str
    producer_apply_result_id: str
    diagnostics_id: str
    output_view: MatchViewV1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_view"] = self.output_view.to_dict()
        return payload


@dataclass(frozen=True)
class PreparedSharedOperatorApplyV1:
    receipt: SharedOperatorApplyReceiptV1
    pixels: np.ndarray


@dataclass(frozen=True)
class SharedOperatorBatchV1:
    schema_id: str
    batch_id: str
    policy_id: str
    claim_ceiling: str
    operator: SharedReferenceOperatorV1
    source_count: int
    sources: tuple[SharedOperatorApplyReceiptV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "batch_id": self.batch_id,
            "policy_id": self.policy_id,
            "claim_ceiling": self.claim_ceiling,
            "operator": self.operator.to_dict(),
            "source_count": self.source_count,
            "sources": [source.to_dict() for source in self.sources],
        }


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the shared-bundle contract"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ReferenceMatchContractError(
            f"{label} must be a bounded non-empty string"
        )
    return value


def _hash(value: Any, label: str) -> str:
    text = _text(value, label)
    if _HEX.fullmatch(text) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return text


def _producer_hash(value: Any, label: str) -> str:
    text = _text(value, label)
    if _PRODUCER_HASH.fullmatch(text) is None:
        raise ReferenceMatchContractError(
            f"{label} must be lowercase sha256:<hex>"
        )
    return text


def _identity_payload(value: Any, identity_key: str) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop(identity_key)
    return payload


def _pixel_sha256(pixels: np.ndarray) -> str:
    encoded = np.asarray(pixels, dtype=">f4", order="C")
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def make_shared_reference_operator_v1(
    *,
    reference: PreparedMatchViewV1,
    compatibility_profile_id: str,
    capability_id: str,
    producer_commit: str,
    producer_bundle_id: str,
    producer_reference_view_id: str,
    model_fingerprint: str,
    options_sha256: str,
) -> SharedReferenceOperatorV1:
    """Bind one opaque producer bundle proven to depend only on reference."""

    validate_prepared_match_view(reference)
    if _COMMIT.fullmatch(producer_commit) is None:
        raise ReferenceMatchContractError(
            "producer_commit must be a full lowercase Git commit"
        )
    provisional = SharedReferenceOperatorV1(
        schema_id=SHARED_OPERATOR_SCHEMA_ID,
        operator_id="0" * 64,
        fit_semantics=_FIT_SEMANTICS,
        batch_transform_policy=_BATCH_POLICY,
        compatibility_profile_id=_text(
            compatibility_profile_id, "compatibility_profile_id"
        ),
        capability_id=_text(capability_id, "capability_id"),
        producer_commit=producer_commit,
        producer_bundle_id=_producer_hash(
            producer_bundle_id, "producer_bundle_id"
        ),
        reference_view_id=reference.descriptor.view_id,
        producer_reference_view_id=_producer_hash(
            producer_reference_view_id,
            "producer_reference_view_id",
        ),
        profile_id=reference.descriptor.profile_id,
        model_fingerprint=_hash(
            model_fingerprint, "model_fingerprint"
        ),
        options_sha256=_hash(options_sha256, "options_sha256"),
    )
    result = replace(
        provisional,
        operator_id=canonical_sha256(
            _identity_payload(provisional, "operator_id")
        ),
    )
    validate_shared_reference_operator_v1(result)
    return result


def validate_shared_reference_operator_v1(
    value: SharedReferenceOperatorV1,
) -> None:
    if not isinstance(value, SharedReferenceOperatorV1):
        raise ReferenceMatchContractError(
            "shared operator must be SharedReferenceOperatorV1"
        )
    if value.schema_id != SHARED_OPERATOR_SCHEMA_ID:
        raise ReferenceMatchContractError("shared operator schema is invalid")
    _hash(value.operator_id, "operator_id")
    if (
        value.fit_semantics != _FIT_SEMANTICS
        or value.batch_transform_policy != _BATCH_POLICY
    ):
        raise ReferenceMatchContractError(
            "shared operator semantics are invalid"
        )
    _text(value.compatibility_profile_id, "compatibility_profile_id")
    _text(value.capability_id, "capability_id")
    if _COMMIT.fullmatch(value.producer_commit) is None:
        raise ReferenceMatchContractError(
            "shared operator producer commit is invalid"
        )
    _producer_hash(value.producer_bundle_id, "producer_bundle_id")
    _hash(value.reference_view_id, "reference_view_id")
    _producer_hash(
        value.producer_reference_view_id,
        "producer_reference_view_id",
    )
    _text(value.profile_id, "profile_id")
    _hash(value.model_fingerprint, "model_fingerprint")
    _hash(value.options_sha256, "options_sha256")
    if value.operator_id != canonical_sha256(
        _identity_payload(value, "operator_id")
    ):
        raise ReferenceMatchContractError(
            "shared operator identity mismatch"
        )


def prepare_shared_operator_apply_v1(
    *,
    operator: SharedReferenceOperatorV1,
    source_index: int,
    source: PreparedMatchViewV1,
    producer_source_view_id: str,
    producer_apply_result_id: str,
    diagnostics_id: str,
    output_pixels: np.ndarray,
) -> PreparedSharedOperatorApplyV1:
    """Copy and bind one exact source application to a shared operator."""

    validate_shared_reference_operator_v1(operator)
    validate_prepared_match_view(source)
    if source.descriptor.profile_id != operator.profile_id:
        raise ReferenceMatchContractError(
            "shared apply source profile differs from operator profile"
        )
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index < 0
        or source_index >= MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "shared apply source_index is invalid"
        )
    pixels = np.array(
        output_pixels,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if tuple(pixels.shape) != source.descriptor.shape:
        raise ReferenceMatchContractError(
            "shared apply output shape differs from source"
        )
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(
            "shared apply output pixels must be finite"
        )
    pixels.flags.writeable = False
    producer_source_id = _producer_hash(
        producer_source_view_id, "producer_source_view_id"
    )
    result_id = _producer_hash(
        producer_apply_result_id, "producer_apply_result_id"
    )
    diagnostic_id = _producer_hash(diagnostics_id, "diagnostics_id")
    provenance = canonical_sha256(
        {
            "schema_id": "neuro-film.shared-operator-output-provenance.v1",
            "operator_id": operator.operator_id,
            "source_index": source_index,
            "source_view_id": source.descriptor.view_id,
            "producer_source_view_id": producer_source_id,
            "producer_apply_result_id": result_id,
            "diagnostics_id": diagnostic_id,
        }
    )
    output_view = make_match_view(
        profile_id=source.descriptor.profile_id,
        pixel_sha256=_pixel_sha256(pixels),
        shape=source.descriptor.shape,
        render_bridge_id=SHARED_OUTPUT_BRIDGE_ID,
        provenance_fingerprint=provenance,
        alpha_mode="absent",
    )
    provisional = SharedOperatorApplyReceiptV1(
        schema_id=SHARED_APPLY_RECEIPT_SCHEMA_ID,
        receipt_id="0" * 64,
        claim_ceiling=SHARED_CLAIM_CEILING,
        operator_id=operator.operator_id,
        source_index=source_index,
        source_view_id=source.descriptor.view_id,
        producer_source_view_id=producer_source_id,
        producer_apply_result_id=result_id,
        diagnostics_id=diagnostic_id,
        output_view=output_view,
    )
    receipt = replace(
        provisional,
        receipt_id=canonical_sha256(
            _identity_payload(provisional, "receipt_id")
        ),
    )
    prepared = PreparedSharedOperatorApplyV1(
        receipt=receipt,
        pixels=pixels,
    )
    validate_prepared_shared_operator_apply_v1(prepared)
    return prepared


def validate_shared_operator_apply_receipt_v1(
    value: SharedOperatorApplyReceiptV1,
) -> None:
    if not isinstance(value, SharedOperatorApplyReceiptV1):
        raise ReferenceMatchContractError(
            "shared apply receipt type is invalid"
        )
    if value.schema_id != SHARED_APPLY_RECEIPT_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "shared apply receipt schema is invalid"
        )
    _hash(value.receipt_id, "receipt_id")
    if value.claim_ceiling != SHARED_CLAIM_CEILING:
        raise ReferenceMatchContractError(
            "shared apply claim ceiling is invalid"
        )
    _hash(value.operator_id, "operator_id")
    if (
        isinstance(value.source_index, bool)
        or not isinstance(value.source_index, int)
        or value.source_index < 0
        or value.source_index >= MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "shared apply source_index is invalid"
        )
    _hash(value.source_view_id, "source_view_id")
    _producer_hash(
        value.producer_source_view_id,
        "producer_source_view_id",
    )
    _producer_hash(
        value.producer_apply_result_id,
        "producer_apply_result_id",
    )
    _producer_hash(value.diagnostics_id, "diagnostics_id")
    validate_match_view(value.output_view)
    if (
        value.output_view.render_bridge_id != SHARED_OUTPUT_BRIDGE_ID
        or value.output_view.alpha_mode != "absent"
    ):
        raise ReferenceMatchContractError(
            "shared apply output view contract is invalid"
        )
    if value.receipt_id != canonical_sha256(
        _identity_payload(value, "receipt_id")
    ):
        raise ReferenceMatchContractError(
            "shared apply receipt identity mismatch"
        )


def validate_prepared_shared_operator_apply_v1(
    value: PreparedSharedOperatorApplyV1,
) -> None:
    if not isinstance(value, PreparedSharedOperatorApplyV1):
        raise ReferenceMatchContractError(
            "prepared shared apply type is invalid"
        )
    validate_shared_operator_apply_receipt_v1(value.receipt)
    validate_prepared_match_view(
        PreparedMatchViewV1(
            descriptor=value.receipt.output_view,
            pixels=value.pixels,
        )
    )


def resolve_shared_operator_batch_v1(
    *,
    operator: SharedReferenceOperatorV1,
    reference: PreparedMatchViewV1,
    sources: Sequence[PreparedMatchViewV1],
    applies: Sequence[PreparedSharedOperatorApplyV1],
) -> SharedOperatorBatchV1:
    """Bind a complete ordered batch to one and only one shared operator."""

    validate_shared_reference_operator_v1(operator)
    validate_prepared_match_view(reference)
    if (
        reference.descriptor.view_id != operator.reference_view_id
        or reference.descriptor.profile_id != operator.profile_id
    ):
        raise ReferenceMatchContractError(
            "shared operator reference binding mismatch"
        )
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not sources
        or len(sources) > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or not isinstance(applies, Sequence)
        or isinstance(applies, (str, bytes))
        or len(sources) != len(applies)
    ):
        raise ReferenceMatchContractError(
            "shared batch source/apply inventory is invalid"
        )
    source_ids: set[str] = set()
    receipt_ids: set[str] = set()
    result_ids: set[str] = set()
    receipts: list[SharedOperatorApplyReceiptV1] = []
    for index, (source, prepared) in enumerate(
        zip(sources, applies, strict=True)
    ):
        validate_prepared_match_view(source)
        validate_prepared_shared_operator_apply_v1(prepared)
        receipt = prepared.receipt
        if (
            receipt.operator_id != operator.operator_id
            or receipt.source_index != index
            or receipt.source_view_id != source.descriptor.view_id
            or receipt.output_view.shape != source.descriptor.shape
            or receipt.output_view.profile_id != operator.profile_id
        ):
            raise ReferenceMatchContractError(
                "shared batch apply binding mismatch"
            )
        if source.descriptor.view_id in source_ids:
            raise ReferenceMatchContractError(
                "shared batch sources must be distinct"
            )
        if (
            receipt.receipt_id in receipt_ids
            or receipt.producer_apply_result_id in result_ids
        ):
            raise ReferenceMatchContractError(
                "shared batch apply identities must be distinct"
            )
        source_ids.add(source.descriptor.view_id)
        receipt_ids.add(receipt.receipt_id)
        result_ids.add(receipt.producer_apply_result_id)
        receipts.append(receipt)
    provisional = SharedOperatorBatchV1(
        schema_id=SHARED_BATCH_SCHEMA_ID,
        batch_id="0" * 64,
        policy_id=SHARED_BATCH_POLICY_ID,
        claim_ceiling=SHARED_CLAIM_CEILING,
        operator=operator,
        source_count=len(receipts),
        sources=tuple(receipts),
    )
    result = replace(
        provisional,
        batch_id=canonical_sha256(
            _identity_payload(provisional, "batch_id")
        ),
    )
    validate_shared_operator_batch_v1(result)
    return result


def validate_shared_operator_batch_v1(
    value: SharedOperatorBatchV1,
) -> None:
    if not isinstance(value, SharedOperatorBatchV1):
        raise ReferenceMatchContractError(
            "shared batch type is invalid"
        )
    if (
        value.schema_id != SHARED_BATCH_SCHEMA_ID
        or value.policy_id != SHARED_BATCH_POLICY_ID
        or value.claim_ceiling != SHARED_CLAIM_CEILING
    ):
        raise ReferenceMatchContractError(
            "shared batch contract identity is invalid"
        )
    _hash(value.batch_id, "batch_id")
    validate_shared_reference_operator_v1(value.operator)
    if (
        isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.sources)
    ):
        raise ReferenceMatchContractError(
            "shared batch source_count is invalid"
        )
    source_ids: set[str] = set()
    result_ids: set[str] = set()
    for index, receipt in enumerate(value.sources):
        validate_shared_operator_apply_receipt_v1(receipt)
        if (
            receipt.operator_id != value.operator.operator_id
            or receipt.source_index != index
            or receipt.output_view.profile_id != value.operator.profile_id
        ):
            raise ReferenceMatchContractError(
                "shared batch receipt binding is invalid"
            )
        if (
            receipt.source_view_id in source_ids
            or receipt.producer_apply_result_id in result_ids
        ):
            raise ReferenceMatchContractError(
                "shared batch receipt identities are duplicated"
            )
        source_ids.add(receipt.source_view_id)
        result_ids.add(receipt.producer_apply_result_id)
    if value.batch_id != canonical_sha256(
        _identity_payload(value, "batch_id")
    ):
        raise ReferenceMatchContractError(
            "shared batch identity mismatch"
        )


def shared_operator_batch_to_json(value: SharedOperatorBatchV1) -> str:
    validate_shared_operator_batch_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def shared_operator_batch_from_json(encoded: str) -> SharedOperatorBatchV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "shared batch is not valid JSON"
        ) from exc
    payload = _strict(payload, _BATCH_KEYS, "shared batch")
    operator_payload = _strict(
        payload["operator"], _OPERATOR_KEYS, "shared operator"
    )
    operator = SharedReferenceOperatorV1(**operator_payload)
    if not isinstance(payload["sources"], list):
        raise ReferenceMatchContractError(
            "shared batch sources must be an array"
        )
    receipts: list[SharedOperatorApplyReceiptV1] = []
    for index, raw in enumerate(payload["sources"]):
        raw = dict(_strict(raw, _RECEIPT_KEYS, f"sources[{index}]"))
        raw["output_view"] = match_view_from_json(
            json.dumps(raw["output_view"], allow_nan=False)
        )
        receipts.append(SharedOperatorApplyReceiptV1(**raw))
    result = SharedOperatorBatchV1(
        schema_id=payload["schema_id"],
        batch_id=payload["batch_id"],
        policy_id=payload["policy_id"],
        claim_ceiling=payload["claim_ceiling"],
        operator=operator,
        source_count=payload["source_count"],
        sources=tuple(receipts),
    )
    validate_shared_operator_batch_v1(result)
    return result


__all__ = [
    "MAX_REFERENCE_MATCH_BATCH_SOURCES",
    "SHARED_APPLY_RECEIPT_SCHEMA_ID",
    "SHARED_BATCH_POLICY_ID",
    "SHARED_BATCH_SCHEMA_ID",
    "SHARED_CLAIM_CEILING",
    "SHARED_OPERATOR_SCHEMA_ID",
    "SHARED_OUTPUT_BRIDGE_ID",
    "PreparedSharedOperatorApplyV1",
    "SharedOperatorApplyReceiptV1",
    "SharedOperatorBatchV1",
    "SharedReferenceOperatorV1",
    "make_shared_reference_operator_v1",
    "prepare_shared_operator_apply_v1",
    "resolve_shared_operator_batch_v1",
    "shared_operator_batch_from_json",
    "shared_operator_batch_to_json",
    "validate_prepared_shared_operator_apply_v1",
    "validate_shared_operator_apply_receipt_v1",
    "validate_shared_operator_batch_v1",
    "validate_shared_reference_operator_v1",
]
