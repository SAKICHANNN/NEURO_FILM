"""Consumer-owned identity receipt for future external-core output pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import (
    PreparedMatchViewV1,
    validate_prepared_match_view,
)
from .core_contracts import (
    CapabilitiesV1,
    DiagnosticsV1,
    MatchViewV1,
    TransformBundleV1,
    make_match_view,
    match_view_from_json,
    validate_core_binding,
    validate_match_view,
)


CORE_APPLY_RECEIPT_SCHEMA_ID = (
    "neuro-film.reference-core-apply-receipt.v1"
)
CORE_APPLY_OUTPUT_CONTRACT_ID = (
    "neuro-film.same-profile-same-shape-f32.v1"
)
CORE_OUTPUT_BRIDGE_ID = "neuro-film.external-core-output.v1"
_DELIVERY_STATE = "candidate-only"
_KEYS = {
    "schema_id",
    "receipt_id",
    "output_contract_id",
    "delivery_state",
    "transform_id",
    "source_view_id",
    "reference_view_id",
    "capability_id",
    "diagnostics_sha256",
    "output_view",
}


@dataclass(frozen=True)
class CoreApplyReceiptV1:
    """Canonical consumer receipt; never a final delivery decision."""

    schema_id: str
    receipt_id: str
    output_contract_id: str
    delivery_state: str
    transform_id: str
    source_view_id: str
    reference_view_id: str
    capability_id: str
    diagnostics_sha256: str
    output_view: MatchViewV1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_view"] = self.output_view.to_dict()
        return payload


@dataclass(frozen=True)
class PreparedCoreApplyReceiptV1:
    """Receipt plus isolated read-only output bytes held by the consumer."""

    receipt: CoreApplyReceiptV1
    pixels: np.ndarray


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _pixel_sha256(pixels: np.ndarray) -> str:
    encoded = np.asarray(pixels, dtype=">f4", order="C")
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def _diagnostics_sha256(diagnostics: DiagnosticsV1) -> str:
    return canonical_sha256(diagnostics.to_dict())


def _receipt_identity_payload(
    value: CoreApplyReceiptV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("receipt_id")
    return payload


def _output_provenance(
    *,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1,
) -> str:
    return canonical_sha256(
        {
            "schema_id": "neuro-film.external-core-output-provenance.v1",
            "source_view_id": source.view_id,
            "reference_view_id": reference.view_id,
            "transform_id": transform.transform_id,
            "capability_id": capabilities.capability_id,
            "diagnostics_sha256": _diagnostics_sha256(diagnostics),
        }
    )


def prepare_core_apply_receipt(
    *,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1,
    output_pixels: np.ndarray,
) -> PreparedCoreApplyReceiptV1:
    """Copy and bind one successful future-core output as a candidate."""

    validate_core_binding(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    if diagnostics.status != "ok" or diagnostics.finite is not True:
        raise ReferenceMatchContractError(
            "only finite ok core output can receive an apply receipt"
        )
    pixels = np.array(
        output_pixels,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if tuple(pixels.shape) != source.shape:
        raise ReferenceMatchContractError(
            "core output shape must match the bound source"
        )
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(
            "core output pixels must be finite"
        )
    pixels.flags.writeable = False
    output_view = make_match_view(
        profile_id=source.profile_id,
        pixel_sha256=_pixel_sha256(pixels),
        shape=source.shape,
        render_bridge_id=CORE_OUTPUT_BRIDGE_ID,
        provenance_fingerprint=_output_provenance(
            source=source,
            reference=reference,
            transform=transform,
            capabilities=capabilities,
            diagnostics=diagnostics,
        ),
        alpha_mode="absent",
    )
    provisional = CoreApplyReceiptV1(
        schema_id=CORE_APPLY_RECEIPT_SCHEMA_ID,
        receipt_id="0" * 64,
        output_contract_id=CORE_APPLY_OUTPUT_CONTRACT_ID,
        delivery_state=_DELIVERY_STATE,
        transform_id=transform.transform_id,
        source_view_id=source.view_id,
        reference_view_id=reference.view_id,
        capability_id=capabilities.capability_id,
        diagnostics_sha256=_diagnostics_sha256(diagnostics),
        output_view=output_view,
    )
    receipt = replace(
        provisional,
        receipt_id=canonical_sha256(
            _receipt_identity_payload(provisional)
        ),
    )
    prepared = PreparedCoreApplyReceiptV1(
        receipt=receipt,
        pixels=pixels,
    )
    validate_prepared_core_apply_receipt(prepared)
    validate_core_apply_receipt_binding(
        receipt,
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    return prepared


def validate_core_apply_receipt(value: CoreApplyReceiptV1) -> None:
    if not isinstance(value, CoreApplyReceiptV1):
        raise ReferenceMatchContractError(
            "core apply receipt must be CoreApplyReceiptV1"
        )
    if value.schema_id != CORE_APPLY_RECEIPT_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported core apply receipt schema"
        )
    _sha256(value.receipt_id, "receipt_id")
    if value.output_contract_id != CORE_APPLY_OUTPUT_CONTRACT_ID:
        raise ReferenceMatchContractError(
            "unsupported core output contract"
        )
    if value.delivery_state != _DELIVERY_STATE:
        raise ReferenceMatchContractError(
            "core apply receipt must remain candidate-only"
        )
    _sha256(value.transform_id, "transform_id")
    _sha256(value.source_view_id, "source_view_id")
    _sha256(value.reference_view_id, "reference_view_id")
    _sha256(value.capability_id, "capability_id")
    _sha256(value.diagnostics_sha256, "diagnostics_sha256")
    validate_match_view(value.output_view)
    if value.output_view.render_bridge_id != CORE_OUTPUT_BRIDGE_ID:
        raise ReferenceMatchContractError(
            "core output view has an unsupported render bridge"
        )
    if value.output_view.alpha_mode != "absent":
        raise ReferenceMatchContractError(
            "core output view must not contain alpha"
        )
    if value.receipt_id != canonical_sha256(
        _receipt_identity_payload(value)
    ):
        raise ReferenceMatchContractError(
            "receipt_id does not match canonical payload"
        )


def validate_core_apply_receipt_binding(
    value: CoreApplyReceiptV1,
    *,
    source: MatchViewV1,
    reference: MatchViewV1,
    transform: TransformBundleV1,
    capabilities: CapabilitiesV1,
    diagnostics: DiagnosticsV1,
) -> None:
    """Validate the receipt against every execution-side identity."""

    validate_core_apply_receipt(value)
    validate_core_binding(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    if diagnostics.status != "ok" or diagnostics.finite is not True:
        raise ReferenceMatchContractError(
            "receipt binding requires finite ok diagnostics"
        )
    identities = (
        value.transform_id == transform.transform_id,
        value.source_view_id == source.view_id,
        value.reference_view_id == reference.view_id,
        value.capability_id == capabilities.capability_id,
        value.diagnostics_sha256 == _diagnostics_sha256(diagnostics),
    )
    if not all(identities):
        raise ReferenceMatchContractError(
            "core apply receipt execution identity mismatch"
        )
    if value.output_view.shape != source.shape:
        raise ReferenceMatchContractError(
            "core output shape does not match source"
        )
    if value.output_view.profile_id != source.profile_id:
        raise ReferenceMatchContractError(
            "core output profile does not match source"
        )
    expected_provenance = _output_provenance(
        source=source,
        reference=reference,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
    )
    if value.output_view.provenance_fingerprint != expected_provenance:
        raise ReferenceMatchContractError(
            "core output provenance mismatch"
        )


def validate_prepared_core_apply_receipt(
    value: PreparedCoreApplyReceiptV1,
) -> None:
    if not isinstance(value, PreparedCoreApplyReceiptV1):
        raise ReferenceMatchContractError(
            "prepared core output must be PreparedCoreApplyReceiptV1"
        )
    validate_core_apply_receipt(value.receipt)
    validate_prepared_match_view(
        PreparedMatchViewV1(
            descriptor=value.receipt.output_view,
            pixels=value.pixels,
        )
    )


def core_apply_receipt_to_json(value: CoreApplyReceiptV1) -> str:
    validate_core_apply_receipt(value)
    return json.dumps(
        value.to_dict(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def core_apply_receipt_from_json(encoded: str) -> CoreApplyReceiptV1:
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "core apply receipt is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "core apply receipt must be an object"
        )
    _strict_keys(payload, _KEYS, "core apply receipt")
    output_payload = payload["output_view"]
    if not isinstance(output_payload, Mapping):
        raise ReferenceMatchContractError(
            "core apply receipt output_view must be an object"
        )
    converted = dict(payload)
    try:
        encoded_view = json.dumps(output_payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "core apply receipt output_view is not finite JSON"
        ) from exc
    converted["output_view"] = match_view_from_json(encoded_view)
    try:
        result = CoreApplyReceiptV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "core apply receipt contains invalid fields"
        ) from exc
    validate_core_apply_receipt(result)
    return result


__all__ = [
    "CORE_APPLY_OUTPUT_CONTRACT_ID",
    "CORE_APPLY_RECEIPT_SCHEMA_ID",
    "CORE_OUTPUT_BRIDGE_ID",
    "CoreApplyReceiptV1",
    "PreparedCoreApplyReceiptV1",
    "core_apply_receipt_from_json",
    "core_apply_receipt_to_json",
    "prepare_core_apply_receipt",
    "validate_core_apply_receipt",
    "validate_core_apply_receipt_binding",
    "validate_prepared_core_apply_receipt",
]
