"""Pinned D-PCT producer-v2 to Neuro-Film consumer adapter.

The adapter independently verifies Zhuise producer identities.  It never
imports a mutable producer checkout and never treats producer identifiers as
Neuro-Film identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import struct
from typing import Any, Mapping

import numpy as np

from .contracts import ReferenceMatchContractError
from .core_adapter import (
    PreparedMatchViewV1,
    validate_prepared_match_view,
)
from .core_apply_receipt import (
    PreparedCoreApplyReceiptV1,
    prepare_core_apply_receipt,
)
from .core_contracts import (
    DIAGNOSTICS_SCHEMA_ID,
    MATCH_PROFILE_DISPLAY_SRGB,
    CapabilitiesV1,
    DiagnosticsV1,
    TransformBundleV1,
    make_capabilities,
    make_transform_bundle,
)


DPCT_COMPATIBILITY_PROFILE_ID = "neuro-film.dpct-consumer.v2"
DPCT_PINNED_COMMIT = "11c581ecdd0a41a840c4e0f94112cfb597e00b0a"
DPCT_PRODUCER_ID = "zhuise-dpct"
DPCT_PRODUCER_PROFILE_ID = (
    "zhuise.display-linear-srgb-d65-relative-f32.v1"
)
DPCT_MATCH_VIEW_SCHEMA = "zhuise.match-view.v1"
DPCT_TRANSFORM_SCHEMA = "zhuise.transform-bundle.v1"
DPCT_DIAGNOSTICS_SCHEMA = "zhuise.diagnostics.v2"
DPCT_APPLY_RESULT_SCHEMA = "zhuise.apply-result.v2"
DPCT_PIXEL_LAYOUT = "f32be-rgb-interleaved-row-major"
DPCT_CANONICAL_JSON = "zhuise-json-sort-keys-utf8-v1"

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_VIEW_KEYS = {
    "schema",
    "profile_id",
    "width",
    "height",
    "channels",
    "pixel_layout",
    "pixel_byte_length",
    "pixel_sha256",
    "view_id",
}
_TRANSFORM_KEYS = {
    "schema",
    "algorithm_id",
    "algorithm_version",
    "capability_id",
    "profile_id",
    "source_view_id",
    "reference_view_id",
    "payload_format",
    "payload_byte_length",
    "payload_sha256",
    "bundle_id",
}
_BACKEND_KEYS = {
    "backend_id",
    "backend_version",
    "build_fingerprint",
}
_MEASUREMENT_KEYS = {
    "all_finite",
    "output_minimum",
    "output_maximum",
    "out_of_gamut_fraction",
    "clipping_fraction",
    "projected_fraction",
    "timing_ms",
    "confidence",
}
_DIAGNOSTICS_KEYS = {
    "schema",
    "canonical_json",
    "status",
    "capability_id",
    "source_view_id",
    "reference_view_id",
    "bundle_id",
    "failure_code",
    "backend",
    "measurements",
    "warnings",
    "diagnostics_id",
}
_RESULT_KEYS = {
    "schema",
    "disposition",
    "capability_id",
    "source_view_id",
    "reference_view_id",
    "bundle_id",
    "diagnostics_id",
    "source_width",
    "source_height",
    "output",
    "result_id",
}


@dataclass(frozen=True)
class DpctProducerAliasesV2:
    """Authoritative producer IDs retained beside consumer identities."""

    compatibility_profile_id: str
    producer_commit: str
    source_view_id: str
    reference_view_id: str
    bundle_id: str
    diagnostics_id: str
    result_id: str
    capability_id: str


@dataclass(frozen=True)
class AdaptedDpctCandidateV2:
    """One verified producer candidate mapped into the consumer boundary."""

    aliases: DpctProducerAliasesV2
    transform: TransformBundleV1
    capabilities: CapabilitiesV1
    diagnostics: DiagnosticsV1
    prepared_output: PreparedCoreApplyReceiptV1


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceMatchContractError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ReferenceMatchContractError(
            f"{label} keys mismatch; "
            f"missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _producer_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be lowercase sha256:<hex>"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a namespaced identifier"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceMatchContractError(
            f"{label} must be a positive integer"
        )
    return value


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceMatchContractError(f"{label} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ReferenceMatchContractError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ReferenceMatchContractError(
            f"{label} must be at least {minimum}"
        )
    if maximum is not None and result > maximum:
        raise ReferenceMatchContractError(
            f"{label} must be at most {maximum}"
        )
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "producer identity contains non-canonical JSON"
        ) from exc


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verify_view(
    envelope: Any,
    pixel_f32be: bytes,
    *,
    prepared: PreparedMatchViewV1 | None,
    label: str,
) -> Mapping[str, Any]:
    view = _strict(envelope, _VIEW_KEYS, label)
    if view["schema"] != DPCT_MATCH_VIEW_SCHEMA:
        raise ReferenceMatchContractError(f"{label} schema mismatch")
    if view["profile_id"] != DPCT_PRODUCER_PROFILE_ID:
        raise ReferenceMatchContractError(f"{label} profile mismatch")
    height = _positive_int(view["height"], f"{label}.height")
    width = _positive_int(view["width"], f"{label}.width")
    if view["channels"] != 3 or view["pixel_layout"] != DPCT_PIXEL_LAYOUT:
        raise ReferenceMatchContractError(f"{label} layout mismatch")
    expected_length = height * width * 12
    if view["pixel_byte_length"] != expected_length:
        raise ReferenceMatchContractError(
            f"{label} pixel_byte_length mismatch"
        )
    if not isinstance(pixel_f32be, bytes) or len(pixel_f32be) != expected_length:
        raise ReferenceMatchContractError(
            f"{label} pixel payload length mismatch"
        )
    pixel_sha = _sha256(pixel_f32be)
    if _producer_hash(view["pixel_sha256"], f"{label}.pixel_sha256") != pixel_sha:
        raise ReferenceMatchContractError(f"{label} pixel SHA-256 mismatch")
    header = (
        b"ZhuiseMatchViewV1\0"
        + DPCT_PRODUCER_PROFILE_ID.encode("ascii")
        + b"\0"
        + DPCT_PIXEL_LAYOUT.encode("ascii")
        + b"\0"
        + struct.pack(">II", height, width)
    )
    expected_view_id = _sha256(header + pixel_f32be)
    if _producer_hash(view["view_id"], f"{label}.view_id") != expected_view_id:
        raise ReferenceMatchContractError(f"{label} view_id mismatch")
    pixels = np.frombuffer(pixel_f32be, dtype=">f4")
    if not np.isfinite(pixels).all():
        raise ReferenceMatchContractError(f"{label} pixels are non-finite")
    if prepared is not None:
        validate_prepared_match_view(prepared)
        if prepared.descriptor.profile_id != MATCH_PROFILE_DISPLAY_SRGB:
            raise ReferenceMatchContractError(
                f"{label} consumer profile is not mapped"
            )
        if prepared.descriptor.shape != (height, width, 3):
            raise ReferenceMatchContractError(
                f"{label} consumer geometry mismatch"
            )
        if prepared.descriptor.pixel_sha256 != pixel_sha.removeprefix(
            "sha256:"
        ):
            raise ReferenceMatchContractError(
                f"{label} consumer pixel identity mismatch"
            )
    return view


def _verify_transform(
    envelope: Any,
    payload: bytes,
    *,
    source_id: str,
    reference_id: str,
) -> Mapping[str, Any]:
    transform = _strict(envelope, _TRANSFORM_KEYS, "producer transform")
    if transform["schema"] != DPCT_TRANSFORM_SCHEMA:
        raise ReferenceMatchContractError("producer transform schema mismatch")
    if transform["profile_id"] != DPCT_PRODUCER_PROFILE_ID:
        raise ReferenceMatchContractError("producer transform profile mismatch")
    for key in (
        "algorithm_id",
        "algorithm_version",
        "capability_id",
        "payload_format",
    ):
        _identifier(transform[key], f"producer transform.{key}")
    if (
        transform["source_view_id"] != source_id
        or transform["reference_view_id"] != reference_id
    ):
        raise ReferenceMatchContractError(
            "producer transform view binding mismatch"
        )
    if not isinstance(payload, bytes):
        raise ReferenceMatchContractError(
            "producer transform payload must be bytes"
        )
    if transform["payload_byte_length"] != len(payload):
        raise ReferenceMatchContractError(
            "producer transform payload length mismatch"
        )
    if transform["payload_sha256"] != _sha256(payload):
        raise ReferenceMatchContractError(
            "producer transform payload SHA-256 mismatch"
        )
    identity = dict(transform)
    bundle_id = identity.pop("bundle_id")
    if _producer_hash(bundle_id, "producer transform.bundle_id") != _sha256(
        _canonical_json(identity)
    ):
        raise ReferenceMatchContractError("producer bundle_id mismatch")
    return transform


def _verify_diagnostics(
    envelope: Any,
    *,
    transform: Mapping[str, Any],
) -> Mapping[str, Any]:
    diagnostics = _strict(
        envelope, _DIAGNOSTICS_KEYS, "producer diagnostics"
    )
    if diagnostics["schema"] != DPCT_DIAGNOSTICS_SCHEMA:
        raise ReferenceMatchContractError(
            "producer diagnostics schema mismatch"
        )
    if diagnostics["canonical_json"] != DPCT_CANONICAL_JSON:
        raise ReferenceMatchContractError(
            "producer diagnostics canonical JSON mismatch"
        )
    if diagnostics["status"] != "candidate":
        raise ReferenceMatchContractError(
            "failed producer diagnostics cannot yield candidate pixels"
        )
    if diagnostics["failure_code"] is not None:
        raise ReferenceMatchContractError(
            "candidate diagnostics must not contain failure_code"
        )
    expected = (
        transform["capability_id"],
        transform["source_view_id"],
        transform["reference_view_id"],
        transform["bundle_id"],
    )
    actual = tuple(
        diagnostics[key]
        for key in (
            "capability_id",
            "source_view_id",
            "reference_view_id",
            "bundle_id",
        )
    )
    if actual != expected:
        raise ReferenceMatchContractError(
            "producer diagnostics transform binding mismatch"
        )
    backend = _strict(
        diagnostics["backend"], _BACKEND_KEYS, "producer backend"
    )
    _identifier(backend["backend_id"], "producer backend.backend_id")
    _identifier(
        backend["backend_version"], "producer backend.backend_version"
    )
    _producer_hash(
        backend["build_fingerprint"],
        "producer backend.build_fingerprint",
    )
    measurements = _strict(
        diagnostics["measurements"],
        _MEASUREMENT_KEYS,
        "producer measurements",
    )
    if measurements["all_finite"] is not True:
        raise ReferenceMatchContractError(
            "producer candidate must report all_finite=true"
        )
    minimum = _finite(
        measurements["output_minimum"], "producer output_minimum"
    )
    maximum = _finite(
        measurements["output_maximum"], "producer output_maximum"
    )
    if minimum > maximum:
        raise ReferenceMatchContractError(
            "producer output range is inconsistent"
        )
    for key in (
        "out_of_gamut_fraction",
        "clipping_fraction",
        "projected_fraction",
    ):
        _finite(
            measurements[key],
            f"producer {key}",
            minimum=0.0,
            maximum=1.0,
        )
    _finite(
        measurements["timing_ms"],
        "producer timing_ms",
        minimum=0.0,
    )
    if measurements["confidence"] is not None:
        _finite(
            measurements["confidence"],
            "producer confidence",
            minimum=0.0,
            maximum=1.0,
        )
    warnings = diagnostics["warnings"]
    if (
        not isinstance(warnings, list)
        or any(not isinstance(item, str) or not item for item in warnings)
    ):
        raise ReferenceMatchContractError(
            "producer warnings must be non-empty strings"
        )
    identity = dict(diagnostics)
    diagnostics_id = identity.pop("diagnostics_id")
    if _producer_hash(
        diagnostics_id, "producer diagnostics.diagnostics_id"
    ) != _sha256(_canonical_json(identity)):
        raise ReferenceMatchContractError("producer diagnostics_id mismatch")
    return diagnostics


def _verify_result(
    envelope: Any,
    output_f32be: bytes,
    *,
    source: Mapping[str, Any],
    transform: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> tuple[Mapping[str, Any], np.ndarray]:
    result = _strict(envelope, _RESULT_KEYS, "producer apply result")
    if (
        result["schema"] != DPCT_APPLY_RESULT_SCHEMA
        or result["disposition"] != "candidate-only"
    ):
        raise ReferenceMatchContractError(
            "producer apply result is not candidate-only v2"
        )
    expected = (
        transform["capability_id"],
        transform["source_view_id"],
        transform["reference_view_id"],
        transform["bundle_id"],
        diagnostics["diagnostics_id"],
    )
    actual = tuple(
        result[key]
        for key in (
            "capability_id",
            "source_view_id",
            "reference_view_id",
            "bundle_id",
            "diagnostics_id",
        )
    )
    if actual != expected:
        raise ReferenceMatchContractError(
            "producer apply result binding mismatch"
        )
    if (
        result["source_width"] != source["width"]
        or result["source_height"] != source["height"]
    ):
        raise ReferenceMatchContractError(
            "producer apply result source geometry mismatch"
        )
    output = _verify_view(
        result["output"],
        output_f32be,
        prepared=None,
        label="producer output",
    )
    if (
        output["width"] != source["width"]
        or output["height"] != source["height"]
        or output["profile_id"] != source["profile_id"]
    ):
        raise ReferenceMatchContractError(
            "producer output geometry/profile mismatch"
        )
    identity = {
        "schema": DPCT_APPLY_RESULT_SCHEMA,
        "disposition": "candidate-only",
        "capability_id": result["capability_id"],
        "source_view_id": result["source_view_id"],
        "reference_view_id": result["reference_view_id"],
        "bundle_id": result["bundle_id"],
        "diagnostics_id": result["diagnostics_id"],
        "source_width": result["source_width"],
        "source_height": result["source_height"],
        "output_view_id": output["view_id"],
    }
    if _producer_hash(result["result_id"], "producer result_id") != _sha256(
        _canonical_json(identity)
    ):
        raise ReferenceMatchContractError("producer result_id mismatch")
    pixels = np.frombuffer(output_f32be, dtype=">f4").astype(
        np.float32, copy=True
    )
    pixels = pixels.reshape(output["height"], output["width"], 3)
    measurements = diagnostics["measurements"]
    if (
        float(np.min(pixels)) != float(measurements["output_minimum"])
        or float(np.max(pixels)) != float(measurements["output_maximum"])
    ):
        raise ReferenceMatchContractError(
            "producer output range does not match delivered pixels"
        )
    return result, pixels


def adapt_dpct_candidate_v2(
    *,
    source: PreparedMatchViewV1,
    reference: PreparedMatchViewV1,
    producer_source: Mapping[str, Any],
    producer_reference: Mapping[str, Any],
    producer_transform: Mapping[str, Any],
    producer_transform_payload: bytes,
    producer_diagnostics: Mapping[str, Any],
    producer_apply_result: Mapping[str, Any],
    output_pixel_f32be: bytes,
    intent_id: str,
) -> AdaptedDpctCandidateV2:
    """Verify one pinned D-PCT v2 delivery and issue a candidate receipt."""

    validate_prepared_match_view(source)
    validate_prepared_match_view(reference)
    source_bytes = source.pixels.astype(">f4", copy=False).tobytes(order="C")
    reference_bytes = (
        reference.pixels.astype(">f4", copy=False).tobytes(order="C")
    )
    producer_source = _verify_view(
        producer_source,
        source_bytes,
        prepared=source,
        label="producer source",
    )
    producer_reference = _verify_view(
        producer_reference,
        reference_bytes,
        prepared=reference,
        label="producer reference",
    )
    producer_transform = _verify_transform(
        producer_transform,
        producer_transform_payload,
        source_id=producer_source["view_id"],
        reference_id=producer_reference["view_id"],
    )
    producer_diagnostics = _verify_diagnostics(
        producer_diagnostics,
        transform=producer_transform,
    )
    producer_apply_result, output_pixels = _verify_result(
        producer_apply_result,
        output_pixel_f32be,
        source=producer_source,
        transform=producer_transform,
        diagnostics=producer_diagnostics,
    )
    backend = producer_diagnostics["backend"]
    measurements = producer_diagnostics["measurements"]
    build_sha256 = backend["build_fingerprint"].removeprefix("sha256:")
    transform = make_transform_bundle(
        producer_id=DPCT_PRODUCER_ID,
        producer_version=backend["backend_version"],
        producer_build_sha256=build_sha256,
        contract_schema_id=DPCT_TRANSFORM_SCHEMA,
        algorithm_id=producer_transform["algorithm_id"],
        algorithm_version=producer_transform["algorithm_version"],
        intent_id=intent_id,
        source_view_id=source.descriptor.view_id,
        reference_view_id=reference.descriptor.view_id,
        payload_schema_id=producer_transform["payload_format"],
        payload_sha256=producer_transform["payload_sha256"].removeprefix(
            "sha256:"
        ),
        capability_requirements=(
            "candidate-only",
            "hard-bounds-no-projection",
            "source-bound-fit",
        ),
        determinism="bounded-numeric",
    )
    capabilities = make_capabilities(
        producer_id=DPCT_PRODUCER_ID,
        producer_version=backend["backend_version"],
        producer_build_sha256=build_sha256,
        contract_schema_ids=(
            DPCT_TRANSFORM_SCHEMA,
            DPCT_DIAGNOSTICS_SCHEMA,
            DPCT_APPLY_RESULT_SCHEMA,
        ),
        supported_profile_ids=(MATCH_PROFILE_DISPLAY_SRGB,),
        supported_algorithm_ids=(producer_transform["algorithm_id"],),
        feature_flags=(
            "candidate-only",
            "hard-bounds-no-projection",
            "source-bound-fit",
        ),
        determinism="bounded-numeric",
    )
    diagnostics = DiagnosticsV1(
        schema_id=DIAGNOSTICS_SCHEMA_ID,
        transform_id=transform.transform_id,
        status="ok",
        fallback_reason=None,
        finite=True,
        out_of_gamut_fraction=float(
            measurements["out_of_gamut_fraction"]
        ),
        clipping_fraction=float(measurements["clipping_fraction"]),
        projected_fraction=float(measurements["projected_fraction"]),
        confidence=(
            None
            if measurements["confidence"] is None
            else float(measurements["confidence"])
        ),
        timing_ms=float(measurements["timing_ms"]),
        backend_id=(
            f"{backend['backend_id']}@{backend['backend_version']}"
        ),
        backend_fingerprint=build_sha256,
        warnings=tuple(sorted(set(producer_diagnostics["warnings"]))),
    )
    prepared_output = prepare_core_apply_receipt(
        source=source.descriptor,
        reference=reference.descriptor,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        output_pixels=output_pixels,
    )
    aliases = DpctProducerAliasesV2(
        compatibility_profile_id=DPCT_COMPATIBILITY_PROFILE_ID,
        producer_commit=DPCT_PINNED_COMMIT,
        source_view_id=producer_source["view_id"],
        reference_view_id=producer_reference["view_id"],
        bundle_id=producer_transform["bundle_id"],
        diagnostics_id=producer_diagnostics["diagnostics_id"],
        result_id=producer_apply_result["result_id"],
        capability_id=producer_transform["capability_id"],
    )
    return AdaptedDpctCandidateV2(
        aliases=aliases,
        transform=transform,
        capabilities=capabilities,
        diagnostics=diagnostics,
        prepared_output=prepared_output,
    )


__all__ = [
    "DPCT_COMPATIBILITY_PROFILE_ID",
    "DPCT_PINNED_COMMIT",
    "AdaptedDpctCandidateV2",
    "DpctProducerAliasesV2",
    "adapt_dpct_candidate_v2",
]
