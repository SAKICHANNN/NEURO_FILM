"""Bind P67 decoded sRGB samples to path-free display-linear MatchViews."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_contracts import MATCH_PROFILE_DISPLAY_SRGB, make_match_view
from .shared_runtime_staging_decode import (
    MAX_DECODED_DIMENSION,
    DecodedRuntimeQualifiedSharedStagingOutputV1,
    RuntimeQualifiedSharedStagingDecodedBatchV1,
    validate_runtime_qualified_shared_staging_decoded_batch_v1,
)


RUNTIME_STAGING_MATCH_VIEW_BRIDGE_SCHEMA_ID = (
    "neuro-film.runtime-staging-match-view-bridge.v1"
)
RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID = (
    "neuro-film.p69-decoded-staging-srgb-eotf-f32.v1"
)
RUNTIME_STAGING_MATCH_VIEW_BRIDGE_CLAIM_CEILING = (
    "process-local-display-linear-match-views-only-"
    "no-persistence-application-or-delivery"
)
_STATE = "prepared-runtime-staging-display-linear-match-views"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "bridge_run_id",
    "decode_id",
    "consumption_id",
    "run_id",
    "bridge_id",
    "profile_id",
    "source_count",
    "state",
    "path_consumption_authorized",
    "persistent_views_authorized",
    "application_authorized",
    "delivery_authorized",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "encoded_output_view_id",
    "decoded_pixel_sha256",
    "output_bit_depth",
    "width",
    "height",
    "match_view_id",
    "match_view_pixel_sha256",
    "provenance_fingerprint",
}


@dataclass(frozen=True)
class RuntimeStagingPreparedMatchViewOutputV1:
    source_index: int
    source_view_id: str
    encoded_output_view_id: str
    decoded_pixel_sha256: str
    output_bit_depth: int
    width: int
    height: int
    match_view_id: str
    match_view_pixel_sha256: str
    provenance_fingerprint: str


@dataclass(frozen=True)
class RuntimeStagingMatchViewBridgeRecordV1:
    schema_id: str
    bridge_run_id: str
    decode_id: str
    consumption_id: str
    run_id: str
    bridge_id: str
    profile_id: str
    source_count: int
    state: str
    path_consumption_authorized: bool
    persistent_views_authorized: bool
    application_authorized: bool
    delivery_authorized: bool
    outputs: tuple[RuntimeStagingPreparedMatchViewOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class RuntimeStagingPreparedMatchViewBatchV1:
    record: RuntimeStagingMatchViewBridgeRecordV1
    decoded: RuntimeQualifiedSharedStagingDecodedBatchV1
    views: tuple[PreparedMatchViewV1, ...]


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the MatchView bridge contract"
        )
    return value


def _identity_payload(
    value: RuntimeStagingMatchViewBridgeRecordV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("bridge_run_id")
    return payload


def _float32_pixel_sha256(pixels: np.ndarray) -> str:
    wire = np.asarray(pixels, dtype=">f4", order="C")
    return hashlib.sha256(wire.tobytes(order="C")).hexdigest()


def _decode_srgb_samples_f32(
    samples: np.ndarray,
    *,
    bit_depth: int,
) -> np.ndarray:
    expected_dtype = np.uint8 if bit_depth == 8 else np.uint16
    if (
        bit_depth not in {8, 16}
        or not isinstance(samples, np.ndarray)
        or samples.dtype != expected_dtype
        or samples.ndim != 3
        or samples.shape[2] != 3
        or samples.flags.writeable
    ):
        raise ReferenceMatchContractError(
            "runtime staging samples do not satisfy the sRGB bridge input"
        )
    denominator = np.float32(255.0 if bit_depth == 8 else 65535.0)
    encoded = np.asarray(samples, dtype=np.float32) / denominator
    threshold = np.float32(0.04045)
    linear = np.where(
        encoded <= threshold,
        encoded / np.float32(12.92),
        np.power(
            (encoded + np.float32(0.055)) / np.float32(1.055),
            np.float32(2.4),
        ),
    )
    result = np.array(linear, dtype=np.float32, order="C", copy=True)
    if (
        not np.isfinite(result).all()
        or float(np.min(result)) < 0.0
        or float(np.max(result)) > 1.0
    ):
        raise ReferenceMatchContractError(
            "runtime staging sRGB EOTF produced invalid pixels"
        )
    result.flags.writeable = False
    return result


def _provenance_fingerprint(
    *,
    decoded: RuntimeQualifiedSharedStagingDecodedBatchV1,
    row: DecodedRuntimeQualifiedSharedStagingOutputV1,
) -> str:
    return canonical_sha256(
        {
            "schema_id": RUNTIME_STAGING_MATCH_VIEW_BRIDGE_SCHEMA_ID,
            "bridge_id": RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID,
            "decode_id": decoded.record.decode_id,
            "consumption_id": decoded.record.consumption_id,
            "run_id": decoded.record.run_id,
            "source_index": row.source_index,
            "source_view_id": row.source_view_id,
            "encoded_output_view_id": row.output_view_id,
            "decoded_pixel_sha256": row.decoded_pixel_sha256,
            "output_format": row.output_format,
            "output_bit_depth": row.output_bit_depth,
            "width": row.width,
            "height": row.height,
            "sample_interpretation": "display-srgb-encoded-rgb",
            "eotf": "iec-61966-2-1-float32-v1",
        }
    )


def prepare_runtime_staging_match_views_v1(
    decoded: RuntimeQualifiedSharedStagingDecodedBatchV1,
) -> RuntimeStagingPreparedMatchViewBatchV1:
    validate_runtime_qualified_shared_staging_decoded_batch_v1(decoded)
    views: list[PreparedMatchViewV1] = []
    outputs: list[RuntimeStagingPreparedMatchViewOutputV1] = []
    for row, samples in zip(
        decoded.record.outputs,
        decoded.pixels,
        strict=True,
    ):
        pixels = _decode_srgb_samples_f32(
            samples,
            bit_depth=row.output_bit_depth,
        )
        pixel_sha256 = _float32_pixel_sha256(pixels)
        provenance = _provenance_fingerprint(
            decoded=decoded,
            row=row,
        )
        descriptor = make_match_view(
            profile_id=MATCH_PROFILE_DISPLAY_SRGB,
            pixel_sha256=pixel_sha256,
            shape=tuple(int(value) for value in pixels.shape),
            render_bridge_id=RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID,
            provenance_fingerprint=provenance,
            alpha_mode="absent",
        )
        prepared = PreparedMatchViewV1(
            descriptor=descriptor,
            pixels=pixels,
        )
        validate_prepared_match_view(prepared)
        views.append(prepared)
        outputs.append(
            RuntimeStagingPreparedMatchViewOutputV1(
                source_index=row.source_index,
                source_view_id=row.source_view_id,
                encoded_output_view_id=row.output_view_id,
                decoded_pixel_sha256=row.decoded_pixel_sha256,
                output_bit_depth=row.output_bit_depth,
                width=row.width,
                height=row.height,
                match_view_id=descriptor.view_id,
                match_view_pixel_sha256=descriptor.pixel_sha256,
                provenance_fingerprint=provenance,
            )
        )
    provisional = RuntimeStagingMatchViewBridgeRecordV1(
        schema_id=RUNTIME_STAGING_MATCH_VIEW_BRIDGE_SCHEMA_ID,
        bridge_run_id="0" * 64,
        decode_id=decoded.record.decode_id,
        consumption_id=decoded.record.consumption_id,
        run_id=decoded.record.run_id,
        bridge_id=RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID,
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        source_count=decoded.record.source_count,
        state=_STATE,
        path_consumption_authorized=False,
        persistent_views_authorized=False,
        application_authorized=False,
        delivery_authorized=False,
        outputs=tuple(outputs),
        claim_ceiling=(
            RUNTIME_STAGING_MATCH_VIEW_BRIDGE_CLAIM_CEILING
        ),
    )
    record = replace(
        provisional,
        bridge_run_id=canonical_sha256(_identity_payload(provisional)),
    )
    result = RuntimeStagingPreparedMatchViewBatchV1(
        record=record,
        decoded=decoded,
        views=tuple(views),
    )
    validate_runtime_staging_prepared_match_view_batch_v1(result)
    return result


def validate_runtime_staging_match_view_bridge_record_v1(
    value: RuntimeStagingMatchViewBridgeRecordV1,
) -> None:
    if not isinstance(value, RuntimeStagingMatchViewBridgeRecordV1):
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge record type is invalid"
        )
    if value.schema_id != RUNTIME_STAGING_MATCH_VIEW_BRIDGE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge schema is invalid"
        )
    for field in (
        "bridge_run_id",
        "decode_id",
        "consumption_id",
        "run_id",
    ):
        _hash(getattr(value, field), field)
    if (
        value.bridge_id != RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID
        or value.profile_id != MATCH_PROFILE_DISPLAY_SRGB
        or value.state != _STATE
        or value.path_consumption_authorized is not False
        or value.persistent_views_authorized is not False
        or value.application_authorized is not False
        or value.delivery_authorized is not False
        or value.claim_ceiling
        != RUNTIME_STAGING_MATCH_VIEW_BRIDGE_CLAIM_CEILING
        or isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge authority/metadata is invalid"
        )
    source_ids: set[str] = set()
    encoded_ids: set[str] = set()
    match_ids: set[str] = set()
    for index, output in enumerate(value.outputs):
        if (
            not isinstance(output, RuntimeStagingPreparedMatchViewOutputV1)
            or output.source_index != index
            or output.output_bit_depth not in {8, 16}
            or isinstance(output.width, bool)
            or isinstance(output.height, bool)
            or not isinstance(output.width, int)
            or not isinstance(output.height, int)
            or output.width <= 0
            or output.height <= 0
            or output.width > MAX_DECODED_DIMENSION
            or output.height > MAX_DECODED_DIMENSION
        ):
            raise ReferenceMatchContractError(
                "runtime staging MatchView bridge output is invalid"
            )
        for field in (
            "source_view_id",
            "encoded_output_view_id",
            "decoded_pixel_sha256",
            "match_view_id",
            "match_view_pixel_sha256",
            "provenance_fingerprint",
        ):
            _hash(getattr(output, field), f"output.{field}")
        if (
            output.source_view_id in source_ids
            or output.encoded_output_view_id in encoded_ids
            or output.match_view_id in match_ids
        ):
            raise ReferenceMatchContractError(
                "runtime staging MatchView identities must be distinct"
            )
        source_ids.add(output.source_view_id)
        encoded_ids.add(output.encoded_output_view_id)
        match_ids.add(output.match_view_id)
    if value.bridge_run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge identity mismatch"
        )


def validate_runtime_staging_prepared_match_view_batch_v1(
    value: RuntimeStagingPreparedMatchViewBatchV1,
) -> None:
    if not isinstance(value, RuntimeStagingPreparedMatchViewBatchV1):
        raise ReferenceMatchContractError(
            "runtime staging prepared MatchView batch type is invalid"
        )
    validate_runtime_staging_match_view_bridge_record_v1(value.record)
    validate_runtime_qualified_shared_staging_decoded_batch_v1(
        value.decoded
    )
    if (
        value.record.decode_id != value.decoded.record.decode_id
        or value.record.consumption_id
        != value.decoded.record.consumption_id
        or value.record.run_id != value.decoded.record.run_id
        or len(value.views) != value.record.source_count
    ):
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge binding mismatch"
        )
    for record, decoded_row, decoded_pixels, prepared in zip(
        value.record.outputs,
        value.decoded.record.outputs,
        value.decoded.pixels,
        value.views,
        strict=True,
    ):
        validate_prepared_match_view(prepared)
        descriptor = prepared.descriptor
        expected_pixels = _decode_srgb_samples_f32(
            decoded_pixels,
            bit_depth=decoded_row.output_bit_depth,
        )
        expected_provenance = _provenance_fingerprint(
            decoded=value.decoded,
            row=decoded_row,
        )
        if (
            record.source_index != decoded_row.source_index
            or record.source_view_id != decoded_row.source_view_id
            or record.encoded_output_view_id != decoded_row.output_view_id
            or record.decoded_pixel_sha256
            != decoded_row.decoded_pixel_sha256
            or record.output_bit_depth != decoded_row.output_bit_depth
            or record.width != decoded_row.width
            or record.height != decoded_row.height
            or record.match_view_id != descriptor.view_id
            or record.match_view_pixel_sha256
            != descriptor.pixel_sha256
            or record.provenance_fingerprint
            != descriptor.provenance_fingerprint
            or record.provenance_fingerprint != expected_provenance
            or descriptor.profile_id != MATCH_PROFILE_DISPLAY_SRGB
            or descriptor.render_bridge_id
            != RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID
            or _float32_pixel_sha256(prepared.pixels)
            != record.match_view_pixel_sha256
            or not np.array_equal(prepared.pixels, expected_pixels)
        ):
            raise ReferenceMatchContractError(
                "runtime staging prepared MatchView does not match binding"
            )


def runtime_staging_match_view_bridge_record_to_json(
    value: RuntimeStagingMatchViewBridgeRecordV1,
) -> str:
    validate_runtime_staging_match_view_bridge_record_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_staging_match_view_bridge_record_from_json(
    encoded: str,
) -> RuntimeStagingMatchViewBridgeRecordV1:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge record is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "runtime staging MatchView bridge")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime staging MatchView outputs must be non-empty"
        )
    outputs: list[RuntimeStagingPreparedMatchViewOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"prepared MatchView {index}")
        try:
            outputs.append(
                RuntimeStagingPreparedMatchViewOutputV1(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime staging MatchView output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeStagingMatchViewBridgeRecordV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging MatchView bridge fields are invalid"
        ) from exc
    validate_runtime_staging_match_view_bridge_record_v1(result)
    return result


__all__ = [
    "RUNTIME_STAGING_MATCH_VIEW_BRIDGE_CLAIM_CEILING",
    "RUNTIME_STAGING_MATCH_VIEW_BRIDGE_ID",
    "RUNTIME_STAGING_MATCH_VIEW_BRIDGE_SCHEMA_ID",
    "RuntimeStagingMatchViewBridgeRecordV1",
    "RuntimeStagingPreparedMatchViewBatchV1",
    "RuntimeStagingPreparedMatchViewOutputV1",
    "prepare_runtime_staging_match_views_v1",
    "runtime_staging_match_view_bridge_record_from_json",
    "runtime_staging_match_view_bridge_record_to_json",
    "validate_runtime_staging_match_view_bridge_record_v1",
    "validate_runtime_staging_prepared_match_view_batch_v1",
]
