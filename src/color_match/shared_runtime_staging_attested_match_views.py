"""Build display-linear MatchViews only from P71 colour-attested bytes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any, Mapping

import numpy as np

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .core_adapter import PreparedMatchViewV1, validate_prepared_match_view
from .core_contracts import MATCH_PROFILE_DISPLAY_SRGB, make_match_view
from .shared_runtime_staging_color_attestation import (
    RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID,
    AttestedRuntimeStagingColorOutputV1,
    RuntimeStagingColorAttestedDecodedBatchV1,
    validate_runtime_staging_color_attested_decoded_batch_v1,
)
from .shared_runtime_staging_decode import (
    MAX_DECODED_DIMENSION,
    MAX_DECODED_OUTPUT_PIXELS,
    DecodedRuntimeQualifiedSharedStagingOutputV1,
)
from .shared_runtime_staging_match_views import (
    _decode_srgb_samples_f32,
    _float32_pixel_sha256,
)
from .strict_json import strict_json_loads
from .srgb_icc_profile import (
    SRGB_ICC_PROFILE_SHA256,
    srgb_icc_profile_v1,
)


RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_SCHEMA_ID = (
    "neuro-film.runtime-staging-attested-match-view-bridge.v2"
)
RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID = (
    "neuro-film.p72-attested-staging-srgb-eotf-f32.v2"
)
RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_CLAIM_CEILING = (
    "process-local-metadata-attested-display-linear-match-views-only-"
    "no-persistence-application-or-delivery"
)
EXPECTED_SRGB_ICC_SHA256 = SRGB_ICC_PROFILE_SHA256
_STATE = "prepared-attested-runtime-staging-display-linear-match-views"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "bridge_run_id",
    "attestation_id",
    "attestation_policy_id",
    "decode_id",
    "consumption_id",
    "run_id",
    "bridge_id",
    "profile_id",
    "expected_profile_sha256",
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
    "metadata_attestation_id",
    "profile_binding",
    "profile_sha256",
    "match_view_id",
    "match_view_pixel_sha256",
    "provenance_fingerprint",
}


@dataclass(frozen=True)
class RuntimeStagingAttestedPreparedMatchViewOutputV2:
    source_index: int
    source_view_id: str
    encoded_output_view_id: str
    decoded_pixel_sha256: str
    output_bit_depth: int
    width: int
    height: int
    metadata_attestation_id: str
    profile_binding: str
    profile_sha256: str
    match_view_id: str
    match_view_pixel_sha256: str
    provenance_fingerprint: str


@dataclass(frozen=True)
class RuntimeStagingAttestedMatchViewBridgeRecordV2:
    schema_id: str
    bridge_run_id: str
    attestation_id: str
    attestation_policy_id: str
    decode_id: str
    consumption_id: str
    run_id: str
    bridge_id: str
    profile_id: str
    expected_profile_sha256: str
    source_count: int
    state: str
    path_consumption_authorized: bool
    persistent_views_authorized: bool
    application_authorized: bool
    delivery_authorized: bool
    outputs: tuple[RuntimeStagingAttestedPreparedMatchViewOutputV2, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class RuntimeStagingAttestedPreparedMatchViewBatchV2:
    record: RuntimeStagingAttestedMatchViewBridgeRecordV2
    attested: RuntimeStagingColorAttestedDecodedBatchV1
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
            f"{label} fields differ from the attested MatchView contract"
        )
    return value


def _identity_payload(
    value: RuntimeStagingAttestedMatchViewBridgeRecordV2,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("bridge_run_id")
    return payload


def _provenance_fingerprint(
    *,
    attested: RuntimeStagingColorAttestedDecodedBatchV1,
    decoded_row: DecodedRuntimeQualifiedSharedStagingOutputV1,
    metadata_row: AttestedRuntimeStagingColorOutputV1,
) -> str:
    return canonical_sha256(
        {
            "schema_id": (
                RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_SCHEMA_ID
            ),
            "bridge_id": RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID,
            "attestation_id": attested.record.attestation_id,
            "attestation_policy_id": attested.record.policy_id,
            "metadata_attestation_id": (
                metadata_row.metadata_attestation_id
            ),
            "profile_binding": metadata_row.profile_binding,
            "profile_sha256": metadata_row.profile_sha256,
            "decode_id": attested.record.decode_id,
            "consumption_id": attested.record.consumption_id,
            "run_id": attested.record.run_id,
            "source_index": decoded_row.source_index,
            "source_view_id": decoded_row.source_view_id,
            "encoded_output_view_id": decoded_row.output_view_id,
            "decoded_pixel_sha256": decoded_row.decoded_pixel_sha256,
            "output_format": decoded_row.output_format,
            "output_bit_depth": decoded_row.output_bit_depth,
            "width": decoded_row.width,
            "height": decoded_row.height,
            "sample_interpretation": (
                "metadata-attested-display-srgb-encoded-rgb"
            ),
            "eotf": "iec-61966-2-1-float32-v1",
        }
    )


def prepare_runtime_staging_attested_match_views_v2(
    attested: RuntimeStagingColorAttestedDecodedBatchV1,
) -> RuntimeStagingAttestedPreparedMatchViewBatchV2:
    if (
        not isinstance(attested, RuntimeStagingColorAttestedDecodedBatchV1)
        or not isinstance(attested.decoded.record.outputs, tuple)
        or not isinstance(attested.decoded.pixels, tuple)
        or not isinstance(attested.decoded.snapshot.record.outputs, tuple)
        or not isinstance(attested.decoded.snapshot.output_bytes, tuple)
    ):
        raise ReferenceMatchContractError(
            "attested MatchView input object graph must be immutable"
        )
    validate_runtime_staging_color_attested_decoded_batch_v1(attested)
    views: list[PreparedMatchViewV1] = []
    outputs: list[RuntimeStagingAttestedPreparedMatchViewOutputV2] = []
    for decoded_row, metadata_row, samples in zip(
        attested.decoded.record.outputs,
        attested.record.outputs,
        attested.decoded.pixels,
        strict=True,
    ):
        pixels = _decode_srgb_samples_f32(
            samples,
            bit_depth=decoded_row.output_bit_depth,
        )
        pixel_sha256 = _float32_pixel_sha256(pixels)
        provenance = _provenance_fingerprint(
            attested=attested,
            decoded_row=decoded_row,
            metadata_row=metadata_row,
        )
        descriptor = make_match_view(
            profile_id=MATCH_PROFILE_DISPLAY_SRGB,
            pixel_sha256=pixel_sha256,
            shape=tuple(int(value) for value in pixels.shape),
            render_bridge_id=(
                RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID
            ),
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
            RuntimeStagingAttestedPreparedMatchViewOutputV2(
                source_index=decoded_row.source_index,
                source_view_id=decoded_row.source_view_id,
                encoded_output_view_id=decoded_row.output_view_id,
                decoded_pixel_sha256=decoded_row.decoded_pixel_sha256,
                output_bit_depth=decoded_row.output_bit_depth,
                width=decoded_row.width,
                height=decoded_row.height,
                metadata_attestation_id=(
                    metadata_row.metadata_attestation_id
                ),
                profile_binding=metadata_row.profile_binding,
                profile_sha256=metadata_row.profile_sha256,
                match_view_id=descriptor.view_id,
                match_view_pixel_sha256=descriptor.pixel_sha256,
                provenance_fingerprint=provenance,
            )
        )
    provisional = RuntimeStagingAttestedMatchViewBridgeRecordV2(
        schema_id=RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_SCHEMA_ID,
        bridge_run_id="0" * 64,
        attestation_id=attested.record.attestation_id,
        attestation_policy_id=attested.record.policy_id,
        decode_id=attested.record.decode_id,
        consumption_id=attested.record.consumption_id,
        run_id=attested.record.run_id,
        bridge_id=RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID,
        profile_id=MATCH_PROFILE_DISPLAY_SRGB,
        expected_profile_sha256=(
            attested.record.expected_profile_sha256
        ),
        source_count=attested.record.source_count,
        state=_STATE,
        path_consumption_authorized=False,
        persistent_views_authorized=False,
        application_authorized=False,
        delivery_authorized=False,
        outputs=tuple(outputs),
        claim_ceiling=(
            RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_CLAIM_CEILING
        ),
    )
    record = replace(
        provisional,
        bridge_run_id=canonical_sha256(_identity_payload(provisional)),
    )
    result = RuntimeStagingAttestedPreparedMatchViewBatchV2(
        record=record,
        attested=attested,
        views=tuple(views),
    )
    validate_runtime_staging_attested_prepared_match_view_batch_v2(
        result
    )
    return result


def validate_runtime_staging_attested_match_view_bridge_record_v2(
    value: RuntimeStagingAttestedMatchViewBridgeRecordV2,
) -> None:
    if not isinstance(value, RuntimeStagingAttestedMatchViewBridgeRecordV2):
        raise ReferenceMatchContractError(
            "attested runtime staging MatchView record type is invalid"
        )
    for field in (
        "bridge_run_id",
        "attestation_id",
        "decode_id",
        "consumption_id",
        "run_id",
        "expected_profile_sha256",
    ):
        _hash(getattr(value, field), field)
    srgb_icc_profile_v1()
    if (
        value.schema_id
        != RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_SCHEMA_ID
        or value.attestation_policy_id
        != RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID
        or value.bridge_id
        != RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID
        or value.profile_id != MATCH_PROFILE_DISPLAY_SRGB
        or value.expected_profile_sha256 != EXPECTED_SRGB_ICC_SHA256
        or value.state != _STATE
        or value.path_consumption_authorized is not False
        or value.persistent_views_authorized is not False
        or value.application_authorized is not False
        or value.delivery_authorized is not False
        or value.claim_ceiling
        != RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_CLAIM_CEILING
        or isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or not isinstance(value.outputs, tuple)
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "attested MatchView bridge authority/metadata is invalid"
        )
    source_ids: set[str] = set()
    encoded_ids: set[str] = set()
    metadata_ids: set[str] = set()
    match_ids: set[str] = set()
    bindings = {
        "png-iccp-exact-v1",
        "jpeg-app2-icc-exact-v1",
        "tiff-34675-icc-exact-v1",
    }
    for index, output in enumerate(value.outputs):
        if (
            not isinstance(
                output,
                RuntimeStagingAttestedPreparedMatchViewOutputV2,
            )
            or isinstance(output.source_index, bool)
            or not isinstance(output.source_index, int)
            or output.source_index != index
            or isinstance(output.output_bit_depth, bool)
            or not isinstance(output.output_bit_depth, int)
            or output.output_bit_depth not in {8, 16}
            or isinstance(output.width, bool)
            or isinstance(output.height, bool)
            or not isinstance(output.width, int)
            or not isinstance(output.height, int)
            or output.width <= 0
            or output.height <= 0
            or output.width > MAX_DECODED_DIMENSION
            or output.height > MAX_DECODED_DIMENSION
            or output.width * output.height > MAX_DECODED_OUTPUT_PIXELS
            or output.profile_binding not in bindings
            or (
                output.profile_binding == "jpeg-app2-icc-exact-v1"
                and output.output_bit_depth != 8
            )
            or output.profile_sha256 != value.expected_profile_sha256
        ):
            raise ReferenceMatchContractError(
                "attested MatchView bridge output is invalid"
            )
        for field in (
            "source_view_id",
            "encoded_output_view_id",
            "decoded_pixel_sha256",
            "metadata_attestation_id",
            "profile_sha256",
            "match_view_id",
            "match_view_pixel_sha256",
            "provenance_fingerprint",
        ):
            _hash(getattr(output, field), f"output.{field}")
        if (
            output.source_view_id in source_ids
            or output.encoded_output_view_id in encoded_ids
            or output.metadata_attestation_id in metadata_ids
            or output.match_view_id in match_ids
        ):
            raise ReferenceMatchContractError(
                "attested MatchView identities must be distinct"
            )
        source_ids.add(output.source_view_id)
        encoded_ids.add(output.encoded_output_view_id)
        metadata_ids.add(output.metadata_attestation_id)
        match_ids.add(output.match_view_id)
    if value.bridge_run_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "attested MatchView bridge identity mismatch"
        )


def validate_runtime_staging_attested_prepared_match_view_batch_v2(
    value: RuntimeStagingAttestedPreparedMatchViewBatchV2,
) -> None:
    if not isinstance(
        value,
        RuntimeStagingAttestedPreparedMatchViewBatchV2,
    ):
        raise ReferenceMatchContractError(
            "attested prepared MatchView batch type is invalid"
        )
    validate_runtime_staging_attested_match_view_bridge_record_v2(
        value.record
    )
    validate_runtime_staging_color_attested_decoded_batch_v1(
        value.attested
    )
    if (
        not isinstance(value.attested.decoded.record.outputs, tuple)
        or not isinstance(value.attested.decoded.pixels, tuple)
        or not isinstance(
            value.attested.decoded.snapshot.record.outputs,
            tuple,
        )
        or not isinstance(
            value.attested.decoded.snapshot.output_bytes,
            tuple,
        )
        or value.record.attestation_id
        != value.attested.record.attestation_id
        or value.record.attestation_policy_id
        != value.attested.record.policy_id
        or value.record.expected_profile_sha256
        != value.attested.record.expected_profile_sha256
        or value.record.decode_id != value.attested.record.decode_id
        or value.record.consumption_id
        != value.attested.record.consumption_id
        or value.record.run_id != value.attested.record.run_id
        or value.record.source_count != value.attested.record.source_count
        or not isinstance(value.views, tuple)
        or len(value.views) != value.record.source_count
    ):
        raise ReferenceMatchContractError(
            "attested MatchView bridge binding mismatch"
        )
    for record, decoded_row, metadata_row, decoded_pixels, prepared in zip(
        value.record.outputs,
        value.attested.decoded.record.outputs,
        value.attested.record.outputs,
        value.attested.decoded.pixels,
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
            attested=value.attested,
            decoded_row=decoded_row,
            metadata_row=metadata_row,
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
            or record.metadata_attestation_id
            != metadata_row.metadata_attestation_id
            or record.profile_binding != metadata_row.profile_binding
            or record.profile_sha256 != metadata_row.profile_sha256
            or record.match_view_id != descriptor.view_id
            or record.match_view_pixel_sha256
            != descriptor.pixel_sha256
            or record.provenance_fingerprint
            != descriptor.provenance_fingerprint
            or record.provenance_fingerprint != expected_provenance
            or descriptor.profile_id != MATCH_PROFILE_DISPLAY_SRGB
            or descriptor.render_bridge_id
            != RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID
            or _float32_pixel_sha256(prepared.pixels)
            != record.match_view_pixel_sha256
            or not np.array_equal(prepared.pixels, expected_pixels)
        ):
            raise ReferenceMatchContractError(
                "attested prepared MatchView does not match binding"
            )


def runtime_staging_attested_match_view_bridge_record_to_json(
    value: RuntimeStagingAttestedMatchViewBridgeRecordV2,
) -> str:
    validate_runtime_staging_attested_match_view_bridge_record_v2(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_staging_attested_match_view_bridge_record_from_json(
    encoded: str,
) -> RuntimeStagingAttestedMatchViewBridgeRecordV2:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReferenceMatchContractError(
            "attested MatchView bridge record is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "attested MatchView bridge")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "attested MatchView outputs must be non-empty"
        )
    outputs: list[RuntimeStagingAttestedPreparedMatchViewOutputV2] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"attested MatchView {index}")
        try:
            outputs.append(
                RuntimeStagingAttestedPreparedMatchViewOutputV2(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "attested MatchView output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeStagingAttestedMatchViewBridgeRecordV2(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "attested MatchView bridge fields are invalid"
        ) from exc
    validate_runtime_staging_attested_match_view_bridge_record_v2(result)
    return result


__all__ = [
    "EXPECTED_SRGB_ICC_SHA256",
    "RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_CLAIM_CEILING",
    "RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_ID",
    "RUNTIME_STAGING_ATTESTED_MATCH_VIEW_BRIDGE_SCHEMA_ID",
    "RuntimeStagingAttestedMatchViewBridgeRecordV2",
    "RuntimeStagingAttestedPreparedMatchViewBatchV2",
    "RuntimeStagingAttestedPreparedMatchViewOutputV2",
    "prepare_runtime_staging_attested_match_views_v2",
    "runtime_staging_attested_match_view_bridge_record_from_json",
    "runtime_staging_attested_match_view_bridge_record_to_json",
    "validate_runtime_staging_attested_match_view_bridge_record_v2",
    "validate_runtime_staging_attested_prepared_match_view_batch_v2",
]
