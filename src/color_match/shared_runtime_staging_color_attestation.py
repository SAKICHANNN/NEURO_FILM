"""Exact in-memory sRGB metadata attestation for P67 decoded staging bytes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import hashlib
import json
import re
import struct
from typing import Any, Mapping
import zlib

from PIL import Image
import tifffile

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .srgb_icc_profile import srgb_icc_profile_v1
from .shared_runtime_staging_decode import (
    RuntimeQualifiedSharedStagingDecodedBatchV1,
    _parse_strict_png,
    _preflight_jpeg,
    _preflight_tiff,
    validate_runtime_qualified_shared_staging_decoded_batch_v1,
)
from .strict_json import strict_json_loads


RUNTIME_STAGING_COLOR_ATTESTATION_SCHEMA_ID = (
    "neuro-film.runtime-staging-color-attestation.v1"
)
RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID = (
    "neuro-film.exact-embedded-srgb-icc-no-conflict.v1"
)
RUNTIME_STAGING_COLOR_ATTESTATION_CLAIM_CEILING = (
    "embedded-srgb-metadata-attested-for-process-local-bridge-only-"
    "no-persistence-application-or-delivery"
)
MAX_EMBEDDED_PROFILE_BYTES = 1024 * 1024
MAX_COMPRESSED_PROFILE_BYTES = 64 * 1024
_STATE = "attested-runtime-staging-embedded-srgb"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "attestation_id",
    "policy_id",
    "decode_id",
    "consumption_id",
    "run_id",
    "expected_profile_sha256",
    "expected_profile_size_bytes",
    "source_count",
    "state",
    "path_consumption_authorized",
    "persistent_attestation_authorized",
    "application_authorized",
    "delivery_authorized",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "output_view_id",
    "encoded_file_sha256",
    "output_format",
    "output_bit_depth",
    "profile_binding",
    "profile_sha256",
    "profile_size_bytes",
    "orientation",
    "metadata_attestation_id",
}


@dataclass(frozen=True)
class AttestedRuntimeStagingColorOutputV1:
    source_index: int
    source_view_id: str
    output_view_id: str
    encoded_file_sha256: str
    output_format: str
    output_bit_depth: int
    profile_binding: str
    profile_sha256: str
    profile_size_bytes: int
    orientation: int
    metadata_attestation_id: str


@dataclass(frozen=True)
class RuntimeStagingColorAttestationRecordV1:
    schema_id: str
    attestation_id: str
    policy_id: str
    decode_id: str
    consumption_id: str
    run_id: str
    expected_profile_sha256: str
    expected_profile_size_bytes: int
    source_count: int
    state: str
    path_consumption_authorized: bool
    persistent_attestation_authorized: bool
    application_authorized: bool
    delivery_authorized: bool
    outputs: tuple[AttestedRuntimeStagingColorOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class RuntimeStagingColorAttestedDecodedBatchV1:
    record: RuntimeStagingColorAttestationRecordV1
    decoded: RuntimeQualifiedSharedStagingDecodedBatchV1


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the colour attestation contract"
        )
    return value


def _identity_payload(
    value: RuntimeStagingColorAttestationRecordV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("attestation_id")
    return payload


def _bounded_decompress_profile(payload: bytes) -> bytes:
    if not 1 <= len(payload) <= MAX_COMPRESSED_PROFILE_BYTES:
        raise ReferenceMatchContractError(
            "PNG compressed iCCP profile exceeds bounds"
        )
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(
            payload,
            MAX_EMBEDDED_PROFILE_BYTES + 1,
        )
    except zlib.error as exc:
        raise ReferenceMatchContractError(
            "PNG iCCP profile compression is invalid"
        ) from exc
    if (
        len(decoded) <= 0
        or len(decoded) > MAX_EMBEDDED_PROFILE_BYTES
        or not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
    ):
        raise ReferenceMatchContractError(
            "PNG iCCP profile exceeds bounds or has trailing data"
        )
    try:
        tail = decoder.flush(
            MAX_EMBEDDED_PROFILE_BYTES + 1 - len(decoded)
        )
    except zlib.error as exc:
        raise ReferenceMatchContractError(
            "PNG iCCP profile compression is invalid"
        ) from exc
    decoded += tail
    if len(decoded) > MAX_EMBEDDED_PROFILE_BYTES:
        raise ReferenceMatchContractError(
            "PNG iCCP profile exceeds bounds"
        )
    return decoded


def _png_profile(raw: bytes, depth: int) -> tuple[bytes, str, int]:
    _parse_strict_png(raw, depth)
    offset = 8
    profiles: list[bytes] = []
    conflicting: list[str] = []
    seen_idat = False
    seen_plte = False
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + length]
        if chunk_type == b"iCCP":
            try:
                name, remainder = payload.split(b"\x00", 1)
            except ValueError as exc:
                raise ReferenceMatchContractError(
                    "PNG iCCP profile name is invalid"
                ) from exc
            if (
                not 1 <= len(name) <= 79
                or any(
                    byte not in range(32, 127)
                    and byte not in range(161, 256)
                    for byte in name
                )
                or name[:1] == b" "
                or name[-1:] == b" "
                or b"  " in name
                or len(remainder) < 2
                or remainder[0] != 0
                or seen_idat
                or seen_plte
            ):
                raise ReferenceMatchContractError(
                    "PNG iCCP profile header is invalid"
                )
            profiles.append(_bounded_decompress_profile(remainder[1:]))
        elif chunk_type in {
            b"cHRM",
            b"cICP",
            b"cLLi",
            b"eXIf",
            b"gAMA",
            b"mDCv",
            b"sBIT",
            b"sRGB",
            b"tEXt",
            b"iTXt",
            b"zTXt",
        }:
            conflicting.append(chunk_type.decode("ascii"))
        if chunk_type == b"IDAT":
            seen_idat = True
        if chunk_type == b"PLTE":
            seen_plte = True
        offset += 12 + length
    if conflicting:
        raise ReferenceMatchContractError(
            "PNG has conflicting colour/orientation metadata: "
            + ",".join(conflicting)
        )
    if len(profiles) != 1:
        raise ReferenceMatchContractError(
            "PNG must contain exactly one embedded iCCP profile"
        )
    return profiles[0], "png-iccp-exact-v1", 1


_JPEG_STANDALONE = frozenset(
    {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}
)


def _jpeg_segments(raw: bytes) -> tuple[tuple[int, bytes], ...]:
    offset = 2
    in_entropy = False
    ended = False
    segments: list[tuple[int, bytes]] = []
    while offset < len(raw):
        if in_entropy:
            marker_offset = raw.find(b"\xff", offset)
            if marker_offset < 0:
                raise ReferenceMatchContractError(
                    "JPEG entropy stream has no final marker"
                )
            offset = marker_offset
        if raw[offset] != 0xFF:
            raise ReferenceMatchContractError(
                "JPEG metadata marker prefix is invalid"
            )
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset >= len(raw):
            raise ReferenceMatchContractError("JPEG marker is truncated")
        marker = raw[offset]
        offset += 1
        if in_entropy and marker == 0x00:
            continue
        if marker in range(0xD0, 0xD8):
            if not in_entropy:
                raise ReferenceMatchContractError(
                    "JPEG restart marker is outside entropy data"
                )
            continue
        in_entropy = False
        if marker == 0xD9:
            ended = True
            break
        if marker in _JPEG_STANDALONE:
            raise ReferenceMatchContractError(
                "JPEG has an unexpected standalone marker"
            )
        if offset + 2 > len(raw):
            raise ReferenceMatchContractError(
                "JPEG metadata length is truncated"
            )
        length = int.from_bytes(raw[offset : offset + 2], "big")
        if length < 2 or offset + length > len(raw):
            raise ReferenceMatchContractError(
                "JPEG metadata segment is invalid"
            )
        payload = raw[offset + 2 : offset + length]
        offset += length
        segments.append((marker, payload))
        if marker == 0xDA:
            in_entropy = True
    if not ended or offset != len(raw):
        raise ReferenceMatchContractError(
            "JPEG final marker/trailing bytes are invalid"
        )
    return tuple(segments)


def _jpeg_profile(raw: bytes, depth: int) -> tuple[bytes, str, int]:
    _preflight_jpeg(raw, depth)
    segments: dict[int, bytes] = {}
    expected_count: int | None = None
    for marker, payload in _jpeg_segments(raw):
        if 0xE0 <= marker <= 0xEF and marker not in {0xE0, 0xE2}:
            raise ReferenceMatchContractError(
                "JPEG alternate metadata namespaces are forbidden"
            )
        if marker == 0xE2 and not payload.startswith(b"ICC_PROFILE\x00"):
            raise ReferenceMatchContractError(
                "JPEG non-ICC APP2 metadata is forbidden"
            )
        if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
            if len(payload) < 14:
                raise ReferenceMatchContractError(
                    "JPEG ICC segment is truncated"
                )
            sequence = int(payload[12])
            count = int(payload[13])
            if (
                count <= 0
                or sequence <= 0
                or sequence > count
                or (expected_count is not None and count != expected_count)
                or sequence in segments
            ):
                raise ReferenceMatchContractError(
                    "JPEG ICC segment ordering is invalid"
                )
            expected_count = count
            segments[sequence] = payload[14:]
    if (
        expected_count is None
        or len(segments) != expected_count
        or set(segments) != set(range(1, expected_count + 1))
    ):
        raise ReferenceMatchContractError(
            "JPEG must contain one complete ICC APP2 sequence"
        )
    if sum(len(segment) for segment in segments.values()) > (
        MAX_EMBEDDED_PROFILE_BYTES
    ):
        raise ReferenceMatchContractError(
            "JPEG embedded ICC profile exceeds bounds"
        )
    profile = b"".join(segments[index] for index in range(1, expected_count + 1))
    if not 1 <= len(profile) <= MAX_EMBEDDED_PROFILE_BYTES:
        raise ReferenceMatchContractError(
            "JPEG embedded ICC profile exceeds bounds"
        )
    orientation = 1
    try:
        with Image.open(BytesIO(raw)) as image:
            pillow_profile = bytes(image.info.get("icc_profile") or b"")
    except (OSError, ValueError, TypeError) as exc:
        raise ReferenceMatchContractError(
            "JPEG metadata cannot be inspected"
        ) from exc
    if pillow_profile != profile:
        raise ReferenceMatchContractError(
            "JPEG ICC parser reconstruction mismatch"
        )
    return profile, "jpeg-app2-icc-exact-v1", orientation


def _tiff_profile(raw: bytes, depth: int) -> tuple[bytes, str, int]:
    _preflight_tiff(raw, depth)
    try:
        with tifffile.TiffFile(BytesIO(raw)) as tif:
            page = tif.pages[0]
            if any(
                page.tags.getall(code)
                for code in (
                    280,
                    281,
                    301,
                    318,
                    319,
                    320,
                    340,
                    341,
                    529,
                    532,
                    700,
                    34377,
                    34665,
                    34853,
                    40965,
                )
            ):
                raise ReferenceMatchContractError(
                    "TIFF alternate metadata namespaces are forbidden"
                )
            profile_tags = tuple(page.tags.getall(34675) or ())
            orientation_tags = tuple(page.tags.getall(274) or ())
            if len(profile_tags) != 1 or len(orientation_tags) > 1:
                raise ReferenceMatchContractError(
                    "TIFF colour/orientation tags must be unique"
                )
            profile_tag = profile_tags[0]
            orientation_tag = (
                orientation_tags[0] if orientation_tags else None
            )
            if (
                profile_tag is not None
                and int(profile_tag.count) > MAX_EMBEDDED_PROFILE_BYTES
            ):
                raise ReferenceMatchContractError(
                    "TIFF embedded ICC profile exceeds bounds"
                )
            profile = (
                bytes(profile_tag.value)
                if profile_tag is not None
                else b""
            )
            orientation = (
                int(orientation_tag.value)
                if orientation_tag is not None
                else 1
            )
    except ReferenceMatchContractError:
        raise
    except (OSError, ValueError, TypeError, tifffile.TiffFileError) as exc:
        raise ReferenceMatchContractError(
            "TIFF colour metadata cannot be inspected"
        ) from exc
    if not 1 <= len(profile) <= MAX_EMBEDDED_PROFILE_BYTES:
        raise ReferenceMatchContractError(
            "TIFF must contain one bounded ICC tag 34675"
        )
    if orientation != 1:
        raise ReferenceMatchContractError(
            "TIFF orientation must be absent or identity"
        )
    return profile, "tiff-34675-icc-exact-v1", orientation


def _extract_profile(
    raw: bytes,
    *,
    format_name: str,
    bit_depth: int,
) -> tuple[bytes, str, int]:
    if format_name == "PNG":
        return _png_profile(raw, bit_depth)
    if format_name == "JPEG":
        return _jpeg_profile(raw, bit_depth)
    if format_name == "TIFF":
        return _tiff_profile(raw, bit_depth)
    raise ReferenceMatchContractError(
        "colour attestation format is unsupported"
    )


def _output_attestation_id(
    *,
    row: Any,
    profile_binding: str,
    profile_sha256: str,
    profile_size_bytes: int,
    orientation: int,
) -> str:
    return canonical_sha256(
        {
            "schema_id": RUNTIME_STAGING_COLOR_ATTESTATION_SCHEMA_ID,
            "policy_id": RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID,
            "source_index": row.source_index,
            "source_view_id": row.source_view_id,
            "output_view_id": row.output_view_id,
            "encoded_file_sha256": row.encoded_file_sha256,
            "output_format": row.output_format,
            "output_bit_depth": row.output_bit_depth,
            "profile_binding": profile_binding,
            "profile_sha256": profile_sha256,
            "profile_size_bytes": profile_size_bytes,
            "orientation": orientation,
        }
    )


def attest_runtime_staging_srgb_metadata_v1(
    decoded: RuntimeQualifiedSharedStagingDecodedBatchV1,
) -> RuntimeStagingColorAttestedDecodedBatchV1:
    validate_runtime_qualified_shared_staging_decoded_batch_v1(decoded)
    expected = srgb_icc_profile_v1()
    expected_sha = hashlib.sha256(expected).hexdigest()
    outputs: list[AttestedRuntimeStagingColorOutputV1] = []
    for row, raw in zip(
        decoded.record.outputs,
        decoded.snapshot.output_bytes,
        strict=True,
    ):
        profile, binding, orientation = _extract_profile(
            raw,
            format_name=row.output_format,
            bit_depth=row.output_bit_depth,
        )
        if profile != expected:
            raise ReferenceMatchContractError(
                "embedded colour profile does not match exact sRGB profile"
            )
        profile_sha = hashlib.sha256(profile).hexdigest()
        metadata_id = _output_attestation_id(
            row=row,
            profile_binding=binding,
            profile_sha256=profile_sha,
            profile_size_bytes=len(profile),
            orientation=orientation,
        )
        outputs.append(
            AttestedRuntimeStagingColorOutputV1(
                source_index=row.source_index,
                source_view_id=row.source_view_id,
                output_view_id=row.output_view_id,
                encoded_file_sha256=row.encoded_file_sha256,
                output_format=row.output_format,
                output_bit_depth=row.output_bit_depth,
                profile_binding=binding,
                profile_sha256=profile_sha,
                profile_size_bytes=len(profile),
                orientation=orientation,
                metadata_attestation_id=metadata_id,
            )
        )
    provisional = RuntimeStagingColorAttestationRecordV1(
        schema_id=RUNTIME_STAGING_COLOR_ATTESTATION_SCHEMA_ID,
        attestation_id="0" * 64,
        policy_id=RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID,
        decode_id=decoded.record.decode_id,
        consumption_id=decoded.record.consumption_id,
        run_id=decoded.record.run_id,
        expected_profile_sha256=expected_sha,
        expected_profile_size_bytes=len(expected),
        source_count=decoded.record.source_count,
        state=_STATE,
        path_consumption_authorized=False,
        persistent_attestation_authorized=False,
        application_authorized=False,
        delivery_authorized=False,
        outputs=tuple(outputs),
        claim_ceiling=RUNTIME_STAGING_COLOR_ATTESTATION_CLAIM_CEILING,
    )
    record = replace(
        provisional,
        attestation_id=canonical_sha256(_identity_payload(provisional)),
    )
    result = RuntimeStagingColorAttestedDecodedBatchV1(
        record=record,
        decoded=decoded,
    )
    validate_runtime_staging_color_attested_decoded_batch_v1(result)
    return result


def validate_runtime_staging_color_attestation_record_v1(
    value: RuntimeStagingColorAttestationRecordV1,
) -> None:
    if not isinstance(value, RuntimeStagingColorAttestationRecordV1):
        raise ReferenceMatchContractError(
            "runtime staging colour attestation record type is invalid"
        )
    expected = srgb_icc_profile_v1()
    expected_sha = hashlib.sha256(expected).hexdigest()
    for field in (
        "attestation_id",
        "decode_id",
        "consumption_id",
        "run_id",
        "expected_profile_sha256",
    ):
        _hash(getattr(value, field), field)
    if (
        value.schema_id != RUNTIME_STAGING_COLOR_ATTESTATION_SCHEMA_ID
        or value.policy_id != RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID
        or value.expected_profile_sha256 != expected_sha
        or value.expected_profile_size_bytes != len(expected)
        or value.state != _STATE
        or value.path_consumption_authorized is not False
        or value.persistent_attestation_authorized is not False
        or value.application_authorized is not False
        or value.delivery_authorized is not False
        or value.claim_ceiling
        != RUNTIME_STAGING_COLOR_ATTESTATION_CLAIM_CEILING
        or isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or not isinstance(value.outputs, tuple)
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "runtime staging colour attestation authority/metadata is invalid"
        )
    source_ids: set[str] = set()
    output_ids: set[str] = set()
    attestation_ids: set[str] = set()
    bindings = {
        "PNG": "png-iccp-exact-v1",
        "JPEG": "jpeg-app2-icc-exact-v1",
        "TIFF": "tiff-34675-icc-exact-v1",
    }
    for index, output in enumerate(value.outputs):
        if (
            not isinstance(output, AttestedRuntimeStagingColorOutputV1)
            or isinstance(output.source_index, bool)
            or not isinstance(output.source_index, int)
            or output.source_index != index
            or output.output_format not in bindings
            or isinstance(output.output_bit_depth, bool)
            or not isinstance(output.output_bit_depth, int)
            or output.output_bit_depth not in {8, 16}
            or (
                output.output_format == "JPEG"
                and output.output_bit_depth != 8
            )
            or output.profile_binding != bindings[output.output_format]
            or output.profile_sha256 != expected_sha
            or isinstance(output.profile_size_bytes, bool)
            or not isinstance(output.profile_size_bytes, int)
            or output.profile_size_bytes != len(expected)
            or isinstance(output.orientation, bool)
            or not isinstance(output.orientation, int)
            or output.orientation != 1
        ):
            raise ReferenceMatchContractError(
                "runtime staging colour attestation output is invalid"
            )
        for field in (
            "source_view_id",
            "output_view_id",
            "encoded_file_sha256",
            "profile_sha256",
            "metadata_attestation_id",
        ):
            _hash(getattr(output, field), f"output.{field}")
        if output.metadata_attestation_id != _output_attestation_id(
            row=output,
            profile_binding=output.profile_binding,
            profile_sha256=output.profile_sha256,
            profile_size_bytes=output.profile_size_bytes,
            orientation=output.orientation,
        ):
            raise ReferenceMatchContractError(
                "runtime staging metadata attestation identity mismatch"
            )
        if (
            output.source_view_id in source_ids
            or output.output_view_id in output_ids
            or output.metadata_attestation_id in attestation_ids
        ):
            raise ReferenceMatchContractError(
                "runtime staging colour attestation identities "
                "must be distinct"
            )
        source_ids.add(output.source_view_id)
        output_ids.add(output.output_view_id)
        attestation_ids.add(output.metadata_attestation_id)
    if value.attestation_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime staging colour attestation identity mismatch"
        )


def validate_runtime_staging_color_attested_decoded_batch_v1(
    value: RuntimeStagingColorAttestedDecodedBatchV1,
) -> None:
    if not isinstance(value, RuntimeStagingColorAttestedDecodedBatchV1):
        raise ReferenceMatchContractError(
            "runtime staging colour-attested batch type is invalid"
        )
    validate_runtime_staging_color_attestation_record_v1(value.record)
    validate_runtime_qualified_shared_staging_decoded_batch_v1(value.decoded)
    if (
        value.record.decode_id != value.decoded.record.decode_id
        or value.record.consumption_id
        != value.decoded.record.consumption_id
        or value.record.run_id != value.decoded.record.run_id
        or value.record.source_count != value.decoded.record.source_count
    ):
        raise ReferenceMatchContractError(
            "runtime staging colour attestation binding mismatch"
        )
    expected = srgb_icc_profile_v1()
    for attested, decoded_row, raw in zip(
        value.record.outputs,
        value.decoded.record.outputs,
        value.decoded.snapshot.output_bytes,
        strict=True,
    ):
        profile, binding, orientation = _extract_profile(
            raw,
            format_name=decoded_row.output_format,
            bit_depth=decoded_row.output_bit_depth,
        )
        if (
            attested.source_index != decoded_row.source_index
            or attested.source_view_id != decoded_row.source_view_id
            or attested.output_view_id != decoded_row.output_view_id
            or attested.encoded_file_sha256
            != decoded_row.encoded_file_sha256
            or attested.output_format != decoded_row.output_format
            or attested.output_bit_depth != decoded_row.output_bit_depth
            or attested.profile_binding != binding
            or attested.orientation != orientation
            or profile != expected
            or hashlib.sha256(profile).hexdigest()
            != attested.profile_sha256
        ):
            raise ReferenceMatchContractError(
                "runtime staging colour attestation does not match bytes"
            )


def runtime_staging_color_attestation_record_to_json(
    value: RuntimeStagingColorAttestationRecordV1,
) -> str:
    validate_runtime_staging_color_attestation_record_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_staging_color_attestation_record_from_json(
    encoded: str,
) -> RuntimeStagingColorAttestationRecordV1:
    try:
        payload = strict_json_loads(encoded)
    except (
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ReferenceMatchContractError(
            "runtime staging colour attestation is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "runtime staging colour attestation")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime staging colour attestation outputs must be non-empty"
        )
    outputs: list[AttestedRuntimeStagingColorOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"colour attestation {index}")
        try:
            outputs.append(
                AttestedRuntimeStagingColorOutputV1(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime staging colour attestation fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeStagingColorAttestationRecordV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging colour attestation fields are invalid"
        ) from exc
    validate_runtime_staging_color_attestation_record_v1(result)
    return result


__all__ = [
    "AttestedRuntimeStagingColorOutputV1",
    "MAX_COMPRESSED_PROFILE_BYTES",
    "MAX_EMBEDDED_PROFILE_BYTES",
    "RUNTIME_STAGING_COLOR_ATTESTATION_CLAIM_CEILING",
    "RUNTIME_STAGING_COLOR_ATTESTATION_POLICY_ID",
    "RUNTIME_STAGING_COLOR_ATTESTATION_SCHEMA_ID",
    "RuntimeStagingColorAttestationRecordV1",
    "RuntimeStagingColorAttestedDecodedBatchV1",
    "attest_runtime_staging_srgb_metadata_v1",
    "runtime_staging_color_attestation_record_from_json",
    "runtime_staging_color_attestation_record_to_json",
    "validate_runtime_staging_color_attestation_record_v1",
    "validate_runtime_staging_color_attested_decoded_batch_v1",
]
