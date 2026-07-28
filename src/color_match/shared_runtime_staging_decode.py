"""Strict path-free decoding of one P65 immutable encoded-byte snapshot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import hashlib
import json
import re
import struct
import warnings
import zlib
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile

from .strict_json import strict_json_loads
from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError
from .shared_runtime_staging_consumption import (
    MAX_CAPTURED_OUTPUT_BYTES,
    RuntimeQualifiedSharedStagingByteSnapshotV1,
    validate_runtime_qualified_shared_staging_byte_snapshot_v1,
)


RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_SCHEMA_ID = (
    "neuro-film.runtime-qualified-shared-staging-decode.v1"
)
RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_CLAIM_CEILING = (
    "process-local-decoded-sample-evidence-only-"
    "no-path-persistence-or-delivery"
)
MAX_DECODED_OUTPUT_PIXELS = 128 * 1024 * 1024
MAX_DECODED_AGGREGATE_PIXELS = 256 * 1024 * 1024
MAX_DECODED_DIMENSION = 32768
_STATE = "decoded-runtime-qualified-shared-staging-samples"
_SCOPE = "current-process-readonly-rgb-sample-arrays"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {
    "schema_id",
    "decode_id",
    "consumption_id",
    "verification_id",
    "run_id",
    "source_count",
    "state",
    "decoded_scope",
    "path_consumption_authorized",
    "persistent_pixels_authorized",
    "delivery_authorized",
    "outputs",
    "claim_ceiling",
}
_OUTPUT_KEYS = {
    "source_index",
    "source_view_id",
    "output_view_id",
    "encoded_file_sha256",
    "encoded_file_size_bytes",
    "output_format",
    "output_bit_depth",
    "width",
    "height",
    "channels",
    "sample_dtype",
    "decoded_pixel_sha256",
}


@dataclass(frozen=True)
class DecodedRuntimeQualifiedSharedStagingOutputV1:
    source_index: int
    source_view_id: str
    output_view_id: str
    encoded_file_sha256: str
    encoded_file_size_bytes: int
    output_format: str
    output_bit_depth: int
    width: int
    height: int
    channels: int
    sample_dtype: str
    decoded_pixel_sha256: str


@dataclass(frozen=True)
class RuntimeQualifiedSharedStagingDecodeRecordV1:
    schema_id: str
    decode_id: str
    consumption_id: str
    verification_id: str
    run_id: str
    source_count: int
    state: str
    decoded_scope: str
    path_consumption_authorized: bool
    persistent_pixels_authorized: bool
    delivery_authorized: bool
    outputs: tuple[DecodedRuntimeQualifiedSharedStagingOutputV1, ...]
    claim_ceiling: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [asdict(output) for output in self.outputs]
        return payload


@dataclass(frozen=True)
class RuntimeQualifiedSharedStagingDecodedBatchV1:
    record: RuntimeQualifiedSharedStagingDecodeRecordV1
    snapshot: RuntimeQualifiedSharedStagingByteSnapshotV1
    pixels: tuple[np.ndarray, ...]


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReferenceMatchContractError(
            f"{label} fields differ from the decode contract"
        )
    return value


def _identity_payload(
    value: RuntimeQualifiedSharedStagingDecodeRecordV1,
) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("decode_id")
    return payload


def _pixel_sha256(array: np.ndarray) -> str:
    if array.dtype == np.uint8:
        wire = np.ascontiguousarray(array).tobytes()
    elif array.dtype == np.uint16:
        wire = np.ascontiguousarray(array.astype(">u2", copy=False)).tobytes()
    else:
        raise ReferenceMatchContractError(
            "decoded samples must be uint8 or uint16"
        )
    return hashlib.sha256(wire).hexdigest()


def _parse_strict_png(
    raw: bytes,
    expected_depth: int,
) -> tuple[int, int]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(signature):
        raise ReferenceMatchContractError("encoded PNG signature is invalid")
    offset = len(signature)
    seen_ihdr = False
    seen_iend = False
    seen_idat = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise ReferenceMatchContractError("encoded PNG chunk is truncated")
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(raw):
            raise ReferenceMatchContractError("encoded PNG chunk is truncated")
        payload = raw[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", raw[end - 4 : end])[0]
        if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            raise ReferenceMatchContractError("encoded PNG CRC is invalid")
        if not seen_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise ReferenceMatchContractError(
                    "encoded PNG must begin with one IHDR"
                )
            width, height, depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (
                width <= 0
                or height <= 0
                or width > MAX_DECODED_DIMENSION
                or height > MAX_DECODED_DIMENSION
                or width * height > MAX_DECODED_OUTPUT_PIXELS
            ):
                raise ReferenceMatchContractError(
                    "encoded PNG exceeds decoded geometry budget"
                )
            if (
                depth != expected_depth
                or color_type != 2
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ReferenceMatchContractError(
                    "encoded PNG sample contract is invalid"
                )
            seen_ihdr = True
        elif chunk_type == b"IHDR":
            raise ReferenceMatchContractError("encoded PNG has duplicate IHDR")
        if chunk_type in {b"acTL", b"fcTL", b"fdAT"}:
            raise ReferenceMatchContractError(
                "animated PNG is not supported"
            )
        if chunk_type == b"IDAT":
            seen_idat = True
        if (
            chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
            and chunk_type[:1].isupper()
        ):
            raise ReferenceMatchContractError(
                "encoded PNG has an unsupported critical chunk"
            )
        if chunk_type == b"IEND":
            if length != 0 or seen_iend or end != len(raw):
                raise ReferenceMatchContractError(
                    "encoded PNG IEND/trailing bytes are invalid"
                )
            seen_iend = True
        offset = end
    if not seen_ihdr or not seen_idat or not seen_iend:
        raise ReferenceMatchContractError("encoded PNG is incomplete")
    return width, height


def _decode_png(raw: bytes, expected_depth: int) -> np.ndarray:
    _parse_strict_png(raw, expected_depth)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    array = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    expected_dtype = np.uint8 if expected_depth == 8 else np.uint16
    if (
        array is None
        or array.dtype != expected_dtype
        or array.ndim != 3
        or array.shape[2] != 3
    ):
        raise ReferenceMatchContractError(
            "encoded PNG does not decode to exact RGB samples"
        )
    return np.ascontiguousarray(array[..., ::-1])


def _decode_jpeg(raw: bytes, expected_depth: int) -> np.ndarray:
    _preflight_jpeg(raw, expected_depth)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            array = np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ReferenceMatchContractError(
            "encoded JPEG cannot be decoded safely"
        ) from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ReferenceMatchContractError(
            "encoded JPEG does not decode to RGB samples"
        )
    return np.ascontiguousarray(array)


def _preflight_jpeg(
    raw: bytes,
    expected_depth: int,
) -> tuple[int, int]:
    if (
        expected_depth != 8
        or not raw.startswith(b"\xff\xd8")
        or not raw.endswith(b"\xff\xd9")
        or raw.find(b"\xff\xd9") != len(raw) - 2
    ):
        raise ReferenceMatchContractError(
            "encoded JPEG boundary/depth is invalid"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if (
                    image.format != "JPEG"
                    or getattr(image, "n_frames", 1) != 1
                    or image.mode != "RGB"
                ):
                    raise ReferenceMatchContractError(
                        "encoded JPEG must be one RGB frame"
                    )
                width, height = image.size
                if width * height > MAX_DECODED_OUTPUT_PIXELS:
                    raise ReferenceMatchContractError(
                        "encoded image exceeds decoded pixel budget"
                    )
    except ReferenceMatchContractError:
        raise
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ReferenceMatchContractError(
            "encoded JPEG cannot be decoded safely"
        ) from exc
    return width, height


def _decode_tiff(raw: bytes, expected_depth: int) -> np.ndarray:
    _preflight_tiff(raw, expected_depth)
    try:
        with tifffile.TiffFile(BytesIO(raw)) as tif:
            array = tif.pages[0].asarray()
    except (OSError, ValueError, TypeError, tifffile.TiffFileError) as exc:
        raise ReferenceMatchContractError(
            "encoded TIFF cannot be decoded safely"
        ) from exc
    expected_dtype = np.uint8 if expected_depth == 8 else np.uint16
    if (
        array.dtype != expected_dtype
        or array.ndim != 3
        or array.shape[2] != 3
    ):
        raise ReferenceMatchContractError(
            "encoded TIFF does not decode to exact RGB samples"
        )
    return np.ascontiguousarray(array)


def _preflight_tiff(
    raw: bytes,
    expected_depth: int,
) -> tuple[int, int]:
    try:
        with tifffile.TiffFile(BytesIO(raw)) as tif:
            if len(tif.pages) != 1:
                raise ReferenceMatchContractError(
                    "encoded TIFF must contain exactly one page"
                )
            page = tif.pages[0]
            if (
                int(page.samplesperpixel) != 3
                or int(page.bitspersample) != expected_depth
                or int(page.photometric) != 2
                or (
                    page.planarconfig is not None
                    and int(page.planarconfig) != 1
                )
            ):
                raise ReferenceMatchContractError(
                    "encoded TIFF sample contract is invalid"
                )
            shape = tuple(int(value) for value in page.shape)
            if (
                len(shape) != 3
                or shape[2] != 3
                or shape[0] <= 0
                or shape[1] <= 0
                or shape[0] > MAX_DECODED_DIMENSION
                or shape[1] > MAX_DECODED_DIMENSION
                or shape[0] * shape[1] > MAX_DECODED_OUTPUT_PIXELS
            ):
                raise ReferenceMatchContractError(
                    "encoded TIFF exceeds decoded geometry budget"
                )
            data_end = max(
                int(offset) + int(count)
                for offset, count in zip(
                    page.dataoffsets,
                    page.databytecounts,
                    strict=True,
                )
            )
            if data_end != len(raw):
                raise ReferenceMatchContractError(
                    "encoded TIFF has trailing or unbound bytes"
                )
    except ReferenceMatchContractError:
        raise
    except (OSError, ValueError, TypeError, tifffile.TiffFileError) as exc:
        raise ReferenceMatchContractError(
            "encoded TIFF cannot be decoded safely"
        ) from exc
    return shape[1], shape[0]


def _preflight_one(
    raw: bytes,
    format_name: str,
    depth: int,
) -> tuple[int, int]:
    if format_name == "PNG":
        return _parse_strict_png(raw, depth)
    if format_name == "JPEG":
        return _preflight_jpeg(raw, depth)
    if format_name == "TIFF":
        return _preflight_tiff(raw, depth)
    raise ReferenceMatchContractError("encoded output format is unsupported")


def _decode_one(raw: bytes, format_name: str, depth: int) -> np.ndarray:
    if format_name == "PNG":
        return _decode_png(raw, depth)
    if format_name == "JPEG":
        return _decode_jpeg(raw, depth)
    if format_name == "TIFF":
        return _decode_tiff(raw, depth)
    raise ReferenceMatchContractError("encoded output format is unsupported")


def decode_runtime_qualified_shared_staging_bytes_v1(
    snapshot: RuntimeQualifiedSharedStagingByteSnapshotV1,
) -> RuntimeQualifiedSharedStagingDecodedBatchV1:
    validate_runtime_qualified_shared_staging_byte_snapshot_v1(snapshot)
    preflight_geometry: list[tuple[int, int]] = []
    aggregate_pixels = 0
    for verified, raw in zip(
        snapshot.verification.outputs,
        snapshot.output_bytes,
        strict=True,
    ):
        width, height = _preflight_one(
            raw,
            verified.output_format,
            verified.output_bit_depth,
        )
        aggregate_pixels += width * height
        if aggregate_pixels > MAX_DECODED_AGGREGATE_PIXELS:
            raise ReferenceMatchContractError(
                "encoded batch exceeds decoded aggregate pixel budget"
            )
        preflight_geometry.append((width, height))
    decoded_arrays: list[np.ndarray] = []
    outputs: list[DecodedRuntimeQualifiedSharedStagingOutputV1] = []
    for verified, raw, expected_geometry in zip(
        snapshot.verification.outputs,
        snapshot.output_bytes,
        preflight_geometry,
        strict=True,
    ):
        try:
            array = _decode_one(
                raw,
                verified.output_format,
                verified.output_bit_depth,
            )
        except ReferenceMatchContractError:
            raise
        except Exception as exc:
            raise ReferenceMatchContractError(
                "encoded output decoder failed closed"
            ) from exc
        height, width, channels = array.shape
        pixels = width * height
        if (
            width <= 0
            or height <= 0
            or width > MAX_DECODED_DIMENSION
            or height > MAX_DECODED_DIMENSION
            or pixels > MAX_DECODED_OUTPUT_PIXELS
            or (width, height) != expected_geometry
        ):
            raise ReferenceMatchContractError(
                "encoded image exceeds decoded geometry budget"
            )
        array.flags.writeable = False
        decoded_arrays.append(array)
        outputs.append(
            DecodedRuntimeQualifiedSharedStagingOutputV1(
                source_index=verified.source_index,
                source_view_id=verified.source_view_id,
                output_view_id=verified.output_view_id,
                encoded_file_sha256=verified.output_file_sha256,
                encoded_file_size_bytes=verified.output_file_size_bytes,
                output_format=verified.output_format,
                output_bit_depth=verified.output_bit_depth,
                width=width,
                height=height,
                channels=channels,
                sample_dtype=(
                    "uint8" if array.dtype == np.uint8 else "uint16"
                ),
                decoded_pixel_sha256=_pixel_sha256(array),
            )
        )
    provisional = RuntimeQualifiedSharedStagingDecodeRecordV1(
        schema_id=RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_SCHEMA_ID,
        decode_id="0" * 64,
        consumption_id=snapshot.record.consumption_id,
        verification_id=snapshot.verification.verification_id,
        run_id=snapshot.verification.run_id,
        source_count=snapshot.record.source_count,
        state=_STATE,
        decoded_scope=_SCOPE,
        path_consumption_authorized=False,
        persistent_pixels_authorized=False,
        delivery_authorized=False,
        outputs=tuple(outputs),
        claim_ceiling=(
            RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_CLAIM_CEILING
        ),
    )
    record = replace(
        provisional,
        decode_id=canonical_sha256(_identity_payload(provisional)),
    )
    result = RuntimeQualifiedSharedStagingDecodedBatchV1(
        record=record,
        snapshot=snapshot,
        pixels=tuple(decoded_arrays),
    )
    validate_runtime_qualified_shared_staging_decoded_batch_v1(result)
    return result


def validate_runtime_qualified_shared_staging_decode_record_v1(
    value: RuntimeQualifiedSharedStagingDecodeRecordV1,
) -> None:
    if not isinstance(value, RuntimeQualifiedSharedStagingDecodeRecordV1):
        raise ReferenceMatchContractError(
            "runtime staging decode record type is invalid"
        )
    if value.schema_id != RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "runtime staging decode schema is invalid"
        )
    for field in (
        "decode_id",
        "consumption_id",
        "verification_id",
        "run_id",
    ):
        _hash(getattr(value, field), field)
    if (
        value.state != _STATE
        or value.decoded_scope != _SCOPE
        or value.path_consumption_authorized is not False
        or value.persistent_pixels_authorized is not False
        or value.delivery_authorized is not False
        or value.claim_ceiling
        != RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_CLAIM_CEILING
        or isinstance(value.source_count, bool)
        or not isinstance(value.source_count, int)
        or value.source_count <= 0
        or value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or value.source_count != len(value.outputs)
    ):
        raise ReferenceMatchContractError(
            "runtime staging decode authority/metadata is invalid"
        )
    aggregate = 0
    source_ids: set[str] = set()
    view_ids: set[str] = set()
    for index, output in enumerate(value.outputs):
        if (
            not isinstance(
                output,
                DecodedRuntimeQualifiedSharedStagingOutputV1,
            )
            or output.source_index != index
            or output.output_format not in {"PNG", "JPEG", "TIFF"}
            or output.output_bit_depth not in {8, 16}
            or (
                output.output_format == "JPEG"
                and output.output_bit_depth != 8
            )
            or output.channels != 3
            or output.sample_dtype
            != ("uint8" if output.output_bit_depth == 8 else "uint16")
        ):
            raise ReferenceMatchContractError(
                "runtime staging decoded output is invalid"
            )
        for field in (
            "source_view_id",
            "output_view_id",
            "encoded_file_sha256",
            "decoded_pixel_sha256",
        ):
            _hash(getattr(output, field), f"output.{field}")
        if (
            isinstance(output.width, bool)
            or isinstance(output.height, bool)
            or not isinstance(output.width, int)
            or not isinstance(output.height, int)
            or output.width <= 0
            or output.height <= 0
            or output.width > MAX_DECODED_DIMENSION
            or output.height > MAX_DECODED_DIMENSION
            or output.width * output.height > MAX_DECODED_OUTPUT_PIXELS
            or isinstance(output.encoded_file_size_bytes, bool)
            or not isinstance(output.encoded_file_size_bytes, int)
            or output.encoded_file_size_bytes <= 0
            or output.encoded_file_size_bytes > MAX_CAPTURED_OUTPUT_BYTES
        ):
            raise ReferenceMatchContractError(
                "runtime staging decoded geometry/size is invalid"
            )
        aggregate += output.width * output.height
        if aggregate > MAX_DECODED_AGGREGATE_PIXELS:
            raise ReferenceMatchContractError(
                "runtime staging decoded batch exceeds pixel budget"
            )
        if (
            output.source_view_id in source_ids
            or output.output_view_id in view_ids
        ):
            raise ReferenceMatchContractError(
                "runtime staging decoded identities must be distinct"
            )
        source_ids.add(output.source_view_id)
        view_ids.add(output.output_view_id)
    if value.decode_id != canonical_sha256(_identity_payload(value)):
        raise ReferenceMatchContractError(
            "runtime staging decode identity mismatch"
        )


def validate_runtime_qualified_shared_staging_decoded_batch_v1(
    value: RuntimeQualifiedSharedStagingDecodedBatchV1,
) -> None:
    if not isinstance(value, RuntimeQualifiedSharedStagingDecodedBatchV1):
        raise ReferenceMatchContractError(
            "runtime staging decoded batch type is invalid"
        )
    validate_runtime_qualified_shared_staging_decode_record_v1(value.record)
    validate_runtime_qualified_shared_staging_byte_snapshot_v1(
        value.snapshot
    )
    if (
        value.record.consumption_id != value.snapshot.record.consumption_id
        or value.record.verification_id
        != value.snapshot.verification.verification_id
        or value.record.run_id != value.snapshot.verification.run_id
        or value.record.source_count != len(value.pixels)
    ):
        raise ReferenceMatchContractError(
            "runtime staging decoded batch binding mismatch"
        )
    for record, verified, array in zip(
        value.record.outputs,
        value.snapshot.verification.outputs,
        value.pixels,
        strict=True,
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.flags.writeable
            or not array.flags.c_contiguous
            or array.shape != (record.height, record.width, record.channels)
            or record.source_index != verified.source_index
            or record.source_view_id != verified.source_view_id
            or record.output_view_id != verified.output_view_id
            or record.encoded_file_sha256
            != verified.output_file_sha256
            or record.encoded_file_size_bytes
            != verified.output_file_size_bytes
            or record.output_format != verified.output_format
            or record.output_bit_depth != verified.output_bit_depth
            or _pixel_sha256(array) != record.decoded_pixel_sha256
        ):
            raise ReferenceMatchContractError(
                "runtime staging decoded pixels do not match their binding"
            )


def runtime_qualified_shared_staging_decode_record_to_json(
    value: RuntimeQualifiedSharedStagingDecodeRecordV1,
) -> str:
    validate_runtime_qualified_shared_staging_decode_record_v1(value)
    return json.dumps(
        value.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def runtime_qualified_shared_staging_decode_record_from_json(
    encoded: str,
) -> RuntimeQualifiedSharedStagingDecodeRecordV1:
    try:
        payload = strict_json_loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging decode record is not valid JSON"
        ) from exc
    payload = _strict(payload, _KEYS, "runtime staging decode record")
    raw_outputs = payload["outputs"]
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ReferenceMatchContractError(
            "runtime staging decode outputs must be non-empty"
        )
    outputs: list[DecodedRuntimeQualifiedSharedStagingOutputV1] = []
    for index, raw in enumerate(raw_outputs):
        raw = _strict(raw, _OUTPUT_KEYS, f"decoded output {index}")
        try:
            outputs.append(
                DecodedRuntimeQualifiedSharedStagingOutputV1(**dict(raw))
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceMatchContractError(
                "runtime staging decoded output fields are invalid"
            ) from exc
    converted = dict(payload)
    converted["outputs"] = tuple(outputs)
    try:
        result = RuntimeQualifiedSharedStagingDecodeRecordV1(**converted)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "runtime staging decode record fields are invalid"
        ) from exc
    validate_runtime_qualified_shared_staging_decode_record_v1(result)
    return result


__all__ = [
    "DecodedRuntimeQualifiedSharedStagingOutputV1",
    "MAX_DECODED_AGGREGATE_PIXELS",
    "MAX_DECODED_DIMENSION",
    "MAX_DECODED_OUTPUT_PIXELS",
    "RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_CLAIM_CEILING",
    "RUNTIME_QUALIFIED_SHARED_STAGING_DECODE_SCHEMA_ID",
    "RuntimeQualifiedSharedStagingDecodeRecordV1",
    "RuntimeQualifiedSharedStagingDecodedBatchV1",
    "decode_runtime_qualified_shared_staging_bytes_v1",
    "runtime_qualified_shared_staging_decode_record_from_json",
    "runtime_qualified_shared_staging_decode_record_to_json",
    "validate_runtime_qualified_shared_staging_decode_record_v1",
    "validate_runtime_qualified_shared_staging_decoded_batch_v1",
]
