"""Fail-closed metadata policy for advertised reference-match file outputs."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from pathlib import Path
from typing import Any
import zlib

import tifffile

from src.inference.render_contract import sha256_file
from src.preprocess.color_management import REC2020_SDR_CICP
from src.preprocess.output_encode import srgb_icc_profile

from .contracts import ReferenceMatchContractError


REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID = (
    "neuro-film.reference-file-output-metadata-policy.v1"
)
REFERENCE_FILE_OUTPUT_METADATA_POLICY_CLAIM_CEILING = (
    "advertised-reference-file-outputs-only-no-source-metadata-copy-"
    "not-arbitrary-encoder-or-third-party-metadata-sanitization"
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IO_CHUNK_BYTES = 1024 * 1024
_PNG_MAX_RETAINED_METADATA_BYTES = 1024 * 1024
_PNG_STRUCTURAL = frozenset({b"IHDR", b"IDAT", b"IEND"})
_PNG_PROFILE_CHUNKS = {
    "srgb-icc.v1": (b"iCCP",),
    "bt2020-sdr-cicp-1-1-0-1.v1": (b"cICP",),
}
_JPEG_ALLOWED_APP_MARKERS = frozenset({0xE0, 0xE2})
_JPEG_JFIF_V1_PAYLOAD = (
    b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
)
_TIFF_ALLOWED_TAGS = frozenset(
    {
        256,  # ImageWidth
        257,  # ImageLength
        258,  # BitsPerSample
        259,  # Compression
        262,  # PhotometricInterpretation
        273,  # StripOffsets
        277,  # SamplesPerPixel
        278,  # RowsPerStrip
        279,  # StripByteCounts
        282,  # XResolution
        283,  # YResolution
        284,  # PlanarConfiguration
        296,  # ResolutionUnit
        305,  # Encoder Software, generated here rather than copied
        34675,  # ICC profile
    }
)


@dataclass(frozen=True)
class ReferenceFileOutputMetadataAttestation:
    """One exact metadata-minimization decision for a rendered file."""

    schema_id: str
    path: Path
    file_sha256: str | None
    format_name: str
    encoding_profile: str
    orientation: int
    observed_metadata: tuple[str, ...]
    sensitive_metadata_absent: bool
    accepted: bool
    failure_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "path": str(self.path.resolve(strict=False)),
            "file_sha256": self.file_sha256,
            "format_name": self.format_name,
            "encoding_profile": self.encoding_profile,
            "orientation": self.orientation,
            "observed_metadata": list(self.observed_metadata),
            "sensitive_metadata_absent": self.sensitive_metadata_absent,
            "accepted": self.accepted,
            "failure_code": self.failure_code,
        }


def reference_file_output_metadata_policy_payload() -> dict[str, Any]:
    """Return the immutable policy independently layered over v1 capabilities."""

    return {
        "schema_id": REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID,
        "source_metadata_copy": "none",
        "orientation": "pixels-normalized-no-nonidentity-tag",
        "forbidden_metadata_classes": [
            "exif",
            "gps",
            "iptc",
            "xmp",
            "source-free-text",
            "embedded-thumbnail",
            "application-private",
        ],
        "allowed_color_bindings": [
            "srgb-icc.v1",
            "bt2020-sdr-cicp-1-1-0-1.v1",
        ],
        "claim_ceiling": REFERENCE_FILE_OUTPUT_METADATA_POLICY_CLAIM_CEILING,
    }


def _read_exact(handle, size: int, label: str) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise ReferenceMatchContractError(f"{label} is truncated")
    return payload


def _png_chunks(path: Path) -> tuple[tuple[bytes, bytes], ...]:
    rows: list[tuple[bytes, bytes]] = []
    with path.open("rb") as handle:
        if _read_exact(handle, len(_PNG_SIGNATURE), "PNG signature") != _PNG_SIGNATURE:
            raise ReferenceMatchContractError("output is not a PNG stream")
        while True:
            header = _read_exact(handle, 8, "PNG chunk header")
            length = struct.unpack(">I", header[:4])[0]
            chunk_type = header[4:]
            retain = chunk_type in _PNG_PROFILE_CHUNKS["srgb-icc.v1"] or (
                chunk_type
                in _PNG_PROFILE_CHUNKS["bt2020-sdr-cicp-1-1-0-1.v1"]
            )
            if retain and length > _PNG_MAX_RETAINED_METADATA_BYTES:
                raise ReferenceMatchContractError(
                    "PNG color-binding chunk exceeds the metadata bound"
                )
            retained = bytearray()
            checksum = zlib.crc32(chunk_type)
            remaining = length
            while remaining:
                block = _read_exact(
                    handle,
                    min(remaining, _PNG_IO_CHUNK_BYTES),
                    "PNG chunk payload",
                )
                checksum = zlib.crc32(block, checksum)
                if retain:
                    retained.extend(block)
                remaining -= len(block)
            expected_crc = struct.unpack(
                ">I",
                _read_exact(handle, 4, "PNG chunk CRC"),
            )[0]
            if checksum & 0xFFFFFFFF != expected_crc:
                raise ReferenceMatchContractError("PNG chunk CRC is invalid")
            rows.append((chunk_type, bytes(retained)))
            if chunk_type == b"IEND":
                if handle.read(1):
                    raise ReferenceMatchContractError(
                        "PNG stream has trailing data"
                    )
                break
    if not rows or rows[-1][0] != b"IEND":
        raise ReferenceMatchContractError("PNG stream is missing IEND")
    return tuple(rows)


def _inspect_png(path: Path, encoding_profile: str) -> tuple[int, tuple[str, ...]]:
    try:
        required_profile = _PNG_PROFILE_CHUNKS[encoding_profile]
    except KeyError as exc:
        raise ReferenceMatchContractError(
            "PNG metadata policy does not recognize encoding_profile"
        ) from exc
    rows = _png_chunks(path)
    types = tuple(chunk_type for chunk_type, _payload in rows)
    allowed = _PNG_STRUCTURAL | frozenset(required_profile)
    if (
        types[0] != b"IHDR"
        or types[-1] != b"IEND"
        or types.count(b"IHDR") != 1
        or types.count(b"IEND") != 1
        or b"IDAT" not in types
    ):
        raise ReferenceMatchContractError(
            "PNG structural chunk inventory is invalid"
        )
    if any(chunk_type not in allowed for chunk_type in types):
        raise ReferenceMatchContractError(
            "PNG contains non-policy ancillary metadata"
        )
    if any(types.count(chunk_type) != 1 for chunk_type in required_profile):
        raise ReferenceMatchContractError(
            "PNG color-binding chunk is missing or duplicated"
        )
    profile_index = types.index(required_profile[0])
    first_idat = types.index(b"IDAT")
    last_idat = len(types) - 1 - types[::-1].index(b"IDAT")
    if (
        profile_index >= first_idat
        or any(chunk_type != b"IDAT" for chunk_type in types[first_idat:last_idat + 1])
    ):
        raise ReferenceMatchContractError(
            "PNG color binding or IDAT ordering is invalid"
        )
    if encoding_profile == "srgb-icc.v1":
        profile_payload = next(
            payload for chunk_type, payload in rows if chunk_type == b"iCCP"
        )
        expected_profile = srgb_icc_profile()
        try:
            name, method_and_compressed = profile_payload.split(b"\x00", 1)
            method = method_and_compressed[0]
            decoder = zlib.decompressobj()
            profile = decoder.decompress(
                method_and_compressed[1:],
                len(expected_profile) + 1,
            )
        except (IndexError, ValueError, zlib.error) as exc:
            raise ReferenceMatchContractError(
                "PNG ICC payload is invalid"
            ) from exc
        if (
            not name
            or method != 0
            or profile != expected_profile
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ReferenceMatchContractError(
                "PNG ICC payload does not match srgb-icc.v1"
            )
    else:
        cicp = next(
            payload for chunk_type, payload in rows if chunk_type == b"cICP"
        )
        if cicp != REC2020_SDR_CICP:
            raise ReferenceMatchContractError(
                "PNG cICP payload does not match the advertised profile"
            )
    return 1, tuple(chunk_type.decode("ascii") for chunk_type in types)


def _jpeg_segments(payload: bytes) -> tuple[tuple[int, bytes], ...]:
    if not payload.startswith(b"\xff\xd8"):
        raise ReferenceMatchContractError("output is not a JPEG stream")
    rows: list[tuple[int, bytes]] = []
    offset = 2
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ReferenceMatchContractError("JPEG marker prefix is invalid")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise ReferenceMatchContractError("JPEG marker is truncated")
        marker = payload[offset]
        offset += 1
        if marker == 0xD9:
            rows.append((marker, b""))
            if offset != len(payload):
                raise ReferenceMatchContractError("JPEG has trailing data")
            return tuple(rows)
        if marker == 0xDA:
            if offset + 2 > len(payload):
                raise ReferenceMatchContractError("JPEG SOS is truncated")
            length = struct.unpack(">H", payload[offset : offset + 2])[0]
            if length < 2 or offset + length > len(payload):
                raise ReferenceMatchContractError("JPEG SOS length is invalid")
            rows.append((marker, payload[offset + 2 : offset + length]))
            scan = offset + length
            while scan + 1 < len(payload):
                if payload[scan] != 0xFF:
                    scan += 1
                    continue
                code = payload[scan + 1]
                if code == 0x00 or 0xD0 <= code <= 0xD7:
                    scan += 2
                    continue
                offset = scan
                break
            else:
                raise ReferenceMatchContractError("JPEG EOI is missing")
            continue
        if marker in {0x01, *range(0xD0, 0xD8)}:
            rows.append((marker, b""))
            continue
        if offset + 2 > len(payload):
            raise ReferenceMatchContractError("JPEG segment is truncated")
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        end = offset + length
        if length < 2 or end > len(payload):
            raise ReferenceMatchContractError("JPEG segment length is invalid")
        rows.append((marker, payload[offset + 2 : end]))
        offset = end
    raise ReferenceMatchContractError("JPEG EOI is missing")


def _inspect_jpeg(path: Path, encoding_profile: str) -> tuple[int, tuple[str, ...]]:
    if encoding_profile != "srgb-icc.v1":
        raise ReferenceMatchContractError(
            "JPEG metadata policy requires srgb-icc.v1"
        )
    segments = _jpeg_segments(path.read_bytes())
    if any(marker == 0xFE for marker, _payload in segments):
        raise ReferenceMatchContractError(
            "JPEG contains a free-form comment"
        )
    app_rows = [(marker, payload) for marker, payload in segments if 0xE0 <= marker <= 0xEF]
    if any(marker not in _JPEG_ALLOWED_APP_MARKERS for marker, _ in app_rows):
        raise ReferenceMatchContractError(
            "JPEG contains non-policy application metadata"
        )
    jfif_rows = [
        payload for marker, payload in app_rows if marker == 0xE0
    ]
    if jfif_rows != [_JPEG_JFIF_V1_PAYLOAD]:
        raise ReferenceMatchContractError(
            "JPEG must contain one exact thumbnail-free JFIF APP0"
        )
    icc_parts: dict[int, bytes] = {}
    icc_count: int | None = None
    for marker, payload in app_rows:
        if marker != 0xE2:
            continue
        if not payload.startswith(b"ICC_PROFILE\x00") or len(payload) < 14:
            raise ReferenceMatchContractError("JPEG APP2 is not ICC metadata")
        sequence, count = payload[12], payload[13]
        if sequence == 0 or count == 0 or sequence > count:
            raise ReferenceMatchContractError("JPEG ICC ordering is invalid")
        if icc_count not in {None, count} or sequence in icc_parts:
            raise ReferenceMatchContractError("JPEG ICC sequence is inconsistent")
        icc_count = count
        icc_parts[sequence] = payload[14:]
    if icc_count is None or set(icc_parts) != set(range(1, icc_count + 1)):
        raise ReferenceMatchContractError("JPEG ICC sequence is incomplete")
    profile = b"".join(icc_parts[index] for index in range(1, icc_count + 1))
    if profile != srgb_icc_profile():
        raise ReferenceMatchContractError(
            "JPEG ICC payload does not match srgb-icc.v1"
        )
    observed = tuple(
        "APP0-JFIF" if marker == 0xE0 else "APP2-ICC"
        for marker, _payload in app_rows
    )
    return 1, observed


def _inspect_tiff(path: Path, encoding_profile: str) -> tuple[int, tuple[str, ...]]:
    if encoding_profile != "srgb-icc.v1":
        raise ReferenceMatchContractError(
            "TIFF metadata policy requires srgb-icc.v1"
        )
    try:
        with tifffile.TiffFile(path) as document:
            if len(document.pages) != 1:
                raise ReferenceMatchContractError(
                    "TIFF output must contain exactly one page"
                )
            page = document.pages[0]
            tag_codes = tuple(sorted(int(tag.code) for tag in page.tags.values()))
            if any(code not in _TIFF_ALLOWED_TAGS for code in tag_codes):
                raise ReferenceMatchContractError(
                    "TIFF contains non-policy metadata"
                )
            orientation_tag = page.tags.get(274)
            orientation = (
                int(orientation_tag.value)
                if orientation_tag is not None
                else 1
            )
            if orientation != 1:
                raise ReferenceMatchContractError(
                    "TIFF orientation must be absent or identity"
                )
            profile_tag = page.tags.get(34675)
            if profile_tag is None or bytes(profile_tag.value) != srgb_icc_profile():
                raise ReferenceMatchContractError(
                    "TIFF ICC payload does not match srgb-icc.v1"
                )
            software_tag = page.tags.get(305)
            if (
                software_tag is not None
                and software_tag.value != "tifffile.py"
            ):
                raise ReferenceMatchContractError(
                    "TIFF encoder Software tag is not the policy constant"
                )
    except ReferenceMatchContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - parser failures are one boundary.
        raise ReferenceMatchContractError(
            "TIFF metadata cannot be inspected"
        ) from exc
    return orientation, tuple(f"TIFF-{code}" for code in tag_codes)


def attest_reference_file_output_metadata(
    path: Path | str,
    *,
    encoding_profile: str,
) -> ReferenceFileOutputMetadataAttestation:
    """Inspect one already-rendered output without modifying it."""

    source = Path(path)
    if not source.is_file():
        raise ReferenceMatchContractError(
            "output metadata attestation requires an existing file"
        )
    try:
        before = sha256_file(source)
    except OSError as exc:
        raise ReferenceMatchContractError(
            "output metadata attestation cannot hash the file"
        ) from exc
    suffix = source.suffix.casefold()
    try:
        if suffix == ".png":
            format_name = "PNG"
            orientation, observed = _inspect_png(source, encoding_profile)
        elif suffix in {".jpg", ".jpeg"}:
            format_name = "JPEG"
            orientation, observed = _inspect_jpeg(source, encoding_profile)
        elif suffix in {".tif", ".tiff"}:
            format_name = "TIFF"
            orientation, observed = _inspect_tiff(source, encoding_profile)
        else:
            raise ReferenceMatchContractError(
                "output metadata attestation does not support this extension"
            )
        after = sha256_file(source)
        if after != before:
            raise ReferenceMatchContractError(
                "output changed during metadata attestation"
            )
    except (OSError, ReferenceMatchContractError) as exc:
        return ReferenceFileOutputMetadataAttestation(
            schema_id=REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID,
            path=source,
            file_sha256=before,
            format_name=suffix.removeprefix(".").upper(),
            encoding_profile=encoding_profile,
            orientation=1,
            observed_metadata=(),
            sensitive_metadata_absent=False,
            accepted=False,
            failure_code=str(exc),
        )
    return ReferenceFileOutputMetadataAttestation(
        schema_id=REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID,
        path=source,
        file_sha256=before,
        format_name=format_name,
        encoding_profile=encoding_profile,
        orientation=orientation,
        observed_metadata=observed,
        sensitive_metadata_absent=True,
        accepted=True,
        failure_code=None,
    )


__all__ = [
    "REFERENCE_FILE_OUTPUT_METADATA_POLICY_CLAIM_CEILING",
    "REFERENCE_FILE_OUTPUT_METADATA_POLICY_ID",
    "ReferenceFileOutputMetadataAttestation",
    "attest_reference_file_output_metadata",
    "reference_file_output_metadata_policy_payload",
]
