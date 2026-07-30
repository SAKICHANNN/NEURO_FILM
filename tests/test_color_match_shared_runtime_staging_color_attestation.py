from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
import struct
import zlib

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image
import pytest
import tifffile

from src.color_match.canonical import canonical_sha256
from src.color_match.contracts import ReferenceMatchContractError
import src.color_match.shared_runtime_staging_color_attestation as attest
from src.color_match.shared_runtime_staging_decode import (
    decode_runtime_qualified_shared_staging_bytes_v1,
)
from src.preprocess.output_encode import srgb_icc_profile
from tests.test_color_match_shared_runtime_staging_decode import (
    _bound_snapshot_from_payloads,
    _real_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_runtime_staging_color_attestation_v1.schema.json"
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _insert_before_iend(raw: bytes, chunk: bytes) -> bytes:
    marker = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    offset = raw.rfind(marker)
    assert offset >= 0
    return raw[:offset] + chunk + raw[offset:]


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return (
        b"\xff"
        + bytes([marker])
        + struct.pack(">H", len(payload) + 2)
        + payload
    )


def _orientation_exif_payload(value: int) -> bytes:
    return (
        b"Exif\x00\x00"
        + b"II\x2a\x00\x08\x00\x00\x00"
        + b"\x01\x00"
        + b"\x12\x01\x03\x00\x01\x00\x00\x00"
        + struct.pack("<H", value)
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def _decode_real(
    tmp_path: Path,
    *,
    suffix: str,
    depth: int,
):
    return decode_runtime_qualified_shared_staging_bytes_v1(
        _real_snapshot(tmp_path, suffix=suffix, depth=depth)
    )


@pytest.mark.parametrize(
    ("suffix", "depth", "format_name", "binding"),
    [
        (".png", 16, "PNG", "png-iccp-exact-v1"),
        (".tiff", 16, "TIFF", "tiff-34675-icc-exact-v1"),
        (".png", 8, "PNG", "png-iccp-exact-v1"),
        (".jpg", 8, "JPEG", "jpeg-app2-icc-exact-v1"),
        (".tif", 8, "TIFF", "tiff-34675-icc-exact-v1"),
    ],
)
def test_real_p62_p65_p67_bytes_attest_exact_embedded_srgb(
    tmp_path: Path,
    suffix: str,
    depth: int,
    format_name: str,
    binding: str,
) -> None:
    decoded = _decode_real(
        tmp_path / f"{format_name}-{depth}",
        suffix=suffix,
        depth=depth,
    )
    result = attest.attest_runtime_staging_srgb_metadata_v1(decoded)

    assert result.decoded is decoded
    assert result.record.source_count == 2
    assert all(
        row.output_format == format_name
        and row.profile_binding == binding
        and row.profile_sha256
        == result.record.expected_profile_sha256
        and row.orientation == 1
        for row in result.record.outputs
    )
    assert result.record.path_consumption_authorized is False
    assert result.record.persistent_attestation_authorized is False
    assert result.record.application_authorized is False
    assert result.record.delivery_authorized is False
    attest.validate_runtime_staging_color_attested_decoded_batch_v1(
        result
    )


def test_record_roundtrip_and_schema_are_exact(tmp_path: Path) -> None:
    result = attest.attest_runtime_staging_srgb_metadata_v1(
        _decode_real(tmp_path, suffix=".png", depth=16)
    )
    encoded = attest.runtime_staging_color_attestation_record_to_json(
        result.record
    )
    restored = attest.runtime_staging_color_attestation_record_from_json(
        encoded
    )

    assert restored == result.record
    Draft202012Validator(json.loads(SCHEMA.read_text("utf-8"))).validate(
        json.loads(encoded)
    )
    assert "\\" not in encoded
    assert ":/" not in encoded


def test_missing_embedded_profile_fails_closed(tmp_path: Path) -> None:
    pixels = np.zeros((4, 5, 3), dtype=np.uint8)
    encoded = BytesIO()
    Image.fromarray(pixels, "RGB").save(encoded, format="PNG")
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (encoded.getvalue(), encoded.getvalue()),
        format_name="PNG",
        depth=8,
    )
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(snapshot)

    with pytest.raises(
        ReferenceMatchContractError,
        match="exactly one embedded iCCP",
    ):
        attest.attest_runtime_staging_srgb_metadata_v1(decoded)


def test_png_profile_decompression_and_trailing_stream_are_bounded() -> None:
    with pytest.raises(ReferenceMatchContractError, match="exceeds bounds"):
        attest._bounded_decompress_profile(
            zlib.compress(b"x" * (attest.MAX_EMBEDDED_PROFILE_BYTES + 1))
        )
    with pytest.raises(ReferenceMatchContractError, match="trailing data"):
        attest._bounded_decompress_profile(
            zlib.compress(srgb_icc_profile()) + b"trailing"
        )
    incompressible = b"".join(
        hashlib.sha256(str(index).encode()).digest()
        for index in range(3000)
    )
    compressed = zlib.compress(incompressible, level=0)
    assert len(compressed) > attest.MAX_COMPRESSED_PROFILE_BYTES
    with pytest.raises(
        ReferenceMatchContractError,
        match="compressed iCCP",
    ):
        attest._bounded_decompress_profile(compressed)


@pytest.mark.parametrize(
    "conflict",
    [
        b"duplicate",
        b"cicp",
        b"exif",
        b"gamma",
        b"chrm",
        b"itxt",
    ],
)
def test_png_duplicate_or_conflicting_metadata_fails_closed(
    tmp_path: Path,
    conflict: bytes,
) -> None:
    decoded = _decode_real(
        tmp_path / "source",
        suffix=".png",
        depth=8,
    )
    raw = decoded.snapshot.output_bytes[0]
    if conflict == b"duplicate":
        offset = raw.find(b"iCCP")
        assert offset >= 4
        length = struct.unpack(">I", raw[offset - 4 : offset])[0]
        chunk = raw[offset - 4 : offset + 8 + length]
    elif conflict == b"cicp":
        chunk = _png_chunk(b"cICP", b"\x01\x0d\x00\x01")
    elif conflict == b"exif":
        chunk = _png_chunk(b"eXIf", b"MM\x00*\x00\x00\x00\x08")
    elif conflict == b"gamma":
        chunk = _png_chunk(b"gAMA", struct.pack(">I", 45455))
    elif conflict == b"chrm":
        chunk = _png_chunk(
            b"cHRM",
            b"\x00\x00\x7a&" * 8,
        )
    else:
        chunk = _png_chunk(
            b"iTXt",
            b"XML:com.adobe.xmp\x00\x00\x00\x00\x00"
            b"<rdf:Description tiff:Orientation=\"6\"/>",
        )
    modified = _insert_before_iend(raw, chunk)
    bound_root = tmp_path / conflict.decode()
    bound_root.mkdir()
    snapshot = _bound_snapshot_from_payloads(
        bound_root,
        (modified, modified),
        format_name="PNG",
        depth=8,
    )
    decoded_modified = decode_runtime_qualified_shared_staging_bytes_v1(
        snapshot
    )

    with pytest.raises(ReferenceMatchContractError):
        attest.attest_runtime_staging_srgb_metadata_v1(decoded_modified)


def test_png_iccp_keyword_and_order_are_strict(tmp_path: Path) -> None:
    decoded = _decode_real(tmp_path, suffix=".png", depth=8)
    raw = decoded.snapshot.output_bytes[0]
    offset = raw.find(b"iCCP")
    assert offset >= 4
    length = struct.unpack(">I", raw[offset - 4 : offset])[0]
    start = offset - 4
    end = offset + 8 + length
    chunk = raw[start:end]
    late = _insert_before_iend(raw[:start] + raw[end:], chunk)
    with pytest.raises(ReferenceMatchContractError, match="header"):
        attest._png_profile(late, 8)

    payload = raw[offset + 4 : offset + 4 + length]
    invalid_payload = b" " + payload
    invalid = (
        raw[:start]
        + _png_chunk(b"iCCP", invalid_payload)
        + raw[end:]
    )
    with pytest.raises(ReferenceMatchContractError, match="header"):
        attest._png_profile(invalid, 8)

    palette = _png_chunk(b"PLTE", b"\x00\x00\x00")
    after_palette = raw[:start] + palette + raw[start:]
    with pytest.raises(ReferenceMatchContractError, match="header"):
        attest._png_profile(after_palette, 8)


def test_jpeg_post_scan_icc_and_exif_are_not_hidden(tmp_path: Path) -> None:
    decoded = _decode_real(tmp_path, suffix=".jpg", depth=8)
    raw = decoded.snapshot.output_bytes[0]
    icc_payload = next(
        payload
        for marker, payload in attest._jpeg_segments(raw)
        if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00")
    )

    duplicate_icc = (
        raw[:-2] + _jpeg_segment(0xE2, icc_payload) + raw[-2:]
    )
    with pytest.raises(ReferenceMatchContractError):
        attest._jpeg_profile(duplicate_icc, 8)

    oriented = (
        raw[:-2]
        + _jpeg_segment(0xE1, _orientation_exif_payload(6))
        + raw[-2:]
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest._jpeg_profile(oriented, 8)


def test_jpeg_duplicate_exif_cannot_hide_nonidentity(
    tmp_path: Path,
) -> None:
    decoded = _decode_real(tmp_path, suffix=".jpg", depth=8)
    raw = decoded.snapshot.output_bytes[0]
    prefixed = (
        raw[:2]
        + _jpeg_segment(0xE1, _orientation_exif_payload(1))
        + _jpeg_segment(0xE1, _orientation_exif_payload(6))
        + raw[2:]
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest._jpeg_profile(prefixed, 8)


def test_jpeg_xmp_and_photoshop_namespaces_fail_closed(
    tmp_path: Path,
) -> None:
    decoded = _decode_real(tmp_path, suffix=".jpg", depth=8)
    raw = decoded.snapshot.output_bytes[0]
    for marker, payload in (
        (
            0xE1,
            b"http://ns.adobe.com/xap/1.0/\x00"
            b"<rdf:Description tiff:Orientation=\"6\"/>",
        ),
        (0xED, b"Photoshop 3.0\x008BIM"),
    ):
        modified = raw[:-2] + _jpeg_segment(marker, payload) + raw[-2:]
        with pytest.raises(
            ReferenceMatchContractError,
            match="alternate metadata",
        ):
            attest._jpeg_profile(modified, 8)


@pytest.mark.parametrize(
    "marker",
    [*range(0xE1, 0xE2), *range(0xE3, 0xF0)],
)
def test_jpeg_non_allowlisted_app_namespaces_fail_closed(
    tmp_path: Path,
    marker: int,
) -> None:
    decoded = _decode_real(tmp_path, suffix=".jpg", depth=8)
    raw = decoded.snapshot.output_bytes[0]
    modified = (
        raw[:-2]
        + _jpeg_segment(marker, b"alternate-metadata")
        + raw[-2:]
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest._jpeg_profile(modified, 8)


def test_jpeg_nonidentity_orientation_fails_closed(tmp_path: Path) -> None:
    pixels = np.zeros((4, 5, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    encoded = BytesIO()
    Image.fromarray(pixels, "RGB").save(
        encoded,
        format="JPEG",
        icc_profile=srgb_icc_profile(),
        exif=exif,
    )
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (encoded.getvalue(), encoded.getvalue()),
        format_name="JPEG",
        depth=8,
    )
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(snapshot)

    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest.attest_runtime_staging_srgb_metadata_v1(decoded)


def test_tiff_nonidentity_orientation_fails_closed(tmp_path: Path) -> None:
    pixels = np.zeros((4, 5, 3), dtype=np.uint8)
    encoded = BytesIO()
    tifffile.imwrite(
        encoded,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[
            (34675, "B", len(srgb_icc_profile()), srgb_icc_profile(), False),
            (274, "H", 1, 6, False),
        ],
    )
    snapshot = _bound_snapshot_from_payloads(
        tmp_path,
        (encoded.getvalue(), encoded.getvalue()),
        format_name="TIFF",
        depth=8,
    )
    decoded = decode_runtime_qualified_shared_staging_bytes_v1(snapshot)

    with pytest.raises(
        ReferenceMatchContractError,
        match="orientation",
    ):
        attest.attest_runtime_staging_srgb_metadata_v1(decoded)


@pytest.mark.parametrize(
    "extra_tags",
    [
        [
            (34675, "B", len(srgb_icc_profile()), srgb_icc_profile(), False),
            (34675, "B", 4, b"EVIL", False),
        ],
        [
            (34675, "B", len(srgb_icc_profile()), srgb_icc_profile(), False),
            (274, "H", 1, 1, False),
            (274, "H", 1, 6, False),
        ],
    ],
)
def test_tiff_duplicate_colour_or_orientation_tags_fail_closed(
    extra_tags: list[tuple[object, ...]],
) -> None:
    encoded = BytesIO()
    tifffile.imwrite(
        encoded,
        np.zeros((4, 5, 3), dtype=np.uint8),
        photometric="rgb",
        metadata=None,
        extratags=extra_tags,
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="must be unique",
    ):
        attest._tiff_profile(encoded.getvalue(), 8)


@pytest.mark.parametrize("code", [700, 34377])
def test_tiff_alternate_metadata_namespaces_fail_closed(
    code: int,
) -> None:
    encoded = BytesIO()
    payload = b"<xmp tiff:Orientation='6'/>"
    tifffile.imwrite(
        encoded,
        np.zeros((4, 5, 3), dtype=np.uint8),
        photometric="rgb",
        metadata=None,
        extratags=[
            (34675, "B", len(srgb_icc_profile()), srgb_icc_profile(), False),
            (code, "B", len(payload), payload, False),
        ],
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest._tiff_profile(encoded.getvalue(), 8)


@pytest.mark.parametrize(
    ("code", "dtype", "count", "value"),
    [
        (280, "B", 1, 1),
        (281, "B", 1, 254),
        (301, "H", 256, tuple(range(256))),
        (318, "2I", 2, (1, 3, 1, 3)),
        (
            319,
            "2I",
            6,
            (1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3),
        ),
        (340, "B", 1, 1),
        (341, "B", 1, 254),
        (529, "2I", 3, (299, 1000, 587, 1000, 114, 1000)),
        (532, "2I", 6, (0, 1, 255, 1, 0, 1, 255, 1, 0, 1, 255, 1)),
    ],
)
def test_tiff_alternate_colour_tags_fail_closed(
    code: int,
    dtype: str,
    count: int,
    value: object,
) -> None:
    encoded = BytesIO()
    tifffile.imwrite(
        encoded,
        np.zeros((4, 5, 3), dtype=np.uint8),
        photometric="rgb",
        metadata=None,
        extratags=[
            (34675, "B", len(srgb_icc_profile()), srgb_icc_profile(), False),
            (code, dtype, count, value, False),
        ],
    )

    with pytest.raises(
        ReferenceMatchContractError,
        match="alternate metadata",
    ):
        attest._tiff_profile(encoded.getvalue(), 8)


def test_json_rejects_unknown_fields_and_self_consistent_forgery(
    tmp_path: Path,
) -> None:
    result = attest.attest_runtime_staging_srgb_metadata_v1(
        _decode_real(tmp_path, suffix=".tif", depth=8)
    )
    payload = result.record.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ReferenceMatchContractError, match="fields differ"):
        attest.runtime_staging_color_attestation_record_from_json(
            json.dumps(payload)
        )
    encoded = attest.runtime_staging_color_attestation_record_to_json(
        result.record
    )
    duplicate = encoded.replace(
        '"schema_id":',
        '"schema_id": "shadow", "schema_id":',
        1,
    )
    with pytest.raises(ReferenceMatchContractError, match="not valid JSON"):
        attest.runtime_staging_color_attestation_record_from_json(duplicate)
    with pytest.raises(ReferenceMatchContractError, match="not valid JSON"):
        attest.runtime_staging_color_attestation_record_from_json(
            encoded.replace("false", "NaN", 1)
        )

    first = replace(
        result.record.outputs[0],
        encoded_file_sha256="f" * 64,
    )
    first = replace(
        first,
        metadata_attestation_id=attest._output_attestation_id(
            row=first,
            profile_binding=first.profile_binding,
            profile_sha256=first.profile_sha256,
            profile_size_bytes=first.profile_size_bytes,
            orientation=first.orientation,
        ),
    )
    forged = replace(
        result.record,
        outputs=(first, *result.record.outputs[1:]),
    )
    forged = replace(
        forged,
        attestation_id=canonical_sha256(
            attest._identity_payload(forged)
        ),
    )
    forged_batch = replace(result, record=forged)

    with pytest.raises(
        ReferenceMatchContractError,
        match="does not match bytes",
    ):
        attest.validate_runtime_staging_color_attested_decoded_batch_v1(
            forged_batch
        )

    mutable_outputs = replace(
        result.record,
        outputs=list(result.record.outputs),  # type: ignore[arg-type]
    )
    mutable_outputs = replace(
        mutable_outputs,
        attestation_id=canonical_sha256(
            attest._identity_payload(mutable_outputs)
        ),
    )
    with pytest.raises(ReferenceMatchContractError):
        attest.validate_runtime_staging_color_attestation_record_v1(
            mutable_outputs
        )
