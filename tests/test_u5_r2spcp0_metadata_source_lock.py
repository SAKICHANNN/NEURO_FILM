from __future__ import annotations

import io
import zipfile

import pytest

from scripts.audit_u5_r2spcp0_metadata_source_lock import (
    canonical_sha256,
    parse_central_directory,
    range_get,
)


def _central_directory(payload: bytes) -> bytes:
    end = payload.rfind(b"PK\x05\x06")
    assert end >= 0
    size = int.from_bytes(payload[end + 12 : end + 16], "little")
    offset = int.from_bytes(payload[end + 16 : end + 20], "little")
    return payload[offset : offset + size]


def test_parse_central_directory_reads_exact_member_facts() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("order_trans.xlsx", b"annotation-one")
        archive.writestr("nested/example.png", b"pixel")
    members = parse_central_directory(_central_directory(stream.getvalue()))
    assert [member.name for member in members] == [
        "order_trans.xlsx",
        "nested/example.png",
    ]
    assert members[0].uncompressed_size == len(b"annotation-one")
    assert members[1].method == zipfile.ZIP_DEFLATED


def test_canonical_sha256_is_key_order_invariant() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_range_get_rejects_out_of_archive_bounds_before_network() -> None:
    with pytest.raises(ValueError, match="invalid bounded archive range"):
        range_get("https://invalid.example", -1, 4)
