from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from scripts.build_u5_r2spcp3_png_eligible_roles import (
    _probe_member,
    _role_sort_key,
    _sorted_ids_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2spcp3_png_eligible_preference_roles_v1.json"


def test_spcp3_contract_filters_payload_before_role_assignment() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "U5.R2SPCP3"
    assert payload["exclusion"]["selected_scene_count_exact"] == 160
    assert payload["eligibility"]["expected_unassigned_scene_count"] == 648
    assert payload["eligibility"]["required_decompressed_signature_hex"] == (
        "89504e470d0a1a0a"
    )
    assert payload["selection"]["role_assignment_before_signature_lock_forbidden"]
    assert payload["eligibility"]["local_record_prefix_bytes_per_member"] == 1024
    assert payload["eligibility"]["maximum_total_range_bytes"] == 1327104
    assert payload["execution"]["maximum_parallel_range_requests"] == 8
    assert payload["execution"]["maximum_attempts_per_exact_range"] == 3
    assert payload["eligibility"]["full_member_read_forbidden"]
    assert payload["eligibility"]["image_decode_forbidden"]


def test_spcp3_identity_and_sort_are_domain_separated() -> None:
    assert _sorted_ids_sha256(["I0002", "I0001"]) == _sorted_ids_sha256(
        ["I0001", "I0002"]
    )
    assert _role_sort_key("I0001") != _role_sort_key("I0002")
    from scripts.build_u5_r2spcp2_preference_roles import _scene_sort_key

    assert _role_sort_key("I0001") != _scene_sort_key("I0001")


def test_raw_deflate_prefix_can_identify_png_without_full_member() -> None:
    png = bytes.fromhex("89504e470d0a1a0a") + b"bounded-prefix-fixture" * 20
    compressor = zlib.compressobj(level=6, wbits=-15)
    compressed = compressor.compress(png) + compressor.flush()
    decoder = zlib.decompressobj(-15)
    observed = decoder.decompress(compressed[:64], 8)
    assert observed == png[:8]
    assert observed != b"\xff\xd8\xff\xe0JFIF"


def test_member_probe_reads_only_frozen_range_and_parses_local_header(monkeypatch) -> None:
    name = "SPCP_dataset/images/I9999_01_01.png"
    png = bytes.fromhex("89504e470d0a1a0a") + b"prefix-only"
    compressor = zlib.compressobj(level=6, wbits=-15)
    compressed = compressor.compress(png) + compressor.flush()
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0,
        8,
        0,
        0,
        0,
        len(compressed),
        len(png),
        len(name.encode()),
        0,
    )
    record = header + name.encode() + compressed + (b"x" * 512)
    calls: list[tuple[int, int]] = []

    def fake_fetch(_url: str, byte_range: tuple[int, int]):
        calls.append(byte_range)
        requested = byte_range[1] - byte_range[0] + 1
        relative = byte_range[0] - 1234
        return (record + (b"x" * requested))[relative : relative + requested], {}

    monkeypatch.setattr(
        "scripts.build_u5_r2spcp3_png_eligible_roles._fetch", fake_fetch
    )
    fact = _probe_member(
        "fixture",
        {"local_offset": 1234, "name": name, "compressed_size": 2048},
        record_prefix_bytes=1024,
        required_signature=bytes.fromhex("89504e470d0a1a0a"),
        maximum_attempts=3,
        retry_delay_seconds=0.0,
    )
    assert calls == [(1234, 2257)]
    assert fact["range_bytes"] == 1024
    assert fact["method"] == 8
    assert fact["required_signature_exact"] is True
