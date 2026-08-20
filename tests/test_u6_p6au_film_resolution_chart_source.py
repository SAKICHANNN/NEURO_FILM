from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.film_resolution_chart_source import (
    _fetch_member,
    _validate_tiff,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p6au_contract_freezes_six_complete_resolution_groups() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p6au_film_resolution_chart_source_v1.json"
    )
    members = contract["selection"]["members"]
    assert len(members) == 18
    for format_name in contract["selection"]["formats"]:
        for stock in contract["selection"]["stocks"]:
            names = [
                row["name"]
                for row in members
                if f"TIFF/{format_name}/{stock}/" in row["name"]
            ]
            assert sorted(name.split("Scans ")[1].split("/")[0] for name in names) == [
                "2K",
                "4K",
                "6K",
            ]


def test_p6au_rgb16_tiff_validation(tmp_path: Path) -> None:
    path = tmp_path / "rgb16.tif"
    values = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    tifffile.imwrite(path, values, photometric="rgb")
    facts = _validate_tiff(path)
    assert facts["shape"] == [6, 8, 3]
    assert facts["dtype"] == "uint16"


def test_p6au_rejects_rgb8(tmp_path: Path) -> None:
    path = tmp_path / "rgb8.tif"
    tifffile.imwrite(path, np.zeros((4, 5, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="not RGB16"):
        _validate_tiff(path)


def test_p6au_accepts_data_descriptor_partial_local_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_path = tmp_path / "source.tif"
    values = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    tifffile.imwrite(payload_path, values, photometric="rgb")
    payload = payload_path.read_bytes()
    import struct
    import zlib

    packed = zlib.compress(payload)[2:-4]
    name = b"TIFF/source.tif"
    header = (
        b"PK\x03\x04"
        + struct.pack(
            "<5H3I2H",
            20,
            8,
            8,
            0,
            0,
            0,
            0,
            len(payload),
            len(name),
            0,
        )
        + name
        + packed
        + bytes(4096)
    )
    monkeypatch.setattr(
        "src.eval.film_resolution_chart_source._range",
        lambda _url, _start, _end: header,
    )
    row = {
        "name": name.decode(),
        "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        "compressed_size": len(packed),
        "uncompressed_size": len(payload),
        "local_header_offset": 10,
        "method": 8,
        "flags": 8,
    }
    result = _fetch_member("https://invalid.test/source.zip", row, tmp_path / "out")
    assert result["bytes"] == len(payload)
