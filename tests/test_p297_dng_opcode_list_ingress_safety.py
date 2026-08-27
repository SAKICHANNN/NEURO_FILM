from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

import src.preprocess.dng_forward_raster as module
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_dng_opcode_lists,
    _parse_optional_opcode_list,
    load_dng_forward_working_image,
)


def _opcode(
    *,
    opcode_id: int = 9,
    version: int = 0x01030000,
    flags: int = 1,
    data: bytes = b"data",
) -> bytes:
    return struct.pack(">IIII", opcode_id, version, flags, len(data)) + data


def _list(*entries: bytes, count: int | None = None, tail: bytes = b"") -> bytes:
    return (
        struct.pack(">I", len(entries) if count is None else count)
        + b"".join(entries)
        + tail
    )


def _page(values: dict[int, bytes]) -> SimpleNamespace:
    return SimpleNamespace(
        tags={code: SimpleNamespace(value=value) for code, value in values.items()},
        pages=None,
    )


def test_optional_opcode_lists_are_summarized_deterministically() -> None:
    pages = [
        (
            "0",
            _page(
                {
                    51022: _list(_opcode(opcode_id=1)),
                    51009: _list(_opcode(), _opcode()),
                }
            ),
        )
    ]
    assert _guard_dng_opcode_lists(pages) == (
        "OpcodeList2(51009)@0:9/9",
        "OpcodeList3(51022)@0:1",
    )


@pytest.mark.parametrize("code", [51008, 51009, 51022])
def test_required_opcode_rejects_each_list(code: int) -> None:
    with pytest.raises(DngForwardRasterError, match="required DNG OpcodeList"):
        _guard_dng_opcode_lists([("0", _page({code: _list(_opcode(flags=0))}))])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\x00\x00\x00", "truncated count"),
        (_list(count=1), "impossible opcode count"),
        (_list(_opcode(data=b"abc")[:-1]), "truncated payload"),
        (_list(_opcode(), tail=b"x"), "trailing bytes"),
        (_list(_opcode(flags=5)), "reserved flags"),
        (_list(_opcode(version=0)), "unsupported version"),
        (_list(_opcode(version=0x01070200)), "unsupported version"),
    ],
)
def test_malformed_or_unsupported_opcode_rejects(payload: bytes, message: str) -> None:
    with pytest.raises(DngForwardRasterError, match=message):
        _parse_optional_opcode_list(payload, tag_name="OpcodeList2", ifd_path="0")


def test_empty_opcode_list_is_valid_and_silent() -> None:
    assert _guard_dng_opcode_lists([("0", _page({51022: _list()}))]) == ()


def test_public_loader_rejects_required_opcode_before_camera_decode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "required-opcode.dng"
    source.write_bytes(b"metadata-only test")

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [_page({51008: _list(_opcode(opcode_id=5, flags=0))})]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    calls = 0

    def fail_decode(_path):
        nonlocal calls
        calls += 1
        raise AssertionError("camera decode must not run")

    monkeypatch.setattr(module.tifffile, "TiffFile", lambda _path: FakeDocument())
    monkeypatch.setattr(module, "_decode_camera_linear_dng", fail_decode)
    with pytest.raises(DngForwardRasterError, match="required DNG OpcodeList1"):
        load_dng_forward_working_image(source)
    assert calls == 0
