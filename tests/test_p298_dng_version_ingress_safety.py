from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.preprocess.dng_forward_raster as module
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_dng_version,
    load_dng_forward_working_image,
)


def _tag(value: bytes, *, dtype: int = 1, count: int = 4) -> SimpleNamespace:
    return SimpleNamespace(value=value, dtype=dtype, count=count)


def _tags(
    version: bytes | None = b"\x01\x04\x00\x00",
    backward: bytes | None = b"\x01\x01\x00\x00",
) -> dict[int, SimpleNamespace]:
    result: dict[int, SimpleNamespace] = {}
    if version is not None:
        result[50706] = _tag(version)
    if backward is not None:
        result[50707] = _tag(backward)
    return result


def test_supported_versions_and_sdk_default_are_exact() -> None:
    assert _guard_dng_version(_tags()) == {
        "version": (1, 4, 0, 0),
        "backward_tag": (1, 1, 0, 0),
        "resolved_backward": (1, 1, 0, 0),
    }
    assert _guard_dng_version(_tags(b"\x01\x02\x00\x00", None)) == {
        "version": (1, 2, 0, 0),
        "backward_tag": None,
        "resolved_backward": (1, 2, 0, 0),
    }


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ({}, "missing required DNGVersion"),
        ({50706: _tag(b"\x01\x04\x00\x00", dtype=3)}, "four BYTE"),
        ({50706: _tag(b"\x01\x04\x00", count=3)}, "four BYTE"),
        (_tags(b"\x00\xff\x00\x00"), "supported range"),
        (_tags(b"\x01\x07\x02\x00"), "supported range"),
        (_tags(backward=b"\x00\xff\x00\x00"), "below 1.0.0.0"),
        (_tags(backward=b"\x01\x05\x00\x00"), "exceeds DNGVersion"),
    ],
)
def test_invalid_version_declarations_reject(
    tags: dict[int, SimpleNamespace], message: str
) -> None:
    with pytest.raises(DngForwardRasterError, match=message):
        _guard_dng_version(tags)


def test_public_loader_rejects_missing_version_before_camera_decode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "renamed-tiff.dng"
    source.write_bytes(b"metadata-only test")

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [SimpleNamespace(tags={}, pages=None)]

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
    with pytest.raises(DngForwardRasterError, match="missing required DNGVersion"):
        load_dng_forward_working_image(source)
    assert calls == 0
