from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.preprocess.dng_forward_raster as module
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_unsupported_profile_tone_curve,
    load_dng_forward_working_image,
)


def _page(*codes: int) -> SimpleNamespace:
    return SimpleNamespace(tags={code: object() for code in codes}, pages=None)


def test_profile_tone_curve_guard_accepts_absent_tag() -> None:
    _guard_unsupported_profile_tone_curve([("0", _page(50721, 50964))])


def test_profile_tone_curve_guard_rejects_tag_and_records_ifds() -> None:
    with pytest.raises(DngForwardRasterError) as caught:
        _guard_unsupported_profile_tone_curve(
            [("1", _page(50940)), ("0", _page(50940))]
        )
    assert str(caught.value).endswith("ProfileToneCurve(50940)@1+0")


def test_public_loader_rejects_tone_curve_before_camera_decode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "profile.dng"
    source.write_bytes(b"metadata-only test")

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [_page(50940)]

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
    with pytest.raises(DngForwardRasterError, match="ProfileToneCurve"):
        load_dng_forward_working_image(source)
    assert calls == 0
