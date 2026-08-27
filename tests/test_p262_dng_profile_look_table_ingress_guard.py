from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.preprocess.dng_forward_raster as module
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _guard_unsupported_profile_look_table,
    load_dng_forward_working_image,
)


def _page(*codes: int) -> SimpleNamespace:
    return SimpleNamespace(tags={code: object() for code in codes}, pages=None)


def test_profile_look_table_guard_accepts_absent_family() -> None:
    _guard_unsupported_profile_look_table([("0", _page(50721, 50964))])


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (50981, "ProfileLookTableDims"),
        (50982, "ProfileLookTableData"),
        (51108, "ProfileLookTableEncoding"),
    ],
)
def test_profile_look_table_guard_rejects_each_tag(code: int, name: str) -> None:
    with pytest.raises(DngForwardRasterError, match=rf"{name}\({code}\)@0"):
        _guard_unsupported_profile_look_table([("0", _page(code))])


def test_profile_look_table_guard_sorts_codes_and_records_ifds() -> None:
    with pytest.raises(DngForwardRasterError) as caught:
        _guard_unsupported_profile_look_table(
            [("1", _page(51108, 50982, 50981)), ("0", _page(51108))]
        )
    assert str(caught.value).endswith(
        "ProfileLookTableDims(50981)@1, "
        "ProfileLookTableData(50982)@1, "
        "ProfileLookTableEncoding(51108)@1+0"
    )


def test_public_loader_rejects_look_table_before_camera_decode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "profile.dng"
    source.write_bytes(b"metadata-only test")

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [_page(50981, 50982)]

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
    with pytest.raises(DngForwardRasterError, match="ProfileLookTable"):
        load_dng_forward_working_image(source)
    assert calls == 0
