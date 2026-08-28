from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import src.preprocess.dng_forward_raster as module
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _linearization_table_delegation,
    load_dng_forward_working_image,
)


def _tag(
    values: np.ndarray | list[int], *, dtype: int = 3, count: int | None = None
) -> SimpleNamespace:
    array = np.asarray(values)
    return SimpleNamespace(
        value=array,
        dtype=dtype,
        count=array.size if count is None else count,
    )


def _page(tag: SimpleNamespace | None = None) -> SimpleNamespace:
    tags = {} if tag is None else {50712: tag}
    return SimpleNamespace(tags=tags, pages=None)


def test_absent_linearization_table_has_no_delegation_receipt() -> None:
    assert _linearization_table_delegation([("0", _page())]) is None


def test_valid_linearization_table_receipt_is_exact() -> None:
    values = np.asarray([0, 1, 7, 65535], dtype=np.uint16)
    assert _linearization_table_delegation([("0/2", _page(_tag(values)))]) == {
        "tag_code": 50712,
        "ifd_path": "0/2",
        "count": 4,
        "payload_u16le_sha256": hashlib.sha256(
            values.astype("<u2", copy=False).tobytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        (_tag([0, 1], dtype=4), "must use TIFF SHORT"),
        (_tag([0], count=1), "count is out of range"),
        (_tag(np.arange(65537, dtype=np.uint32), count=65537), "count is out of range"),
        (_tag(np.asarray([0, 1], dtype=np.uint32)), "one uint16 array"),
        (_tag(np.asarray([[0, 1]], dtype=np.uint16)), "one uint16 array"),
        (_tag(np.asarray([0, 1], dtype=np.uint16), count=3), "one uint16 array"),
    ],
)
def test_invalid_linearization_table_rejects(
    tag: SimpleNamespace, message: str
) -> None:
    with pytest.raises(DngForwardRasterError, match=message):
        _linearization_table_delegation([("0", _page(tag))])


def test_multiple_linearization_tables_reject() -> None:
    tag = _tag(np.asarray([0, 1], dtype=np.uint16))
    with pytest.raises(DngForwardRasterError, match="multiple"):
        _linearization_table_delegation([("0", _page(tag)), ("0/1", _page(tag))])


def test_public_loader_rejects_invalid_table_before_camera_decode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "invalid-linearization-table.dng"
    source.write_bytes(b"metadata-only test")

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [_page(_tag([0, 1], dtype=4))]

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
    with pytest.raises(DngForwardRasterError, match="must use TIFF SHORT"):
        load_dng_forward_working_image(source)
    assert calls == 0
