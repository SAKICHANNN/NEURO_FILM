from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.preprocess import load_aces2065_openexr_working_image
from src.preprocess.aces2065_openexr import (
    ACES2065_ADOPTED_NEUTRAL,
    ACES2065_CHROMATICITIES,
    ACES2065_TO_ACESCG_MATRIX,
    Aces2065OpenExrError,
    _load_with_module,
)


def test_opt_in_ingress_is_exported_without_loading_openexr() -> None:
    assert callable(load_aces2065_openexr_working_image)


class _FakeFile:
    def __init__(self, header: dict[str, object], channels: dict[str, object]):
        self._header = header
        self._channels = channels
        self.parts = [object()]

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def header(self) -> dict[str, object]:
        return self._header

    def channels(self) -> dict[str, object]:
        return self._channels


def _module(
    pixels: np.ndarray,
    *,
    metadata: dict[str, object] | None = None,
    channels: dict[str, object] | None = None,
    parts: int = 1,
):
    scanline = object()
    header: dict[str, object] = {
        "acesImageContainerFlag": 1,
        "adoptedNeutral": ACES2065_ADOPTED_NEUTRAL,
        "chromaticities": ACES2065_CHROMATICITIES,
        "colorInteropID": "lin_ap0_scene",
        "type": scanline,
    }
    if metadata:
        header.update(metadata)
    packed = channels or {"RGB": SimpleNamespace(pixels=pixels)}

    def make_file(_path: str) -> _FakeFile:
        value = _FakeFile(header, packed)
        value.parts = [object() for _ in range(parts)]
        return value

    return SimpleNamespace(File=make_file, scanlineimage=scanline)


def test_strict_ap0_ingress_returns_owned_acescg_working_image(tmp_path: Path) -> None:
    ap0 = np.asarray(
        [[[-0.1, 0.2, 1.4], [0.5, 0.7, 0.9]]], dtype=np.float32
    )
    source = tmp_path / "master.exr"
    source.write_bytes(b"fixture")
    working = _load_with_module(source, _module(ap0))
    expected = np.asarray(
        ap0.astype(np.float64) @ ACES2065_TO_ACESCG_MATRIX.T,
        dtype=np.float32,
    )
    assert np.array_equal(working.pixels, expected)
    assert working.pixels.flags.c_contiguous
    assert working.pixels.flags.owndata
    assert working.pixels.flags.writeable
    assert working.working_space == "acescg_ap1_d60"
    assert working.transfer_state == "scene_linear"
    assert working.source_path == source
    ap0[...] = 0.0
    assert np.array_equal(working.pixels, expected)


@pytest.mark.parametrize(
    ("metadata", "parts"),
    [
        ({"acesImageContainerFlag": 0}, 1),
        ({"colorInteropID": "lin_ap1_scene"}, 1),
        ({"adoptedNeutral": (0.3127, 0.329)}, 1),
        ({"chromaticities": (0.0,) * 8}, 1),
        ({"type": object()}, 1),
        ({}, 2),
    ],
)
def test_wrong_container_identity_rejects_before_output(
    tmp_path: Path, metadata: dict[str, object], parts: int
) -> None:
    source = tmp_path / "wrong.exr"
    source.write_bytes(b"fixture")
    with pytest.raises(Aces2065OpenExrError):
        _load_with_module(
            source,
            _module(np.zeros((1, 1, 3), dtype=np.float32), metadata=metadata, parts=parts),
        )


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((1, 1, 3), dtype=np.float16),
        np.zeros((1, 1, 4), dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
        np.full((1, 1, 3), 65505.0, dtype=np.float32),
        np.zeros((0, 1, 3), dtype=np.float32),
    ],
)
def test_invalid_pixels_reject_before_output(tmp_path: Path, pixels: np.ndarray) -> None:
    source = tmp_path / "wrong.exr"
    source.write_bytes(b"fixture")
    with pytest.raises(Aces2065OpenExrError):
        _load_with_module(source, _module(pixels))


def test_extra_channel_group_rejects(tmp_path: Path) -> None:
    source = tmp_path / "wrong.exr"
    source.write_bytes(b"fixture")
    channels = {
        "RGB": SimpleNamespace(pixels=np.zeros((1, 1, 3), dtype=np.float32)),
        "A": SimpleNamespace(pixels=np.ones((1, 1), dtype=np.float32)),
    }
    with pytest.raises(Aces2065OpenExrError):
        _load_with_module(
            source,
            _module(np.zeros((1, 1, 3), dtype=np.float32), channels=channels),
        )
