from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.preprocess.openexr_chromaticity_ingress import (
    OPENEXR_DEFAULT_REC709_D65,
    OpenExrChromaticityError,
    _conversion_matrix,
    _load_with_module,
)


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
    chromaticities: tuple[float, ...] | None = OPENEXR_DEFAULT_REC709_D65,
    channels: dict[str, object] | None = None,
):
    scanline = object()
    header: dict[str, object] = {"type": scanline}
    if chromaticities is not None:
        header["chromaticities"] = chromaticities
    packed = channels or {"RGB": SimpleNamespace(pixels=pixels)}

    def make_file(_path: str) -> _FakeFile:
        return _FakeFile(header, packed)

    return SimpleNamespace(File=make_file, scanlineimage=scanline)


def test_rec709_matrix_maps_neutral_to_neutral() -> None:
    matrix = _conversion_matrix(OPENEXR_DEFAULT_REC709_D65)
    neutral = matrix @ np.ones(3, dtype=np.float64)
    assert np.max(np.abs(neutral - 1.0)) <= 2e-7


def test_explicit_and_authorized_default_rec709_are_exact(tmp_path: Path) -> None:
    pixels = np.asarray([[[-0.1, 0.5, 1.25], [0.2, 0.3, 0.4]]], dtype=np.float16)
    source = tmp_path / "fixture.exr"
    source.write_bytes(b"fixture")
    explicit = _load_with_module(
        source, _module(pixels), allow_standard_rec709_default=False
    )
    defaulted = _load_with_module(
        source,
        _module(pixels, chromaticities=None),
        allow_standard_rec709_default=True,
    )
    assert np.array_equal(explicit.pixels, defaulted.pixels)
    assert explicit.pixels.flags.owndata
    assert explicit.pixels.flags.writeable
    assert explicit.pixels.flags.c_contiguous
    assert explicit.working_space == "linear_rec2020"
    assert defaulted.hdr_metadata["color_identity_mode"].endswith("rec709_d65")


def test_missing_chromaticities_reject_without_authorization(tmp_path: Path) -> None:
    source = tmp_path / "fixture.exr"
    source.write_bytes(b"fixture")
    with pytest.raises(OpenExrChromaticityError):
        _load_with_module(
            source,
            _module(np.zeros((1, 1, 3), dtype=np.float32), chromaticities=None),
            allow_standard_rec709_default=False,
        )


@pytest.mark.parametrize(
    "chromaticities",
    [
        (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127),
        (0.64, 0.33, 0.64, 0.33, 0.15, 0.06, 0.3127, 0.329),
        (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.0),
        (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, float("nan"), 0.329),
    ],
)
def test_invalid_chromaticities_reject(
    tmp_path: Path, chromaticities: tuple[float, ...]
) -> None:
    source = tmp_path / "fixture.exr"
    source.write_bytes(b"fixture")
    with pytest.raises(OpenExrChromaticityError):
        _load_with_module(
            source,
            _module(
                np.zeros((1, 1, 3), dtype=np.float32), chromaticities=chromaticities
            ),
            allow_standard_rec709_default=False,
        )


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((1, 1, 3), dtype=np.uint16),
        np.zeros((1, 1, 4), dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
        np.full((1, 1, 3), 65_505.0, dtype=np.float32),
        np.zeros((0, 1, 3), dtype=np.float32),
    ],
)
def test_invalid_pixels_reject(tmp_path: Path, pixels: np.ndarray) -> None:
    source = tmp_path / "fixture.exr"
    source.write_bytes(b"fixture")
    with pytest.raises(OpenExrChromaticityError):
        _load_with_module(source, _module(pixels), allow_standard_rec709_default=False)


def test_alpha_or_extra_group_rejects(tmp_path: Path) -> None:
    source = tmp_path / "fixture.exr"
    source.write_bytes(b"fixture")
    channels = {
        "RGB": SimpleNamespace(pixels=np.zeros((1, 1, 3), dtype=np.float32)),
        "A": SimpleNamespace(pixels=np.ones((1, 1), dtype=np.float32)),
    }
    with pytest.raises(OpenExrChromaticityError):
        _load_with_module(
            source,
            _module(np.zeros((1, 1, 3), dtype=np.float32), channels=channels),
            allow_standard_rec709_default=False,
        )
