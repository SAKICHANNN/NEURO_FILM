from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from src.inference.romm_rec2020_velvia import (
    ROMMRec2020RenderError,
    render_supported_prophoto_velvia_rec2020,
)
from src.inference.romm_rec2020_velvia_staged import (
    render_supported_prophoto_velvia_rec2020_staged,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"
FIXTURE = (
    ROOT
    / "data/wide_gamut/u1_4c18_prophoto_24mp/a0957-IMG_0032_6000x4000.tif"
)


def _write_source(path: Path) -> None:
    with tifffile.TiffFile(FIXTURE) as document:
        profile = bytes(document.pages[0].tags[34675].value)
    rng = np.random.default_rng(1901)
    pixels = rng.integers(0, 65536, size=(47, 61, 3), dtype=np.uint16)
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )


def test_staged_renderer_is_byte_and_receipt_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    _write_source(source)

    full_receipt = render_supported_prophoto_velvia_rec2020(
        source, expected, profile_path=PROFILE, root=ROOT
    )
    staged_receipt = render_supported_prophoto_velvia_rec2020_staged(
        source,
        actual,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "scratch",
        row_chunk=13,
    )

    assert actual.read_bytes() == expected.read_bytes()
    assert staged_receipt == full_receipt
    assert not list((tmp_path / "scratch").iterdir())


def test_staged_renderer_rejects_invalid_chunk_without_output(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "output.png"
    _write_source(source)
    try:
        render_supported_prophoto_velvia_rec2020_staged(
            source,
            output,
            profile_path=PROFILE,
            root=ROOT,
            scratch_dir=tmp_path / "scratch",
            row_chunk=0,
        )
    except ROMMRec2020RenderError as exc:
        assert "row_chunk" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid chunk was accepted")
    assert not output.exists()
