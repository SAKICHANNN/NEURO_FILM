from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tifffile

from scripts.build_u1_4c18_prophoto_24mp_fixture import build_fixture


def test_prophoto_scale_fixture_is_exact_and_create_only(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "output.tif"
    profile = b"test-profile"
    pixels = np.arange(5 * 7 * 3, dtype=np.uint16).reshape(5, 7, 3)
    tifffile.imwrite(
        source,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    report = build_fixture(source, output, width=12, height=8)
    with tifffile.TiffFile(output) as document:
        stored_profile = bytes(document.pages[0].tags[34675].value)
        stored = document.pages[0].asarray()
    assert stored.shape == (8, 12, 3)
    assert stored.dtype == np.uint16
    assert stored_profile == profile
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["exact_sample_and_icc_readback"] is True
