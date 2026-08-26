from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p247_acescg_openexr_24mp_resources import _fill_probe
from scripts.audit_p248_acescg_openexr_scanline_streaming import (
    P248Error,
    _expected_sha,
    _generate_block,
    _safe_extract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p248_acescg_openexr_scanline_streaming_v1.json"
SOURCE = ROOT / "src/eval/p248_acescg_openexr_scanline_writer.cpp"


def test_p248_block_generation_matches_frozen_p247_formula() -> None:
    height, width, period = 37, 43, 1024
    expected = np.empty((height, width, 3), dtype=np.float32)
    _fill_probe(expected, 64, period)
    observed = np.concatenate(
        [_generate_block(y0, min(7, height - y0), width, period) for y0 in range(0, height, 7)]
    )
    assert np.array_equal(observed, expected)
    assert _expected_sha(height, width, 7, period) == hashlib.sha256(
        expected.tobytes(order="C")
    ).hexdigest()


def test_p248_config_freezes_materially_different_scanline_mechanism() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["probe"]["write_row_block"] == 16
    assert config["probe"]["compression"] == "ZIP_COMPRESSION"
    assert config["gates"]["maximum_worker_process_tree_rss_bytes"] == 2_147_483_648
    assert config["bindings"]["openexr_source_sha256"] == (
        "ab893d8003773ccd9a5556b2caf38da591ae37e20b06ee9d589a08984c5191f2"
    )
    assert config["bindings"]["imath_tag"] == "v3.2.2"
    text = SOURCE.read_text(encoding="utf-8")
    assert "writePixels(count)" in text
    assert "MoveFileExW" in text
    assert "ZIP_COMPRESSION" in text


def test_p248_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    import io
    import tarfile

    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        member = tarfile.TarInfo("../escape")
        payload = b"bad"
        member.size = len(payload)
        target.addfile(member, io.BytesIO(payload))
    with pytest.raises(P248Error, match="unsafe archive member"):
        _safe_extract(archive, tmp_path / "out")
