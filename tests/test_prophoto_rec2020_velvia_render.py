from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.inference.romm_rec2020_velvia import (
    PROPHOTO_PROFILE_ID,
    ROMMRec2020RenderError,
    load_profile,
    render_official_romm_velvia_rec2020,
    render_supported_prophoto_velvia_rec2020,
)
from src.preprocess import FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"
SCRIPT = ROOT / "scripts/render_supported_prophoto_rec2020_velvia.py"
NATURAL_SOURCE = (
    ROOT
    / "data/external/fivek-bq0-casebank512-v1/expert_c/a2798-kme_383.tif"
)


def _matrix_shaper_profile() -> bytes:
    with tifffile.TiffFile(NATURAL_SOURCE) as document:
        return bytes(document.pages[0].tags[34675].value)


def _write_supported_prophoto(path: Path) -> None:
    profile = _matrix_shaper_profile()
    assert hashlib.sha256(profile).hexdigest() == FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256
    pixels = np.asarray(
        [
            [[65535, 0, 0], [0, 65535, 0], [0, 0, 65535]],
            [[4096, 32768, 61440], [49152, 8192, 32768], [32768, 32768, 32768]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )


def test_supported_prophoto_profile_is_exactly_evidence_bound() -> None:
    profile, profile_sha256 = load_profile(PROFILE, root=ROOT)
    assert profile["profile_id"] == PROPHOTO_PROFILE_ID
    assert profile_sha256
    assert profile["input"]["embedded_icc_sha256s"] == [
        FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256
    ]
    assert profile["production_default_changed"] is False


def test_supported_prophoto_renderer_is_deterministic_and_profile_strict(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_supported_prophoto(source)
    receipt_a = render_supported_prophoto_velvia_rec2020(
        source, first, profile_path=PROFILE, root=ROOT
    )
    receipt_b = render_supported_prophoto_velvia_rec2020(
        source, second, profile_path=PROFILE, root=ROOT
    )
    assert first.read_bytes() == second.read_bytes()
    assert receipt_a == receipt_b
    assert receipt_a["profile_id"] == PROPHOTO_PROFILE_ID
    assert receipt_a["input"]["embedded_icc_sha256"] == (
        FIVEK_PROPHOTO_MATRIX_SHAPER_ICC_SHA256
    )
    assert receipt_a["output"]["exact_sample_readback"] is True
    with pytest.raises(ROMMRec2020RenderError, match="official ROMM renderer"):
        render_official_romm_velvia_rec2020(
            source,
            tmp_path / "wrong.png",
            profile_path=PROFILE,
            root=ROOT,
        )


def test_supported_prophoto_cli_writes_bound_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.tiff"
    output = tmp_path / "output.png"
    receipt_path = tmp_path / "receipt.json"
    _write_supported_prophoto(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
            "--receipt",
            str(receipt_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert completed.stdout.strip() == str(output)
    assert receipt["profile_id"] == PROPHOTO_PROFILE_ID
    assert receipt["production_default_changed"] is False
