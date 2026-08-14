from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.inference.prophoto_rec2020_replay import (
    replay_supported_prophoto_velvia_rec2020,
)
from src.inference.romm_rec2020_velvia import (
    ROMMRec2020RenderError,
    render_supported_prophoto_velvia_rec2020,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"
SCRIPT = ROOT / "scripts/replay_supported_prophoto_rec2020_velvia.py"
NATURAL_SOURCE = (
    ROOT / "data/external/fivek-bq0-casebank512-v1/expert_c/a2798-kme_383.tif"
)


def _write_supported_prophoto(path: Path) -> None:
    with tifffile.TiffFile(NATURAL_SOURCE) as document:
        profile = bytes(document.pages[0].tags[34675].value)
    pixels = np.asarray(
        [[[65535, 0, 0], [0, 65535, 0]], [[0, 0, 65535], [32768] * 3]],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )


def test_supported_prophoto_receipt_replays_exactly(tmp_path: Path) -> None:
    source = tmp_path / "source.tiff"
    original = tmp_path / "original.png"
    replay = tmp_path / "replay.png"
    _write_supported_prophoto(source)
    receipt = render_supported_prophoto_velvia_rec2020(
        source, original, profile_path=PROFILE, root=ROOT
    )
    replayed = replay_supported_prophoto_velvia_rec2020(
        receipt, source, replay, profile_path=PROFILE, root=ROOT
    )
    assert replayed == receipt
    assert replay.read_bytes() == original.read_bytes()


def test_supported_prophoto_replay_rejects_identity_drift_without_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    original = tmp_path / "original.png"
    rejected = tmp_path / "rejected.png"
    _write_supported_prophoto(source)
    receipt = render_supported_prophoto_velvia_rec2020(
        source, original, profile_path=PROFILE, root=ROOT
    )
    drifted = json.loads(json.dumps(receipt))
    drifted["look"]["median_residual_scale"] = 0.0
    with pytest.raises(ROMMRec2020RenderError, match="receipt mismatch"):
        replay_supported_prophoto_velvia_rec2020(
            drifted, source, rejected, profile_path=PROFILE, root=ROOT
        )
    assert not rejected.exists()


def test_supported_prophoto_cli_replays_existing_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.tiff"
    original = tmp_path / "original.png"
    replay = tmp_path / "replay.png"
    receipt_path = tmp_path / "receipt.json"
    _write_supported_prophoto(source)
    receipt = render_supported_prophoto_velvia_rec2020(
        source, original, profile_path=PROFILE, root=ROOT
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(receipt_path),
            "--output",
            str(replay),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert replay.read_bytes() == original.read_bytes()
