from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocess import load_working_image, working_image_to_legacy_srgb8


ROOT = Path(__file__).resolve().parents[1]


def test_working_image_legacy_adapter_round_trips_srgb_fixture(tmp_path: Path) -> None:
    source = np.array(
        [[[0, 32, 128], [64, 192, 255]], [[12, 80, 160], [24, 120, 240]]],
        dtype=np.uint8,
    )
    path = tmp_path / "fixture.png"
    Image.fromarray(source, mode="RGB").save(path)

    working = load_working_image(path)
    adapted = np.asarray(working_image_to_legacy_srgb8(working))

    assert working.working_space == "linear_srgb"
    assert working.transfer_state == "display_linear"
    assert working.source_transfer_state == "display_referred"
    assert np.max(np.abs(adapted.astype(np.int16) - source.astype(np.int16))) <= 1


def test_working_image_legacy_adapter_rejects_wrong_color_state(tmp_path: Path) -> None:
    path = tmp_path / "fixture.png"
    Image.new("RGB", (2, 2), (128, 64, 32)).save(path)
    working = load_working_image(path)
    working.transfer_state = "unknown"

    try:
        working_image_to_legacy_srgb8(working)
    except ValueError as exc:
        assert "requires linear_srgb/display_linear" in str(exc)
    else:
        raise AssertionError("wrong color state must fail closed")


@pytest.mark.parametrize(
    ("suffix", "expected_output_format"),
    [(".jpg", "JPEG"), (".png", "PNG"), (".tiff", "TIFF")],
)
def test_render_film_e2e_uses_working_image_for_sdr_rasters(
    tmp_path: Path, suffix: str, expected_output_format: str
) -> None:
    source = np.zeros((18, 24, 3), dtype=np.uint8)
    source[..., 0] = np.arange(24, dtype=np.uint8)[None, :] * 10
    source[..., 1] = 96
    source[..., 2] = np.arange(18, dtype=np.uint8)[:, None] * 12
    input_path = tmp_path / f"input{suffix}"
    output_path = tmp_path / f"output{suffix}"
    Image.fromarray(source, mode="RGB").save(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(input_path),
            "--output",
            str(output_path),
            "--write-metrics",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with Image.open(output_path) as rendered:
        assert rendered.format == expected_output_format
        assert rendered.mode == "RGB"
        assert rendered.size == (24, 18)
    metrics = json.loads(output_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_decode"]["working_space"] == "linear_srgb"
    assert metrics["input_decode"]["transfer_state"] == "display_linear"
    assert metrics["input_decode"]["source_transfer_state"] == "display_referred"
    assert metrics["input_decode"]["bit_depth_in"] == 8
    assert metrics["input_decode"]["legacy_8bit_adapter"] is True
    assert metrics["output_claim"]["output_label"] == "film-inspired"
    assert metrics["output_claim"]["color_state_policy"] == "look_approximation_only"
    assert metrics["output_claim"]["calibrated_reference_allowed"] is False
    assert metrics["output_encode"]["format"] == expected_output_format
    assert metrics["output_encode"]["bit_depth"] == 8
    assert metrics["output_encode"]["transfer"] == "sRGB"
    assert metrics["output_encode"]["icc_profile"] == "embedded standard sRGB"
    assert len(metrics["output_encode"]["icc_profile_sha256"]) == 64
