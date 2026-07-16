from __future__ import annotations

from collections.abc import Callable
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import cv2
import tifffile
from PIL import Image

from src.preprocess import (
    load_working_image,
    save_srgb16_png,
    save_srgb16_tiff,
    working_image_to_legacy_srgb8,
    working_image_to_srgb_float,
)


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
        assert "requires linear_srgb with display_linear or scene_linear" in str(exc)
    else:
        raise AssertionError("wrong color state must fail closed")


def test_working_image_legacy_adapter_accepts_explicit_linear_srgb_scene_state(tmp_path: Path) -> None:
    path = tmp_path / "fixture.png"
    Image.new("RGB", (2, 2), (128, 64, 32)).save(path)
    working = load_working_image(path)
    working.transfer_state = "scene_linear"
    adapted = working_image_to_legacy_srgb8(working)
    assert adapted.mode == "RGB"
    assert adapted.size == (2, 2)
    encoded_float = working_image_to_srgb_float(working)
    assert encoded_float.dtype == np.float32


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
    assert len(metrics["output_encode"]["icc_profile_fingerprint_sha256"]) == 64


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [(".png", save_srgb16_png), (".tiff", save_srgb16_tiff)],
)
def test_render_film_e2e_records_tiff_png16_ingress_and_legacy_output_boundary(
    tmp_path: Path, suffix: str, writer: Callable[[np.ndarray, Path], str]
) -> None:
    rgb = np.linspace(0.0, 1.0, 18 * 24 * 3, dtype=np.float32).reshape(18, 24, 3)
    input_path = tmp_path / f"input16{suffix}"
    output_path = tmp_path / f"output16_{suffix[1:]}.png"
    writer(rgb, input_path)
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
    metrics = json.loads(output_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_decode"]["bit_depth_in"] == 16
    assert metrics["input_decode"]["legacy_8bit_adapter"] is True
    assert metrics["output_encode"]["bit_depth"] == 8


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_render_film_opt_in_float_path_writes_true_srgb16(tmp_path: Path, suffix: str) -> None:
    rgb = np.linspace(0.01, 0.99, 24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3)
    input_path = tmp_path / "input16.tiff"
    output_path = tmp_path / f"output16{suffix}"
    save_srgb16_tiff(rgb, input_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(input_path),
            "--output",
            str(output_path),
            "--output-bit-depth",
            "16",
            "--write-metrics",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    if suffix == ".png":
        decoded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)[..., ::-1]
    else:
        decoded = tifffile.imread(output_path)
    assert decoded.dtype == np.uint16
    assert len(np.unique(decoded)) > 256
    with Image.open(output_path) as rendered:
        assert rendered.info.get("icc_profile")
    metrics = json.loads(output_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_decode"]["legacy_8bit_adapter"] is False
    assert metrics["input_decode"]["internal_color_precision"] == "float32"
    assert metrics["output_encode"]["bit_depth"] == 16


def test_render_film_rejects_16bit_jpeg_before_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpg"
    Image.new("RGB", (4, 4), (30, 60, 90)).save(input_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(input_path),
            "--output",
            str(output_path),
            "--output-bit-depth",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "16-bit output requires" in completed.stderr
    assert not output_path.exists()
