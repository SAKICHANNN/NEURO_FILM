from __future__ import annotations

import argparse
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

from scripts.render_film import build_color_render, build_color_render_float
from src.filmfx import composite_layers, dust_scratch_layer, grain_residual_layer, halation_layer

from src.preprocess import (
    load_working_image,
    save_srgb16_png,
    save_srgb16_tiff,
    working_image_to_legacy_srgb8,
    working_image_to_srgb_float,
)


ROOT = Path(__file__).resolve().parents[1]


def _render_args(style: str = "velvia_50") -> argparse.Namespace:
    return argparse.Namespace(
        stats=ROOT / "configs" / "film_color_stats.json",
        profile_config=ROOT / "configs" / "color_rendering_profiles.yaml",
        preset="safe-rich",
        style=style,
        seed=7,
        guardrails=ROOT / "configs" / "color_guardrails.json",
    )


def _quantize8(rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


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
    assert metrics["input_decode"]["legacy_8bit_adapter"] is False
    assert metrics["input_decode"]["internal_color_precision"] == "float32"
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
def test_render_film_e2e_records_tiff_png16_ingress_and_float_output_boundary(
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
    assert metrics["input_decode"]["legacy_8bit_adapter"] is False
    assert metrics["input_decode"]["internal_color_precision"] == "float32"
    assert metrics["output_encode"]["bit_depth"] == 8


def test_default_float_path_is_exact_for_srgb8_colour_and_bounded_for_effects(tmp_path: Path) -> None:
    y, x = np.mgrid[0:48, 0:64]
    source = np.stack((x * 4, y * 5, (x + y) * 2), axis=2).clip(0, 255).astype(np.uint8)
    input_path = tmp_path / "input.png"
    Image.fromarray(source, mode="RGB").save(input_path)
    working = load_working_image(input_path)
    render_args = _render_args()

    legacy = np.asarray(
        build_color_render(working_image_to_legacy_srgb8(working), render_args),
        dtype=np.float32,
    ) / 255.0
    floating = build_color_render_float(working_image_to_srgb_float(working), render_args)
    assert np.array_equal(_quantize8(legacy), _quantize8(floating))

    def with_effects(base: np.ndarray) -> np.ndarray:
        layers = [
            grain_residual_layer(base, strength=0.35, seed=7, color=True),
            halation_layer(base, strength=0.35),
            dust_scratch_layer(base.shape, strength=0.25, seed=24),
        ]
        return composite_layers(base, layers, output_margin=4)

    delta = np.abs(
        _quantize8(with_effects(legacy)).astype(np.int16)
        - _quantize8(with_effects(floating)).astype(np.int16)
    )
    assert int(delta.max()) <= 1


def test_default_8bit_export_uses_float_detail_from_16bit_input(tmp_path: Path) -> None:
    rgb = np.random.default_rng(20260717).random((48, 64, 3), dtype=np.float32)
    input_path = tmp_path / "input16.tiff"
    output_path = tmp_path / "output.png"
    save_srgb16_tiff(rgb, input_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(input_path),
            "--style",
            "hp5",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    working = load_working_image(input_path)
    render_args = _render_args("hp5")
    expected_float = _quantize8(
        composite_layers(
            build_color_render_float(working_image_to_srgb_float(working), render_args),
            [],
            output_margin=4,
        )
    )
    expected_legacy = _quantize8(
        composite_layers(
            np.asarray(
                build_color_render(working_image_to_legacy_srgb8(working), render_args),
                dtype=np.float32,
            )
            / 255.0,
            [],
            output_margin=4,
        )
    )
    with Image.open(output_path) as rendered:
        actual = np.asarray(rendered.convert("RGB"))
    assert np.array_equal(actual, expected_float)
    assert not np.array_equal(actual, expected_legacy)


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


def test_render_film_rejects_ultra_hdr_marker_before_output(tmp_path: Path) -> None:
    input_path = tmp_path / "ultra_hdr.jpg"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (8, 6), (30, 60, 90)).save(input_path)
    input_path.write_bytes(
        input_path.read_bytes()
        + b'<rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0"/>'
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "refusing SDR fallback" in completed.stderr
    assert not output_path.exists()
