from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.inference import (
    RECIPE_SCHEMA_ID_V3,
    replay_style_safe_recipe_to_file,
    validate_render_recipe,
)
from src.preprocess import save_srgb16_png, srgb_icc_profile_sha256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _decoded_rgb16(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert value is not None and value.dtype == np.uint16
    return value[..., ::-1]


def test_default_png16_encoder_remains_exact_level_six(tmp_path: Path) -> None:
    rgb = np.linspace(0.0, 1.0, 31 * 37 * 3, dtype=np.float32).reshape(
        31, 37, 3
    )
    default = tmp_path / "default.png"
    explicit = tmp_path / "explicit.png"
    save_srgb16_png(rgb, default)
    save_srgb16_png(rgb, explicit, compression_level=6)
    assert default.read_bytes() == explicit.read_bytes()
    for invalid in (-1, 10, 1.0, True):
        with pytest.raises(ValueError, match="compression level"):
            save_srgb16_png(  # type: ignore[arg-type]
                rgb, tmp_path / "invalid.png", compression_level=invalid
            )


def test_fast_png16_preserves_samples_and_icc(tmp_path: Path) -> None:
    rgb = np.random.default_rng(72).random((61, 67, 3), dtype=np.float32)
    baseline = tmp_path / "baseline.png"
    fast = tmp_path / "fast.png"
    save_srgb16_png(rgb, baseline, compression_level=6)
    save_srgb16_png(rgb, fast, compression_level=0)
    assert np.array_equal(_decoded_rgb16(fast), _decoded_rgb16(baseline))
    with Image.open(fast) as image:
        profile = image.info.get("icc_profile", b"")
    assert hashlib.sha256(profile).hexdigest() == srgb_icc_profile_sha256()


def test_cli_fast_export_writes_v3_recipe_and_replays_exactly(
    tmp_path: Path,
) -> None:
    pixels = np.arange(27 * 31 * 3, dtype=np.uint8).reshape(27, 31, 3)
    source = tmp_path / "input.png"
    output = tmp_path / "fast.png"
    replay = tmp_path / "replay.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--output-bit-depth",
        "16",
        "--png-compression",
        "0",
        "--write-recipe",
        "--write-metrics",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    recipe = json.loads(output.with_suffix(".recipe.json").read_text())
    assert recipe["schema_id"] == RECIPE_SCHEMA_ID_V3
    assert recipe["render"]["look_amount"] == 1.0
    assert recipe["output"]["png_compression"] == 0
    validate_render_recipe(recipe)
    assert replay_style_safe_recipe_to_file(
        recipe,
        profile_path=PROFILE,
        output_path=replay,
        root=ROOT,
    ) == recipe["output"]["sha256"]
    assert replay.read_bytes() == output.read_bytes()
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["output_encode"]["png_compression"] == 0


def test_cli_rejects_inapplicable_compression_before_input_decode(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(tmp_path / "missing.png"),
        "--png-compression",
        "0",
        "--output",
        str(tmp_path / "output.png"),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert "requires safe_lab 16-bit PNG output" in completed.stderr


def test_v3_schema_is_strict_and_requires_compression() -> None:
    schema = json.loads(
        (ROOT / "configs/schemas/render_recipe_v3.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == RECIPE_SCHEMA_ID_V3
    assert "png_compression" in schema["properties"]["output"]["required"]
