from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, features

from src.inference import replay_style_safe_recipe_to_file
from src.preprocess import inspect_input, load_working_image
from src.preprocess.avif_sdr import StrictSdrAvifError, inspect_strict_sdr_avif

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
P278_ROOT = ROOT / "data/external/p278_libavif_gainmap_v1"


def _require_avif() -> None:
    if not features.check("avif"):
        pytest.skip("Pillow AVIF support is unavailable")


def _rgb_avif(path: Path) -> None:
    _require_avif()
    y, x = np.mgrid[:29, :37]
    pixels = np.stack(
        (
            (x * 11 + y * 3 + 17) % 251,
            (x * 5 + y * 13 + 31) % 251,
            (x * 7 + y * 19 + 47) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, quality=100, subsampling="4:4:4", speed=8
    )


def _replace_after(data: bytes, marker: bytes, offset: int, value: bytes) -> bytes:
    index = data.index(marker) + len(marker) + offset
    return data[:index] + value + data[index + len(value) :]


def test_generated_srgb_avif_is_explicitly_admitted(tmp_path: Path) -> None:
    path = tmp_path / "ordinary.avif"
    _rgb_avif(path)
    strict = inspect_strict_sdr_avif(path)
    assert strict.width == 37
    assert strict.height == 29
    assert strict.bits_per_channel == (8, 8, 8)
    assert strict.nclx == (1, 13, 6, True)

    inspection = inspect_input(path)
    assert inspection.format_name == "AVIF"
    assert inspection.source_profile.kind == "nclx"
    assert not any(warning.code == "assumed_srgb" for warning in inspection.warnings)
    assert not any(
        warning.code == "unsupported_dynamic_range" for warning in inspection.warnings
    )
    working = load_working_image(path)
    assert working.pixels.shape == (29, 37, 3)
    assert working.pixels.dtype == np.float32
    assert np.isfinite(working.pixels).all()
    assert working.working_space == "linear_srgb"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("ten_bit", "8-bit"),
        ("pq", "NCLX"),
    ],
)
def test_non_sdr_or_non_8bit_avif_rejects_before_pixels(
    tmp_path: Path, mutation: str, message: str
) -> None:
    path = tmp_path / f"{mutation}.avif"
    _rgb_avif(path)
    data = path.read_bytes()
    if mutation == "ten_bit":
        data = _replace_after(data, b"pixi", 5, b"\x0a\x0a\x0a")
    else:
        data = _replace_after(data, b"nclx", 2, (16).to_bytes(2, "big"))
    path.write_bytes(data)
    with pytest.raises(StrictSdrAvifError, match=message):
        inspect_strict_sdr_avif(path)
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(path)


def test_alpha_and_sequence_avif_reject_before_working_pixels(tmp_path: Path) -> None:
    _require_avif()
    alpha = tmp_path / "alpha.avif"
    Image.new("RGBA", (11, 9), (10, 20, 30, 127)).save(alpha, quality=100)
    with pytest.raises(StrictSdrAvifError, match="alpha|auxiliary"):
        inspect_strict_sdr_avif(alpha)
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(alpha)

    sequence = tmp_path / "sequence.avif"
    frames = [Image.new("RGB", (11, 9), color) for color in ((1, 2, 3), (4, 5, 6))]
    frames[0].save(sequence, save_all=True, append_images=frames[1:], duration=20)
    with pytest.raises(StrictSdrAvifError, match="sequence|brand"):
        inspect_strict_sdr_avif(sequence)
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(sequence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("truncated", "truncated|invalid"),
        ("icc", "NCLX"),
        ("unknown_property", "unsupported.*property"),
    ],
)
def test_malformed_or_ambiguous_avif_rejects_before_pixels(
    tmp_path: Path, mutation: str, message: str
) -> None:
    path = tmp_path / f"{mutation}.avif"
    _rgb_avif(path)
    data = path.read_bytes()
    if mutation == "truncated":
        data = data[:80]
    elif mutation == "icc":
        data = data.replace(b"nclx", b"prof", 1)
    else:
        data = data.replace(b"av1C", b"zzzz", 1)
    path.write_bytes(data)
    with pytest.raises(StrictSdrAvifError, match=message):
        inspect_strict_sdr_avif(path)
    with pytest.raises(ValueError, match="refusing SDR fallback"):
        load_working_image(path)


def test_all_p278_gainmap_and_structural_fixtures_remain_rejected() -> None:
    fixtures = sorted(P278_ROOT.glob("*.avif"))
    if len(fixtures) != 6:
        pytest.skip("exact P278 fixture set is unavailable")
    for path in fixtures:
        with pytest.raises(StrictSdrAvifError):
            inspect_strict_sdr_avif(path)
        with pytest.raises(ValueError, match="refusing SDR fallback"):
            load_working_image(path)


def test_product_cli_and_recipe_replay_accept_strict_sdr_avif(tmp_path: Path) -> None:
    source = tmp_path / "source.avif"
    output = tmp_path / "look.png"
    replay = tmp_path / "replay.png"
    _rgb_avif(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--output",
            str(output),
            "--write-recipe",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe_path = output.with_suffix(".recipe.json")
    assert output.exists() and recipe_path.exists()
    import json

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["evidence_grade"] == "look-approximation"
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    replay_style_safe_recipe_to_file(
        recipe, profile_path=PROFILE, output_path=replay, root=ROOT
    )
    assert replay.read_bytes() == output.read_bytes()
