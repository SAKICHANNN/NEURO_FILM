from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.inference.style_safe_engine import replay_style_safe_recipe_to_file
from src.inference.three_stock_preview import render_three_stock_previews_to_directory
from src.preprocess import inspect_input, load_working_image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/external/u7_21a_strict_sdr_heic_v1"
POSITIVE = SOURCE_ROOT / "heif_other__arrow.heic"
NCLX = SOURCE_ROOT / "synthetic__srgb_nclx.heic"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_exact_display_p3_heic_enters_owned_linear_srgb() -> None:
    inspection = inspect_input(POSITIVE)
    assert inspection.source_kind == "raster"
    assert inspection.format_name == "HEIC"
    assert (inspection.width, inspection.height) == (3024, 4032)
    assert inspection.mode == "RGB"
    assert inspection.bit_depth == 8
    assert inspection.frame_count == 1
    assert inspection.has_alpha is False
    assert inspection.orientation == 1
    assert inspection.source_profile.kind == "icc"
    assert inspection.hdr_metadata["strict_sdr_heic"] == "accepted"

    working = load_working_image(POSITIVE)
    assert working.pixels.shape == (4032, 3024, 3)
    assert working.pixels.dtype == np.float32
    assert working.pixels.flags.owndata
    assert np.isfinite(working.pixels).all()
    assert working.working_space == "linear_srgb"
    assert working.transfer_state == "display_linear"
    assert working.orientation_applied is True
    assert _sha256(working.pixels) == (
        "068b9e4ed3b091be6ceefe3d75a814c5acaa3791f407bc705b6a201bcf52440a"
    )


def test_synthetic_exact_srgb_nclx_branch_is_explicit_and_bounded() -> None:
    inspection = inspect_input(NCLX)
    assert inspection.format_name == "HEIC"
    assert inspection.source_profile.kind == "nclx"
    assert inspection.hdr_metadata == {
        "strict_sdr_heic": "accepted",
        "colour_identity": "nclx",
    }
    assert not any(warning.code == "assumed_srgb" for warning in inspection.warnings)
    working = load_working_image(NCLX)
    assert working.pixels.shape == (29, 37, 3)
    assert _sha256(working.pixels) == (
        "18a18d7df1f284e76bba33c4121f539ef8cb7a60e5ee53a425478837767a4984"
    )


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("heif__RGB_10__128x128.heif", "8-bit"),
        ("heif__RGBA_8__128x128.heif", "RGB"),
        ("heif__RGB_8__128x128.heif", "colour|ICC|NCLX"),
        ("heif_other__nokia__bird_burst.heic", "sequence|still-image|one"),
        ("heif_other__pug.heic", "auxiliary|depth|gain"),
        ("heif_special__aux_YCbCr.heic", "auxiliary|gain"),
        ("heif_corrupted__corrupted.heic", "invalid|corrupt|input"),
        ("heif_truncated__truncated.heic", "sequence|one|truncated"),
    ],
)
def test_non_strict_heic_rejects_before_sdr_fallback(name: str, reason: str) -> None:
    path = SOURCE_ROOT / name
    inspection = inspect_input(path)
    assert inspection.format_name == "HEIC"
    assert inspection.hdr_metadata["strict_sdr_heic"] == "rejected"
    assert any(
        warning.code == "unsupported_dynamic_range" for warning in inspection.warnings
    )
    with pytest.raises(ValueError, match=reason):
        load_working_image(path)


def test_product_requirements_choose_decode_only_heif_dependency() -> None:
    requirements = (ROOT / "requirements-product-v2.txt").read_text(encoding="utf-8")
    assert "pi-heif==1.4.0" in requirements
    assert "pillow-heif" not in requirements.casefold()


def test_public_product_cli_recipe_replay_and_three_preview_accept_heic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ektar.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(NCLX),
            "--style",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--use-render-profile",
            "--render-profile",
            str(PRODUCT_PROFILE),
            "--write-recipe",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["render"]["style"] == "ektar_100"
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    replay = tmp_path / "replay.png"
    assert (
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=PRODUCT_PROFILE,
            output_path=replay,
            root=ROOT,
        )
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert replay.read_bytes() == output.read_bytes()

    manifest = render_three_stock_previews_to_directory(
        NCLX,
        tmp_path / "previews",
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=1_073,
        tile_size=16,
        tile_workers=1,
        include_input_preview=True,
    )
    assert len(manifest["rows"]) == 3
    assert {row["style_id"] for row in manifest["rows"]} == {
        "velvia_50",
        "portra_400",
        "ektar_100",
    }
    assert all(Path(row["output_path"]).is_file() for row in manifest["rows"])
