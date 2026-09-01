from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import (
    RenderContractError,
    build_render_recipe,
    load_render_profile,
    verify_render_recipe_inputs,
)
from src.inference.style_safe_engine import replay_style_safe_color_recipe
from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2j_product_recipe_catalog_enforcement_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
ERROR = "available product-catalog look"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run_recipe(
    source: Path, output: Path, *, profile: Path, style: str
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--style",
            style,
            "--use-render-profile",
            "--render-profile",
            str(profile),
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
    return json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))


def _dummy_render(style: str) -> dict[str, object]:
    return {
        "engine_id": "safe_lab_v1",
        "preset": "safe-rich",
        "style": style,
        "seed": 7,
        "color_parameters": {},
        "effects": {
            "grain": {"strength": 0.0, "seed": 7, "color": True},
            "halation": {
                "strength": 0.0,
                "model": "simple",
                "preset": None,
                "control_mode": "locked",
                "resolved_parameters": None,
            },
            "dust": {"strength": 0.0, "seed": 24},
        },
    }


def test_contract_binds_historical_core_and_parent_evidence() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    locks = config["source_locks"]
    assert locks["render_contract_sha256"] == (
        "9da36402186feb2a390d5a667b854195fc37b0b47fd148672ae88b3d9932b27d"
    )
    assert_historical_evidence_binding(
        ROOT,
        {
            "path": "src/inference/style_safe_engine.py",
            "sha256": locks["style_safe_engine_sha256"],
        },
    )
    assert _sha(PRODUCT_PROFILE) == locks["product_profile_sha256"]
    assert (
        _sha(ROOT / "docs/evidence/U7_2I_PRODUCT_LOOK_CATALOG_ENFORCEMENT_RESULT.json")
        == locks["parent_u7_2i_evidence_sha256"]
    )


@pytest.mark.parametrize(
    "style",
    [
        "hp5",
        "tri_x_400",
        "vision3_250d",
        "vision3_500t",
        "portra_800",
        "generic_bw",
        "unknown",
        "",
    ],
)
def test_product_recipe_builder_rejects_non_catalog_style_before_file_reads(
    tmp_path: Path, style: str
) -> None:
    profile = load_render_profile(PRODUCT_PROFILE, root=ROOT)
    with pytest.raises(RenderContractError, match=ERROR):
        build_render_recipe(
            profile_path=PRODUCT_PROFILE,
            profile=profile,
            input_path=tmp_path / "must_not_hash_input.png",
            input_metadata={},
            render_metadata=_dummy_render(style),
            output_path=tmp_path / "must_not_hash_output.png",
            output_format="PNG",
            output_bit_depth=8,
            output_icc_fingerprint_sha256="1" * 64,
            output_claim={},
            software_commit="2" * 40,
        )


@pytest.mark.parametrize("style", ["hp5", "tri_x_400", "vision3_500t", "generic_bw"])
def test_forged_product_recipe_rejects_before_input_hash_or_decode(
    tmp_path: Path, style: str
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "legacy.png"
    _source(source)
    recipe = _run_recipe(source, output, profile=LEGACY_PROFILE, style="hp5")
    product = load_render_profile(PRODUCT_PROFILE, root=ROOT)
    forged = copy.deepcopy(recipe)
    forged["profile"] = {
        "profile_id": product["profile_id"],
        "profile_version": product["profile_version"],
        "sha256": _sha(PRODUCT_PROFILE),
    }
    forged["assets"] = copy.deepcopy(product["assets"])
    forged["render"]["style"] = style
    forged["render"]["color_parameters"] = copy.deepcopy(
        product["style_parameters"][style]
    )
    forged["claim"]["claim_ceiling"] = product["evidence"]["claim_ceiling"]
    forged["input"]["path"] = str(tmp_path / "must_not_hash_or_decode.png")

    with pytest.raises(RenderContractError, match=ERROR):
        verify_render_recipe_inputs(forged, profile_path=PRODUCT_PROFILE, root=ROOT)
    replay_error = (
        ERROR if style != "generic_bw" else "product look 'generic_bw' is unavailable"
    )
    with pytest.raises(ValueError, match=replay_error):
        replay_style_safe_color_recipe(forged, profile_path=PRODUCT_PROFILE, root=ROOT)


@pytest.mark.parametrize("style", ["velvia_50", "portra_400", "ektar_100"])
def test_available_product_recipes_still_verify_and_replay(
    tmp_path: Path, style: str
) -> None:
    source = tmp_path / f"{style}-source.png"
    output = tmp_path / f"{style}.png"
    _source(source)
    recipe = _run_recipe(source, output, profile=PRODUCT_PROFILE, style=style)
    verify_render_recipe_inputs(recipe, profile_path=PRODUCT_PROFILE, root=ROOT)
    replay = replay_style_safe_color_recipe(
        recipe, profile_path=PRODUCT_PROFILE, root=ROOT
    )
    assert replay.shape == (47, 61, 3)
    assert replay.dtype == np.float32
    assert np.isfinite(replay).all()


def test_legacy_hp5_recipe_still_verifies_and_replays(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "hp5.png"
    _source(source)
    recipe = _run_recipe(source, output, profile=LEGACY_PROFILE, style="hp5")
    verify_render_recipe_inputs(recipe, profile_path=LEGACY_PROFILE, root=ROOT)
    replay = replay_style_safe_color_recipe(
        recipe, profile_path=LEGACY_PROFILE, root=ROOT
    )
    assert replay.shape == (47, 61, 3)
    assert replay.dtype == np.float32
    assert np.isfinite(replay).all()
