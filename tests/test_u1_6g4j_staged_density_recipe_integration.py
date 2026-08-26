from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.pipeline_color_baseline import load_guardrail_config
from src.filmfx import (
    STAGED_DENSITY_VERSION,
    composite_staged_density_rows,
    execute_staged_density_halation_default,
)
from src.inference import (
    RECIPE_SCHEMA_ID_V4,
    RenderContractError,
    load_render_profile,
    render_resolved_safe_lab_rgb,
    replay_style_safe_recipe,
    replay_style_safe_recipe_to_file,
    validate_render_recipe,
)
from src.preprocess import (
    load_working_image,
    save_srgb16_png,
    working_image_to_srgb_float,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:192, :256]
    rgb = np.stack(
        (
            (3 * xx + yy) % 256,
            (xx + 5 * yy) % 256,
            ((xx // 8) * 19 + (yy // 6) * 13) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _run_cli(source: Path, output: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        "velvia_50",
        "--use-render-profile",
        "--halation-model",
        "staged-density-research",
        "--halation",
        "1.0",
        "--tile-size",
        "64",
        "--output-bit-depth",
        "16",
        "--png-compression",
        "0",
        "--write-recipe",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))


def test_staged_density_cli_v4_replay_and_direct_executor_are_exact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "candidate.png"
    replay = tmp_path / "replay.png"
    direct = tmp_path / "direct.png"
    _source(source)
    recipe = _run_cli(source, output)

    assert recipe["schema_id"] == RECIPE_SCHEMA_ID_V4
    resolved = recipe["render"]["effects"]["halation"]["resolved_parameters"]
    assert resolved == {
        "executor_version": STAGED_DENSITY_VERSION,
        "tile_size": 64,
        "source_row_chunk": 64,
        "coarse_row_chunk": 7,
        "composite_row_chunk": 64,
    }
    validate_render_recipe(recipe)
    replay_style_safe_recipe_to_file(
        recipe,
        profile_path=PROFILE,
        output_path=replay,
        root=ROOT,
    )
    assert replay.read_bytes() == output.read_bytes()

    profile = load_render_profile(PROFILE, root=ROOT)
    working = load_working_image(source)
    statistics = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )
    base = render_resolved_safe_lab_rgb(
        working_image_to_srgb_float(working),
        style="velvia_50",
        style_statistics=statistics["styles"]["velvia_50"],
        style_parameters=profile["style_parameters"]["velvia_50"],
        guardrails=load_guardrail_config(
            ROOT / "configs/color_guardrails.json", "velvia_50"
        ),
        seed=7,
        tile_size=64,
    )
    layer, _ = execute_staged_density_halation_default(
        base, tile_size=64, source_row_chunk=64, coarse_row_chunk=7
    )
    direct_rgb = composite_staged_density_rows(
        base, layer, row_chunk=64, output_margin=4
    )
    save_srgb16_png(direct_rgb, direct, compression_level=0)
    assert direct.read_bytes() == output.read_bytes()


def test_staged_density_v4_mutations_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "candidate.png"
    _source(source)
    recipe = _run_cli(source, output)
    mutations = []
    for path, value in (
        (("schema_id",), "kmcfm.render-recipe.v3"),
        (("render", "effects", "halation", "model"), "physical"),
        (("render", "effects", "halation", "strength"), 0.5),
        (("render", "effects", "halation", "control_mode"), "expert"),
        (("render", "effects", "halation", "preset"), "generic"),
        (
            (
                "render",
                "effects",
                "halation",
                "resolved_parameters",
                "executor_version",
            ),
            "staged-density-halation-v0",
        ),
        (
            ("render", "effects", "halation", "resolved_parameters", "tile_size"),
            63,
        ),
        (
            (
                "render",
                "effects",
                "halation",
                "resolved_parameters",
                "coarse_row_chunk",
            ),
            8,
        ),
    ):
        mutation = copy.deepcopy(recipe)
        target = mutation
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(mutation)
    extra = copy.deepcopy(recipe)
    extra["render"]["effects"]["halation"]["resolved_parameters"]["unknown"] = 1
    mutations.append(extra)

    for mutation in mutations:
        with pytest.raises(RenderContractError):
            validate_render_recipe(mutation)
    with pytest.raises(Exception, match="tile size differs"):
        replay_style_safe_recipe(
            recipe,
            profile_path=PROFILE,
            root=ROOT,
            tile_size=63,
        )


def test_v4_schema_is_strict_and_does_not_change_legacy_schema_files() -> None:
    schema = json.loads(
        (ROOT / "configs/schemas/render_recipe_v4.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == RECIPE_SCHEMA_ID_V4
    halation = schema["properties"]["render"]["properties"]["effects"][
        "properties"
    ]["halation"]
    assert halation["properties"]["model"]["const"] == "staged-density-research"
