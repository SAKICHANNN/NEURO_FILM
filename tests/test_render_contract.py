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
    PROFILE_SCHEMA_ID,
    RECIPE_SCHEMA_ID,
    RenderContractError,
    build_render_recipe,
    load_render_profile,
    migrate_legacy_safe_rich,
    validate_render_profile,
    validate_render_recipe,
    verify_render_recipe_files,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_v1.json"


def _profile() -> dict:
    return load_render_profile(PROFILE, root=ROOT)


def test_tracked_schemas_are_strict_and_profile_is_exact_migration() -> None:
    profile_schema = json.loads((ROOT / "configs" / "schemas" / "render_profile_v1.schema.json").read_text())
    recipe_schema = json.loads((ROOT / "configs" / "schemas" / "render_recipe_v1.schema.json").read_text())
    assert profile_schema["additionalProperties"] is False
    assert recipe_schema["additionalProperties"] is False
    assert profile_schema["properties"]["schema_id"]["const"] == PROFILE_SCHEMA_ID
    assert recipe_schema["properties"]["schema_id"]["const"] == RECIPE_SCHEMA_ID
    migrated = migrate_legacy_safe_rich(
        ROOT / "configs" / "color_rendering_profiles.yaml",
        ROOT / "configs" / "film_color_stats.json",
        ROOT / "configs" / "color_guardrails.json",
        root=ROOT,
    )
    assert _profile() == migrated


def test_profile_rejects_unknown_keys_nonfinite_values_and_claim_escalation() -> None:
    profile = _profile()
    unknown = copy.deepcopy(profile)
    unknown["surprise"] = True
    with pytest.raises(RenderContractError, match="keys mismatch"):
        validate_render_profile(unknown)
    nonfinite = copy.deepcopy(profile)
    nonfinite["style_parameters"]["velvia_50"]["strength"] = float("nan")
    with pytest.raises(RenderContractError, match="finite"):
        validate_render_profile(nonfinite)
    escalation = copy.deepcopy(profile)
    escalation["evidence"]["calibrated_reference_allowed"] = True
    with pytest.raises(RenderContractError, match="held-out S3"):
        validate_render_profile(escalation)


def test_profile_asset_hash_mismatch_fails_closed() -> None:
    profile = _profile()
    profile["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(RenderContractError, match="asset hash mismatch"):
        validate_render_profile(profile, root=ROOT)


def test_recipe_builder_hashes_files_and_rejects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    output = tmp_path / "output.png"
    source.write_bytes(b"input")
    output.write_bytes(b"output")
    profile = _profile()
    recipe = build_render_recipe(
        profile_path=PROFILE,
        profile=profile,
        input_path=source,
        input_metadata={
            "color_state": "display_referred",
            "working_space": "linear_srgb",
            "source_profile_kind": "assumed_srgb",
            "source_profile_fingerprint_sha256": None,
            "bit_depth": 8,
            "warnings": [],
        },
        render_metadata={
            "engine_id": "safe_lab_v1",
            "preset": "safe-rich",
            "style": "velvia_50",
            "seed": 7,
            "color_parameters": profile["style_parameters"]["velvia_50"],
            "effects": {
                "grain": {"strength": 0.0, "seed": 7, "color": True},
                "halation": {"strength": 0.0, "model": "simple", "preset": None, "control_mode": "locked", "resolved_parameters": None},
                "dust": {"strength": 0.0, "seed": 24},
            },
        },
        output_path=output,
        output_format="PNG",
        output_bit_depth=8,
        output_icc_fingerprint_sha256="1" * 64,
        output_claim={
            "render_mode": "Style-safe",
            "output_label": "film-inspired",
            "evidence_grade": "look-approximation",
            "input_color_state": "display_referred",
            "color_state_policy": "look_approximation_only",
            "calibrated_reference_allowed": False,
        },
        software_commit="2" * 40,
    )
    assert recipe["input"]["sha256"] == hashlib.sha256(b"input").hexdigest()
    assert recipe["output"]["sha256"] == hashlib.sha256(b"output").hexdigest()
    validate_render_recipe(recipe)
    verify_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)
    mutation = copy.deepcopy(recipe)
    mutation["render"]["effects"]["grain"]["strength"] = float("inf")
    with pytest.raises(RenderContractError, match="finite"):
        validate_render_recipe(mutation)
    mutation = copy.deepcopy(recipe)
    mutation["claim"]["calibrated_reference_allowed"] = True
    with pytest.raises(RenderContractError, match="calibrated Reference"):
        validate_render_recipe(mutation)
    output.write_bytes(b"changed")
    with pytest.raises(RenderContractError, match="output file hash mismatch"):
        verify_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)


def test_renderer_recipe_is_opt_in_and_does_not_change_output(tmp_path: Path) -> None:
    pixels = np.zeros((20, 24, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(24, dtype=np.uint8)[None, :] * 10
    pixels[..., 1] = 96
    pixels[..., 2] = np.arange(20, dtype=np.uint8)[:, None] * 11
    source = tmp_path / "input.png"
    plain = tmp_path / "plain.png"
    replay = tmp_path / "replay.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    base = [sys.executable, str(ROOT / "scripts" / "render_film.py"), str(source)]
    first = subprocess.run(base + ["--output", str(plain)], cwd=ROOT, capture_output=True, text=True)
    second = subprocess.run(
        base + ["--output", str(replay), "--write-recipe", "--write-metrics"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert plain.read_bytes() == replay.read_bytes()
    assert not plain.with_suffix(".recipe.json").exists()
    recipe_path = replay.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    validate_render_recipe(recipe)
    verify_render_recipe_files(recipe, profile_path=PROFILE, root=ROOT)
    assert recipe["schema_id"] == RECIPE_SCHEMA_ID
    assert recipe["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert recipe["output"]["sha256"] == hashlib.sha256(replay.read_bytes()).hexdigest()
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    metrics = json.loads(replay.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["render_recipe"]["schema_id"] == RECIPE_SCHEMA_ID
    assert metrics["render_recipe"]["sha256"] == hashlib.sha256(recipe_path.read_bytes()).hexdigest()
