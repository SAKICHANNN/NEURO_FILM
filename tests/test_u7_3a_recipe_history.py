from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.inference import (
    RECIPE_HISTORY_SCHEMA_ID,
    RecipeHistoryError,
    build_render_recipe_history,
    load_render_profile,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _recipe(*, style: str = "velvia_50") -> dict:
    profile = load_render_profile(PROFILE_PATH, root=ROOT)
    return {
        "schema_id": "kmcfm.render-recipe.v1",
        "profile": {
            "profile_id": profile["profile_id"],
            "profile_version": profile["profile_version"],
            "sha256": "1" * 64,
        },
        "assets": copy.deepcopy(profile["assets"]),
        "input": {
            "path": "Z:/files/that/do/not/exist/input.png",
            "sha256": "2" * 64,
            "color_state": "display_referred",
            "working_space": "linear_srgb",
            "source_profile_kind": "assumed_srgb",
            "source_profile_fingerprint_sha256": None,
            "bit_depth": 8,
            "warnings": [],
        },
        "render": {
            "engine_id": "safe_lab_v1",
            "preset": "safe-rich",
            "style": style,
            "seed": 17,
            "color_parameters": copy.deepcopy(profile["style_parameters"][style]),
            "effects": {
                "grain": {"strength": 0.25, "seed": 17, "color": False},
                "halation": {
                    "strength": 0.0,
                    "model": "simple",
                    "preset": None,
                    "control_mode": "locked",
                    "resolved_parameters": None,
                },
                "dust": {"strength": 0.0, "seed": 19},
            },
        },
        "output": {
            "path": "Z:/files/that/do/not/exist/output.png",
            "sha256": "3" * 64,
            "format": "PNG",
            "bit_depth": 16,
            "transfer": "sRGB",
            "icc_profile_fingerprint_sha256": "4" * 64,
        },
        "claim": {
            "render_mode": "Style-safe",
            "output_label": "film-inspired",
            "evidence_grade": "look-approximation",
            "input_color_state": "display_referred",
            "color_state_policy": "look_approximation_only",
            "calibrated_reference_allowed": False,
            "claim_ceiling": "test-only look approximation",
        },
        "software": {"commit": "5" * 40},
    }


def _write_recipe(path: Path, value: object) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def test_catalog_summarizes_valid_recipe_without_reading_referenced_files(
    tmp_path: Path,
) -> None:
    payload = _write_recipe(tmp_path / "nested/z.recipe.json", _recipe())
    catalog = build_render_recipe_history(tmp_path)
    assert catalog["schema_id"] == RECIPE_HISTORY_SCHEMA_ID
    assert catalog["status"] == "ready"
    assert catalog["counts"] == {"discovered": 1, "valid": 1, "invalid": 0}
    row = catalog["entries"][0]
    assert row["recipe_path"] == "nested/z.recipe.json"
    assert row["recipe_sha256"]
    assert row["style"] == "velvia_50"
    assert row["enabled_effects"] == ["grain"]
    assert row["input_path"].startswith("Z:/files/that/do/not/exist")
    assert row["output_path"].startswith("Z:/files/that/do/not/exist")
    assert len(payload) < 2 * 1024 * 1024


def test_invalid_rows_are_isolated_and_paths_are_sorted(tmp_path: Path) -> None:
    (tmp_path / "a.recipe.json").write_text("{", encoding="utf-8")
    invalid = _recipe()
    invalid["claim"]["calibrated_reference_allowed"] = True
    _write_recipe(tmp_path / "b.recipe.json", invalid)
    _write_recipe(tmp_path / "c.recipe.json", _recipe(style="portra_400"))
    catalog = build_render_recipe_history(tmp_path)
    assert catalog["status"] == "partial"
    assert catalog["counts"] == {"discovered": 3, "valid": 1, "invalid": 2}
    assert [row["recipe_path"] for row in catalog["entries"]] == [
        "a.recipe.json",
        "b.recipe.json",
        "c.recipe.json",
    ]
    assert [row["error_code"] for row in catalog["entries"][:2]] == [
        "recipe_invalid_json",
        "recipe_contract_invalid",
    ]


def test_empty_root_returns_explicit_empty_state(tmp_path: Path) -> None:
    assert build_render_recipe_history(tmp_path) == {
        "schema_id": RECIPE_HISTORY_SCHEMA_ID,
        "status": "empty",
        "counts": {"discovered": 0, "valid": 0, "invalid": 0},
        "entries": [],
    }


def test_file_limit_fails_closed(tmp_path: Path) -> None:
    _write_recipe(tmp_path / "a.recipe.json", _recipe())
    _write_recipe(tmp_path / "b.recipe.json", _recipe())
    with pytest.raises(RecipeHistoryError, match="file limit exceeded"):
        build_render_recipe_history(tmp_path, maximum_recipe_files=1)


def test_oversized_and_invalid_utf8_recipes_are_isolated(tmp_path: Path) -> None:
    (tmp_path / "a.recipe.json").write_bytes(b"x" * 33)
    (tmp_path / "b.recipe.json").write_bytes(b"\xff")
    catalog = build_render_recipe_history(tmp_path, maximum_recipe_bytes=32)
    assert catalog["status"] == "invalid"
    assert [row["error_code"] for row in catalog["entries"]] == [
        "recipe_too_large",
        "recipe_invalid_utf8",
    ]


@pytest.mark.parametrize(
    ("files", "size"),
    [(0, 10), (1, 0), (True, 10), (1, True)],
)
def test_invalid_scan_bounds_fail_closed(tmp_path: Path, files: int, size: int) -> None:
    with pytest.raises(RecipeHistoryError, match="positive integer"):
        build_render_recipe_history(
            tmp_path,
            maximum_recipe_files=files,
            maximum_recipe_bytes=size,
        )
