from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.color_match import (
    ReferenceMatchContractError,
    fit_reference_look,
    load_reference_look_recipe,
    replay_reference_batch,
    render_reference_batch,
    save_reference_look_recipe,
)
from src.preprocess.types import SourceProfile, WorkingImage


def _working(seed: int, path: str) -> WorkingImage:
    pixels = np.random.default_rng(seed).uniform(
        0.08,
        0.82,
        size=(13, 17, 3),
    ).astype(np.float32)
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "test fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path(path),
    )


def test_recipe_file_save_is_repeat_exact_and_loads(tmp_path: Path) -> None:
    recipe = fit_reference_look(_working(27201, "reference.png"))
    path = tmp_path / "look.json"
    first_hash = save_reference_look_recipe(recipe, path)
    first_bytes = path.read_bytes()
    second_hash = save_reference_look_recipe(recipe, path)
    assert path.read_bytes() == first_bytes
    assert second_hash == first_hash
    assert load_reference_look_recipe(path) == recipe
    assert not list(tmp_path.glob(".look.json.*.tmp"))


def test_loaded_recipe_replay_matches_in_memory_batch_bytes(tmp_path: Path) -> None:
    recipe = fit_reference_look(_working(27202, "reference.png"))
    sources = [_working(27203, "a.png"), _working(27204, "b.png")]
    path = tmp_path / "look.json"
    save_reference_look_recipe(recipe, path)
    expected = render_reference_batch(recipe, sources)
    replayed = replay_reference_batch(path, sources)
    assert [row.image.pixels.tobytes() for row in replayed] == [
        row.image.pixels.tobytes() for row in expected
    ]
    assert [row.diagnostics for row in replayed] == [
        row.diagnostics for row in expected
    ]


def test_load_rejects_tampered_recipe_id(tmp_path: Path) -> None:
    recipe = fit_reference_look(_working(27205, "reference.png"))
    path = tmp_path / "look.json"
    save_reference_look_recipe(recipe, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["destination_lab_mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceMatchContractError, match="does not match canonical payload"):
        load_reference_look_recipe(path)


@pytest.mark.parametrize("name", ["missing.json", "directory"])
def test_load_rejects_non_file_path(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if name == "directory":
        path.mkdir()
    with pytest.raises(ReferenceMatchContractError, match="existing file"):
        load_reference_look_recipe(path)


def test_load_rejects_oversized_recipe_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ReferenceMatchContractError, match="bounded contract"):
        load_reference_look_recipe(path)


def test_save_rejects_directory_target(tmp_path: Path) -> None:
    with pytest.raises(ReferenceMatchContractError, match="must not be a directory"):
        save_reference_look_recipe(
            fit_reference_look(_working(27206, "reference.png")),
            tmp_path,
        )
