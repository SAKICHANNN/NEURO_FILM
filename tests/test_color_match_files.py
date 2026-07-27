from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from src.color_match import (
    ReferenceMatchContractError,
    load_reference_look_recipe,
    match_reference_files,
)
from src.preprocess import inspect_input


def _image(path: Path, seed: int, *, format_name: str | None = None) -> None:
    array = np.random.default_rng(seed).integers(
        20,
        221,
        size=(31, 37, 3),
        dtype=np.uint8,
    )
    Image.fromarray(array, mode="RGB").save(path, format=format_name)


def test_file_adapter_matches_png_jpeg_tiff_batch_and_saves_recipe(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    sources = [
        tmp_path / "a.png",
        tmp_path / "b.jpg",
        tmp_path / "c.tiff",
    ]
    _image(reference, 27301)
    _image(sources[0], 27302)
    _image(sources[1], 27303)
    _image(sources[2], 27304)
    outputs = [
        tmp_path / "out-a.png",
        tmp_path / "out-b.png",
        tmp_path / "out-c.tiff",
    ]
    recipe_path = tmp_path / "reference-look.json"
    result = match_reference_files(
        reference,
        sources,
        outputs,
        recipe_path=recipe_path,
        output_bit_depth=16,
    )
    assert result.recipe_path == recipe_path
    assert result.recipe_file_sha256 is not None
    assert load_reference_look_recipe(recipe_path) == result.recipe
    assert len(result.outputs) == 3
    assert [row.source_path for row in result.outputs] == sources
    assert [row.output_path for row in result.outputs] == outputs
    assert all(row.output_bit_depth == 16 for row in result.outputs)
    assert all(row.diagnostics.recipe_id == result.recipe.recipe_id for row in result.outputs)
    assert all(
        row.safety.policy_id == "reference-render-guard.v1"
        for row in result.outputs
    )
    assert all(
        row.safety.action in {"applied", "identity-fallback"}
        for row in result.outputs
    )
    assert inspect_input(outputs[0]).bit_depth == 16
    assert inspect_input(outputs[2]).bit_depth == 16
    with tifffile.TiffFile(outputs[2]) as tif:
        assert tif.pages[0].dtype == np.uint16


def test_file_adapter_repeat_is_byte_deterministic(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    _image(reference, 27305)
    _image(source, 27306)
    first_output = tmp_path / "first.png"
    second_output = tmp_path / "second.png"
    first_recipe = tmp_path / "first.json"
    second_recipe = tmp_path / "second.json"
    first = match_reference_files(
        reference,
        [source],
        [first_output],
        recipe_path=first_recipe,
        output_bit_depth=16,
    )
    second = match_reference_files(
        reference,
        [source],
        [second_output],
        recipe_path=second_recipe,
        output_bit_depth=16,
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_recipe.read_bytes() == second_recipe.read_bytes()
    assert first.outputs[0].output_sha256 == second.outputs[0].output_sha256
    assert first.recipe.recipe_id == second.recipe.recipe_id


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".tiff"])
def test_file_adapter_supports_srgb8_outputs(tmp_path: Path, suffix: str) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / f"output{suffix}"
    _image(reference, 27307)
    _image(source, 27308)
    result = match_reference_files(
        reference,
        [source],
        [output],
        output_bit_depth=8,
    )
    assert result.outputs[0].output_bit_depth == 8
    assert inspect_input(output).bit_depth == 8


def test_file_adapter_rejects_output_input_collision_before_writing(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    _image(reference, 27309)
    _image(source, 27310)
    source_before = source.read_bytes()
    with pytest.raises(ReferenceMatchContractError, match="must not overwrite"):
        match_reference_files(reference, [source], [source])
    assert source.read_bytes() == source_before


def test_file_adapter_rejects_duplicate_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    output = tmp_path / "output.png"
    _image(reference, 27311)
    _image(a, 27312)
    _image(b, 27313)
    with pytest.raises(ReferenceMatchContractError, match="must be unique"):
        match_reference_files(reference, [a, b], [output, output])


def test_file_adapter_cleans_all_staging_files_on_late_source_failure(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    output_a = tmp_path / "output-a.png"
    output_b = tmp_path / "output-b.png"
    recipe = tmp_path / "look.json"
    _image(reference, 27314)
    _image(good, 27315)
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported raster input"):
        match_reference_files(
            reference,
            [good, bad],
            [output_a, output_b],
            recipe_path=recipe,
        )
    assert not output_a.exists()
    assert not output_b.exists()
    assert not recipe.exists()
    assert not list(tmp_path.glob(".*.reference-match-stage.*"))


def test_file_adapter_rejects_16_bit_jpeg(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    _image(reference, 27316)
    _image(source, 27317)
    with pytest.raises(
        ReferenceMatchContractError,
        match="unsupported 16-bit output extension",
    ):
        match_reference_files(
            reference,
            [source],
            [tmp_path / "output.jpg"],
            output_bit_depth=16,
        )


def test_file_adapter_rolls_back_all_prior_outputs_on_commit_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.color_match import files

    reference = tmp_path / "reference.png"
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    output_a = tmp_path / "output-a.png"
    output_b = tmp_path / "output-b.png"
    recipe = tmp_path / "look.json"
    _image(reference, 27318)
    _image(source_a, 27319)
    _image(source_b, 27320)
    output_a.write_bytes(b"old-output-a")
    output_b.write_bytes(b"old-output-b")
    recipe.write_bytes(b"old-recipe")
    original = files._replace

    def fail_second_stage(source: Path, destination: Path) -> None:
        if (
            destination == output_b
            and "reference-match-stage" in source.name
        ):
            raise OSError("injected final-commit failure")
        original(source, destination)

    monkeypatch.setattr(files, "_replace", fail_second_stage)
    with pytest.raises(OSError, match="injected final-commit failure"):
        match_reference_files(
            reference,
            [source_a, source_b],
            [output_a, output_b],
            recipe_path=recipe,
        )
    assert output_a.read_bytes() == b"old-output-a"
    assert output_b.read_bytes() == b"old-output-b"
    assert recipe.read_bytes() == b"old-recipe"
    assert not list(tmp_path.glob(".*.reference-match-stage.*"))
    assert not list(tmp_path.glob(".*.reference-match-backup"))
