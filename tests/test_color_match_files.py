from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import tifffile
from jsonschema import Draft202012Validator, ValidationError
from PIL import Image

from src.color_match import (
    REFERENCE_FILE_OUTPUT_CAPABILITIES_ID,
    ReferenceMatchContractError,
    inspect_reference_file_input,
    inspect_reference_file_inputs,
    load_reference_look_recipe,
    match_reference_files,
    reference_file_output_capabilities,
    reference_file_output_capabilities_payload,
    resolve_reference_file_output_capability,
)
from src.preprocess import (
    SourceProfile,
    WorkingImage,
    inspect_input,
    load_working_image,
    save_rec2020_16_png,
)


def _image(path: Path, seed: int, *, format_name: str | None = None) -> None:
    array = np.random.default_rng(seed).integers(
        20,
        221,
        size=(31, 37, 3),
        dtype=np.uint8,
    )
    Image.fromarray(array, mode="RGB").save(path, format=format_name)


def _rec2020_image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).uniform(
        0.02,
        0.98,
        size=(31, 37, 3),
    ).astype(np.float32)
    save_rec2020_16_png(
        WorkingImage(
            pixels=pixels,
            working_space="linear_rec2020",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("cicp", "BT.2020 SDR test"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=16,
            source_path=path,
            warnings=[],
        ),
        path,
    )


def test_file_output_capabilities_are_exact_and_versioned() -> None:
    assert REFERENCE_FILE_OUTPUT_CAPABILITIES_ID == (
        "neuro-film.reference-file-output-capabilities.v1"
    )
    capabilities = reference_file_output_capabilities()
    assert [
        (
            row.working_space,
            row.transfer_state,
            row.output_bit_depth,
            row.extensions,
            row.encoding_profile,
        )
        for row in capabilities
    ] == [
        (
            "linear_srgb",
            "display_linear",
            8,
            (".jpeg", ".jpg", ".png", ".tif", ".tiff"),
            "srgb-icc.v1",
        ),
        (
            "linear_srgb",
            "display_linear",
            16,
            (".png", ".tif", ".tiff"),
            "srgb-icc.v1",
        ),
        (
            "linear_rec2020",
            "display_linear",
            16,
            (".png",),
            "bt2020-sdr-cicp-1-1-0-1.v1",
        ),
    ]


def test_file_output_capability_resolver_uses_the_exact_public_matrix() -> None:
    row = resolve_reference_file_output_capability(
        working_space="linear_rec2020",
        transfer_state="display_linear",
        output_bit_depth=16,
        output_extension=".PNG",
    )
    assert row == reference_file_output_capabilities()[2]

    with pytest.raises(
        ReferenceMatchContractError,
        match="requires 16-bit PNG",
    ):
        resolve_reference_file_output_capability(
            working_space="linear_rec2020",
            transfer_state="display_linear",
            output_bit_depth=8,
            output_extension=".png",
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match="dot-prefixed",
    ):
        resolve_reference_file_output_capability(
            working_space="linear_srgb",
            transfer_state="display_linear",
            output_bit_depth=16,
            output_extension="png",
        )


def test_file_output_capability_payload_validates_against_public_schema() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "schemas"
        / "reference_file_output_capabilities_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = reference_file_output_capabilities_payload()
    validator.validate(payload)
    mutated = json.loads(json.dumps(payload))
    mutated["capabilities"][2]["encoding_profile"] = "srgb-icc.v1"
    with pytest.raises(ValidationError):
        validator.validate(mutated)


def test_file_input_preflight_is_ordered_hash_bound_and_schema_valid(
    tmp_path: Path,
) -> None:
    srgb = tmp_path / "srgb.png"
    rec2020 = tmp_path / "rec2020.png"
    missing = tmp_path / "missing.png"
    _image(srgb, 28701)
    _rec2020_image(rec2020, 28702)

    payload = inspect_reference_file_inputs([srgb, rec2020, missing])
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "schemas"
        / "reference_file_input_inspection_batch_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)

    assert [row["path"] for row in payload["inspections"]] == [
        str(path.resolve()) for path in (srgb, rec2020, missing)
    ]
    assert [row["accepted"] for row in payload["inspections"]] == [
        True,
        True,
        False,
    ]
    assert payload["inspections"][0]["working_space"] == "linear_srgb"
    assert payload["inspections"][1]["working_space"] == "linear_rec2020"
    assert payload["inspections"][2]["failure_code"] == "not-a-file"


def test_file_input_preflight_detects_mutation_during_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _image(source, 28703)
    from src.color_match import files as files_module

    real_load = files_module.load_working_image

    def mutate_after_decode(path):
        image = real_load(path)
        Path(path).write_bytes(Path(path).read_bytes() + b"\x00")
        return image

    monkeypatch.setattr(files_module, "load_working_image", mutate_after_decode)
    result = inspect_reference_file_input(source)
    assert result.accepted is False
    assert result.file_sha256 is None
    assert result.failure_code == "file-changed-during-inspection"


def test_file_input_preflight_structures_decoder_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.dng"
    source.write_bytes(b"bounded-invalid-raw")
    from src.color_match import files as files_module

    def unavailable_decoder(_path):
        raise RuntimeError("raw decoder unavailable")

    monkeypatch.setattr(
        files_module,
        "load_working_image",
        unavailable_decoder,
    )
    result = inspect_reference_file_input(source)
    assert result.accepted is False
    assert result.file_sha256 is not None
    assert result.working_space is None
    assert result.failure_code == "decode-or-color-state-rejected"


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
        row.safety.policy_id == "reference-render-guard.v2"
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


@pytest.mark.parametrize("changed_input", ["reference", "source"])
def test_file_adapter_rejects_input_changed_during_decode(
    tmp_path: Path,
    monkeypatch,
    changed_input: str,
) -> None:
    from src.color_match import files

    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    _image(reference, 27341)
    _image(source, 27342)
    changed_path = reference if changed_input == "reference" else source
    original_load = files.load_working_image

    def mutate_after_decode(path: Path):
        image = original_load(path)
        if Path(path) == changed_path:
            _image(changed_path, 27343)
        return image

    monkeypatch.setattr(files, "load_working_image", mutate_after_decode)
    with pytest.raises(
        ReferenceMatchContractError,
        match=f"{changed_input} file changed while it was being decoded",
    ):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
        )
    assert not output.exists()
    assert not recipe.exists()


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


def test_file_adapter_preserves_rec2020_sdr_boundary(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    report = tmp_path / "report.json"
    _rec2020_image(reference, 27346)
    _rec2020_image(source, 27347)

    result = match_reference_files(
        reference,
        [source],
        [output],
        report_path=report,
        output_bit_depth=16,
    )
    restored = load_working_image(output)
    report_payload = json.loads(report.read_text(encoding="utf-8"))

    assert result.recipe.reference_working_space == "linear_rec2020"
    assert result.outputs[0].diagnostics.source_working_space == (
        "linear_rec2020"
    )
    assert result.outputs[0].output_format == "PNG"
    assert restored.working_space == "linear_rec2020"
    assert restored.transfer_state == "display_linear"
    assert inspect_input(output).source_profile.kind == "cicp"
    assert report_payload["outputs"][0]["candidate_diagnostics"][
        "source_working_space"
    ] == "linear_rec2020"
    assert report_payload["outputs"][0]["output_sha256"] == (
        result.outputs[0].output_sha256
    )


@pytest.mark.parametrize(
    ("bit_depth", "suffix"),
    [(8, ".png"), (16, ".tiff")],
)
def test_rec2020_file_output_rejects_unsupported_encoding(
    tmp_path: Path,
    bit_depth: int,
    suffix: str,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / f"output{suffix}"
    recipe = tmp_path / "recipe.json"
    _rec2020_image(reference, 27348)
    _rec2020_image(source, 27349)

    with pytest.raises(
        ReferenceMatchContractError,
        match="linear_rec2020 file output requires 16-bit PNG",
    ):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
            output_bit_depth=bit_depth,
        )
    assert not output.exists()
    assert not recipe.exists()


def test_file_adapter_preserves_each_source_rail_in_mixed_sdr_batch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    srgb_source = tmp_path / "srgb-source.png"
    rec2020_source = tmp_path / "rec2020-source.png"
    srgb_output = tmp_path / "srgb-output.png"
    rec2020_output = tmp_path / "rec2020-output.png"
    _image(reference, 27350)
    _image(srgb_source, 27351)
    _rec2020_image(rec2020_source, 27352)

    result = match_reference_files(
        reference,
        [srgb_source, rec2020_source],
        [srgb_output, rec2020_output],
        output_bit_depth=16,
    )

    assert [row.diagnostics.source_working_space for row in result.outputs] == [
        "linear_srgb",
        "linear_rec2020",
    ]
    assert load_working_image(srgb_output).working_space == "linear_srgb"
    assert load_working_image(rec2020_output).working_space == (
        "linear_rec2020"
    )


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


def test_file_adapter_rejects_committed_bytes_that_differ_from_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.color_match import files

    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    _image(reference, 27344)
    _image(source, 27345)
    output.write_bytes(b"old-output")
    recipe.write_bytes(b"old-recipe")
    original_replace = files._replace

    def corrupt_after_publish(source_path: Path, destination: Path) -> None:
        original_replace(source_path, destination)
        if (
            destination == output
            and "reference-match-stage" in source_path.name
        ):
            destination.write_bytes(b"corrupted-after-publish")

    monkeypatch.setattr(files, "_replace", corrupt_after_publish)
    with pytest.raises(
        ReferenceMatchContractError,
        match="committed destination bytes differ from staged hash",
    ):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
        )
    assert output.read_bytes() == b"old-output"
    assert recipe.read_bytes() == b"old-recipe"
    assert not list(tmp_path.glob(".*.reference-match-stage.*"))
    assert not list(tmp_path.glob(".*.reference-match-backup"))


def test_match_releases_reference_pixels_before_source_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import files

    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    reference.write_bytes(b"reference")
    source.write_bytes(b"source")
    reference_ref: weakref.ReferenceType[WorkingImage] | None = None

    def load_reference(
        path: Path,
        *,
        label: str,
    ) -> tuple[WorkingImage, str]:
        nonlocal reference_ref
        assert path == reference
        assert label == "reference"
        image = WorkingImage(
            pixels=np.zeros((2, 3, 3), dtype=np.float32),
            working_space="linear_srgb",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("assumed_srgb", "lifetime test"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=8,
            source_path=path,
            warnings=[],
        )
        reference_ref = weakref.ref(image)
        return image, "1" * 64

    def fit_reference(
        image: WorkingImage,
        *,
        policy,
    ):
        assert reference_ref is not None
        assert reference_ref() is image
        return object()

    def execute_render(recipe, sources, outputs, **kwargs):
        assert reference_ref is not None
        assert reference_ref() is None
        assert sources == (source,)
        assert outputs == (output,)
        return (), None, None

    monkeypatch.setattr(files, "_load_stable_working_image", load_reference)
    monkeypatch.setattr(files, "fit_reference_look", fit_reference)
    monkeypatch.setattr(files, "_execute_file_render", execute_render)

    result = match_reference_files(reference, [source], [output])
    assert result.outputs == ()


def test_render_releases_source_pixels_before_output_encode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import files

    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    source_path.write_bytes(b"source")
    source_ref: weakref.ReferenceType[WorkingImage] | None = None
    output_image = WorkingImage(
        pixels=np.zeros((2, 3, 3), dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "output lifetime test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=output_path,
        warnings=[],
    )

    def load_source(
        path: Path,
        *,
        label: str,
    ) -> tuple[WorkingImage, str]:
        nonlocal source_ref
        assert path == source_path
        assert label == "source"
        image = WorkingImage(
            pixels=np.zeros((2, 3, 3), dtype=np.float32),
            working_space="linear_srgb",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("assumed_srgb", "source lifetime test"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=8,
            source_path=path,
            warnings=[],
        )
        source_ref = weakref.ref(image)
        return image, "2" * 64

    def render_guarded(recipe, source, **kwargs):
        assert source_ref is not None
        assert source_ref() is source
        return SimpleNamespace(
            image=output_image,
            candidate_diagnostics=object(),
            safety=object(),
        )

    def encode(image, destination, **kwargs):
        assert source_ref is not None
        assert source_ref() is None
        assert image is output_image
        destination.write_bytes(b"encoded")
        return "PNG", 0.0

    monkeypatch.setattr(files, "_load_stable_working_image", load_source)
    monkeypatch.setattr(files, "render_reference_look_guarded", render_guarded)
    monkeypatch.setattr(files, "_encode_working_image", encode)
    monkeypatch.setattr(files, "_commit_staged_batch", lambda *args, **kwargs: None)

    prepared, recipe_hash, report_hash = files._execute_file_render(
        object(),
        (source_path,),
        (output_path,),
        guard_policy=None,
        output_bit_depth=16,
    )
    assert len(prepared) == 1
    assert recipe_hash is None
    assert report_hash is None


def test_post_commit_backup_cleanup_retry_does_not_report_false_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.color_match import files

    destination = tmp_path / "output.bin"
    stage = tmp_path / ".output.token.reference-match-stage.bin"
    destination.write_bytes(b"old")
    stage.write_bytes(b"new")
    cleanup = [stage]
    original_unlink = Path.unlink
    failures = 0

    def fail_first_backup_unlink(
        path: Path,
        *args,
        **kwargs,
    ) -> None:
        nonlocal failures
        if (
            path.name.endswith(".reference-match-backup")
            and failures == 0
        ):
            failures += 1
            raise OSError("transient backup cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_backup_unlink)
    files._commit_staged_batch(
        ((stage, destination),),
        token="token",
        cleanup=cleanup,
    )
    assert failures == 1
    assert destination.read_bytes() == b"new"
    assert cleanup == []
    assert not list(tmp_path.glob(".*.reference-match-backup"))


def test_concurrent_batch_commit_cannot_publish_mixed_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.color_match import files

    destinations = (tmp_path / "one.bin", tmp_path / "two.bin")
    for index, destination in enumerate(destinations):
        destination.write_bytes(f"old-{index}".encode("ascii"))
    a_stages = (tmp_path / "a-one.stage", tmp_path / "a-two.stage")
    b_stages = (tmp_path / "b-one.stage", tmp_path / "b-two.stage")
    for stage, value in zip(a_stages, (b"A1", b"A2"), strict=True):
        stage.write_bytes(value)
    for stage, value in zip(b_stages, (b"B1", b"B2"), strict=True):
        stage.write_bytes(value)

    first_published = threading.Event()
    release_first = threading.Event()
    original_replace = files._replace
    failures: list[BaseException] = []

    def pause_after_first_publish(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if source == a_stages[0]:
            first_published.set()
            if not release_first.wait(5):
                raise TimeoutError("concurrency test release timed out")

    monkeypatch.setattr(files, "_replace", pause_after_first_publish)

    def commit_a() -> None:
        try:
            files._commit_staged_batch(
                tuple(zip(a_stages, destinations, strict=True)),
                token="a" * 32,
                cleanup=list(a_stages),
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=commit_a)
    thread.start()
    assert first_published.wait(5)
    with pytest.raises(
        ReferenceMatchContractError,
        match="already locked",
    ):
        files._commit_staged_batch(
            tuple(zip(b_stages, destinations, strict=True)),
            token="b" * 32,
            cleanup=list(b_stages),
        )
    release_first.set()
    thread.join(5)

    assert not thread.is_alive()
    assert failures == []
    assert tuple(path.read_bytes() for path in destinations) == (b"A1", b"A2")


def test_batch_commit_target_lock_is_cross_process(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    destination = tmp_path / "output.bin"
    stage = tmp_path / "output.stage"
    stage.write_bytes(b"new")
    script = "\n".join(
        (
            "import sys, time",
            "from pathlib import Path",
            "from src.color_match.transaction_lock "
            "import target_transaction_lock",
            "ready, release, target = map(Path, sys.argv[1:])",
            "with target_transaction_lock((target,)):",
            "    ready.write_text('ready', encoding='ascii')",
            "    deadline = time.monotonic() + 10",
            "    while not release.exists():",
            "        if time.monotonic() >= deadline:",
            "            raise TimeoutError('release timeout')",
            "        time.sleep(0.02)",
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(ready),
            str(release),
            str(destination),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("child lock acquisition timeout")
            time.sleep(0.02)
        assert ready.is_file()
        from src.color_match import files

        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            files._commit_staged_batch(
                ((stage, destination),),
                token="cross-process",
                cleanup=[stage],
            )
    finally:
        release.write_text("release", encoding="ascii")
        stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)
    assert not destination.exists()


def test_runtime_and_generic_transactions_share_one_lock_namespace(
    tmp_path: Path,
) -> None:
    from src.color_match import files
    from src.color_match.shared_runtime_staging_transaction import (
        _target_transaction_lock,
    )

    destination = tmp_path / "output.bin"
    stage = tmp_path / "output.stage"
    stage.write_bytes(b"new")

    with _target_transaction_lock((destination,)):
        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            files._commit_staged_batch(
                ((stage, destination),),
                token="cross-class",
                cleanup=[stage],
            )

    assert not destination.exists()


def test_transaction_lock_rejects_empty_and_canonical_duplicate_targets(
    tmp_path: Path,
) -> None:
    from src.color_match.transaction_lock import target_transaction_lock

    with pytest.raises(
        ReferenceMatchContractError,
        match="must not be empty",
    ):
        with target_transaction_lock(()):
            raise AssertionError("empty lock inventory must not be entered")

    target = tmp_path / "output.bin"
    alias = target.parent / "." / target.name
    with pytest.raises(
        ReferenceMatchContractError,
        match="must be unique",
    ):
        with target_transaction_lock((target, alias)):
            raise AssertionError("duplicate lock inventory must not be entered")


def test_transaction_lock_collides_real_and_symlink_aliases(
    tmp_path: Path,
) -> None:
    from src.color_match.transaction_lock import target_transaction_lock

    real_parent = tmp_path / "real"
    alias_parent = tmp_path / "alias"
    real_parent.mkdir()
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    real_target = real_parent / "output.bin"
    alias_target = alias_parent / "output.bin"
    with target_transaction_lock((real_target,)):
        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            with target_transaction_lock((alias_target,)):
                raise AssertionError("path alias must not bypass target lock")
