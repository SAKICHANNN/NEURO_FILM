from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time

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
