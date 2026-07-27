from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import src.color_match.files as file_module
import src.color_match.reporting as reporting_module
from src.color_match import (
    ReferenceMatchContractError,
    match_reference_files,
    replay_reference_files,
)


def _image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).integers(
        16,
        240,
        size=(27, 35, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def _assert_no_transaction_debris(root: Path) -> None:
    assert not list(root.glob(".*.reference-match-stage*"))
    assert not list(root.glob(".*.reference-match-backup"))


def test_fit_report_path_collision_rejects_before_any_artifact(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    _image(reference, 27901)
    _image(source, 27902)
    with pytest.raises(ReferenceMatchContractError, match="report.*overwrite"):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
            report_path=output,
        )
    assert not output.exists()
    assert not recipe.exists()
    _assert_no_transaction_debris(tmp_path)


def test_report_payload_failure_leaves_existing_fit_artifacts_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27903)
    _image(source, 27904)
    old = {
        output: b"old-output",
        recipe: b"old-recipe",
        report: b"old-report",
    }
    for path, content in old.items():
        path.write_bytes(content)

    def fail_report(_result):
        raise ReferenceMatchContractError("injected report build failure")

    monkeypatch.setattr(
        reporting_module,
        "build_file_match_report",
        fail_report,
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="injected report build failure",
    ):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
            report_path=report,
        )
    for path, content in old.items():
        assert path.read_bytes() == content
    _assert_no_transaction_debris(tmp_path)


def test_fit_report_commit_failure_rolls_back_outputs_recipe_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27905)
    _image(source, 27906)
    old = {
        output: b"old-output",
        recipe: b"old-recipe",
        report: b"old-report",
    }
    for path, content in old.items():
        path.write_bytes(content)
    original_replace = file_module._replace

    def fail_report_stage(source_path: Path, destination: Path) -> None:
        if (
            destination == report
            and "reference-match-stage" in source_path.name
        ):
            raise OSError("injected report commit failure")
        original_replace(source_path, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report_stage)
    with pytest.raises(OSError, match="injected report commit failure"):
        match_reference_files(
            reference,
            [source],
            [output],
            recipe_path=recipe,
            report_path=report,
        )
    for path, content in old.items():
        assert path.read_bytes() == content
    _assert_no_transaction_debris(tmp_path)


def test_replay_report_commit_failure_rolls_back_output_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.png"
    fit_source = tmp_path / "fit-source.png"
    recipe = tmp_path / "recipe.json"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    report = tmp_path / "report.json"
    _image(reference, 27907)
    _image(fit_source, 27908)
    _image(source, 27909)
    match_reference_files(
        reference,
        [fit_source],
        [tmp_path / "fit-output.png"],
        recipe_path=recipe,
    )
    recipe_before = recipe.read_bytes()
    output.write_bytes(b"old-output")
    report.write_bytes(b"old-report")
    original_replace = file_module._replace

    def fail_report_stage(source_path: Path, destination: Path) -> None:
        if (
            destination == report
            and "reference-match-stage" in source_path.name
        ):
            raise OSError("injected replay report commit failure")
        original_replace(source_path, destination)

    monkeypatch.setattr(file_module, "_replace", fail_report_stage)
    with pytest.raises(
        OSError,
        match="injected replay report commit failure",
    ):
        replay_reference_files(
            recipe,
            [source],
            [output],
            report_path=report,
        )
    assert output.read_bytes() == b"old-output"
    assert report.read_bytes() == b"old-report"
    assert recipe.read_bytes() == recipe_before
    _assert_no_transaction_debris(tmp_path)


def test_transactional_report_hash_matches_committed_bytes(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27910)
    _image(source, 27911)
    result = match_reference_files(
        reference,
        [source],
        [output],
        recipe_path=recipe,
        report_path=report,
    )
    from src.inference import sha256_file

    assert result.report_path == report
    assert result.report_file_sha256 == sha256_file(report)
    assert result.outputs[0].output_sha256 == sha256_file(output)
    _assert_no_transaction_debris(tmp_path)
