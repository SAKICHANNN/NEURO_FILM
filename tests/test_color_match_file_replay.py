from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
from PIL import Image
import numpy as np
import pytest
from referencing import Registry, Resource

from src.color_match import (
    REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID,
    ReferenceMatchContractError,
    ReferenceRenderGuardPolicy,
    build_file_replay_report,
    load_reference_look_recipe_bound,
    match_reference_files,
    replay_reference_files,
    save_file_replay_report,
)
from src.inference import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "match_reference_color.py"
SCHEMAS = ROOT / "configs" / "schemas"


def _image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).integers(
        16,
        240,
        size=(31, 37, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def _fit_recipe(
    tmp_path: Path,
    *,
    seed: int = 27801,
) -> tuple[Path, Path]:
    reference = tmp_path / "reference.png"
    source = tmp_path / "fit-source.png"
    recipe = tmp_path / "look.json"
    _image(reference, seed)
    _image(source, seed + 1)
    match_reference_files(
        reference,
        [source],
        [tmp_path / "fit-output.png"],
        recipe_path=recipe,
        guard_policy=ReferenceRenderGuardPolicy(
            allow_research_baseline=True,
        ),
    )
    return reference, recipe


def test_file_replay_needs_no_reference_and_reproduces_output_bytes(
    tmp_path: Path,
) -> None:
    reference, recipe_path = _fit_recipe(tmp_path)
    source = tmp_path / "new-source.png"
    first_output = tmp_path / "first.png"
    second_output = tmp_path / "second.png"
    _image(source, 27803)
    policy = ReferenceRenderGuardPolicy(allow_research_baseline=True)
    first = replay_reference_files(
        recipe_path,
        [source],
        [first_output],
        guard_policy=policy,
    )
    reference.unlink()
    second = replay_reference_files(
        recipe_path,
        [source],
        [second_output],
        guard_policy=policy,
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first.recipe == second.recipe
    assert first.recipe_file_sha256 == second.recipe_file_sha256
    assert first.outputs[0].safety.accepted is True


def test_bound_loader_hashes_the_exact_loaded_recipe_bytes(
    tmp_path: Path,
) -> None:
    _reference, recipe_path = _fit_recipe(tmp_path)
    recipe, digest = load_reference_look_recipe_bound(recipe_path)
    assert digest == sha256_file(recipe_path)
    assert recipe.recipe_id == json.loads(
        recipe_path.read_text(encoding="utf-8")
    )["recipe_id"]


def test_replay_report_binds_recipe_sources_outputs_and_schema(
    tmp_path: Path,
) -> None:
    _reference, recipe_path = _fit_recipe(tmp_path)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    report_path = tmp_path / "replay-report.json"
    _image(source, 27804)
    result = replay_reference_files(recipe_path, [source], [output])
    payload = build_file_replay_report(result)
    assert payload["schema_id"] == REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID
    assert payload["operation"] == "recipe-replay"
    assert payload["recipe_file"]["sha256"] == sha256_file(recipe_path)
    assert payload["outputs"][0]["source_sha256"] == sha256_file(source)
    assert payload["outputs"][0]["output_sha256"] == sha256_file(output)
    assert payload["outputs"][0]["safety"]["action"] == "identity-fallback"

    replay_schema = json.loads(
        (
            SCHEMAS / "reference_match_replay_report_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    match_schema = json.loads(
        (SCHEMAS / "reference_match_report_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(replay_schema)
    registry = Registry().with_resource(
        match_schema["$id"],
        Resource.from_contents(match_schema),
    )
    Draft202012Validator(
        replay_schema,
        registry=registry,
    ).validate(payload)
    first_hash = save_file_replay_report(result, report_path)
    first_bytes = report_path.read_bytes()
    second_hash = save_file_replay_report(result, report_path)
    assert report_path.read_bytes() == first_bytes
    assert first_hash == second_hash


def test_replay_cleans_staging_and_commits_nothing_on_late_failure(
    tmp_path: Path,
) -> None:
    _reference, recipe_path = _fit_recipe(tmp_path)
    good = tmp_path / "good.png"
    invalid = tmp_path / "invalid.png"
    first_output = tmp_path / "first.png"
    second_output = tmp_path / "second.png"
    _image(good, 27805)
    invalid.write_bytes(b"not an image")
    with pytest.raises((OSError, ValueError)):
        replay_reference_files(
            recipe_path,
            [good, invalid],
            [first_output, second_output],
        )
    assert not first_output.exists()
    assert not second_output.exists()
    assert not list(tmp_path.glob("*.reference-match-stage*"))


@pytest.mark.parametrize("protected", ["recipe", "source"])
def test_replay_never_overwrites_recipe_or_source(
    tmp_path: Path,
    protected: str,
) -> None:
    _reference, recipe_path = _fit_recipe(tmp_path)
    source = tmp_path / "source.png"
    _image(source, 27806)
    output = recipe_path if protected == "recipe" else source
    before = output.read_bytes()
    with pytest.raises(ReferenceMatchContractError, match="overwrite"):
        replay_reference_files(recipe_path, [source], [output])
    assert output.read_bytes() == before


def test_tampered_recipe_fails_before_any_output(tmp_path: Path) -> None:
    _reference, recipe_path = _fit_recipe(tmp_path)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _image(source, 27807)
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    payload["destination_lab_mean"][0] += 1.0
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ReferenceMatchContractError,
        match="recipe_id does not match",
    ):
        replay_reference_files(recipe_path, [source], [output])
    assert not output.exists()


def test_cli_replays_one_recipe_to_n_sources_without_reference(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    fit_source = tmp_path / "fit-source.png"
    fit_output = tmp_path / "fit-output.png"
    recipe = tmp_path / "look.json"
    fit_report = tmp_path / "fit-report.json"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_output = tmp_path / "first-output.png"
    second_output = tmp_path / "second-output.png"
    replay_report = tmp_path / "replay-report.json"
    _image(reference, 27808)
    _image(fit_source, 27809)
    _image(first, 27810)
    _image(second, 27811)
    fit = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--source",
            str(fit_source),
            "--output",
            str(fit_output),
            "--recipe",
            str(recipe),
            "--report",
            str(fit_report),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert fit.returncode == 0, fit.stderr
    reference.unlink()
    replay = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--recipe-input",
            str(recipe),
            "--source",
            str(first),
            "--source",
            str(second),
            "--output",
            str(first_output),
            "--output",
            str(second_output),
            "--report",
            str(replay_report),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stderr
    summary = json.loads(replay.stdout)
    report = json.loads(replay_report.read_text(encoding="utf-8"))
    assert summary["operation"] == "recipe-replay"
    assert summary["output_count"] == 2
    assert summary["recipe_id"] == json.loads(fit.stdout)["recipe_id"]
    assert report["reference_pixel_sha256"]
    assert "reference" not in report
    assert first_output.is_file()
    assert second_output.is_file()


def test_cli_rejects_conflicting_recipe_modes_before_outputs(
    tmp_path: Path,
) -> None:
    _reference, recipe = _fit_recipe(tmp_path)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    report = tmp_path / "report.json"
    _image(source, 27812)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--recipe-input",
            str(recipe),
            "--recipe",
            str(tmp_path / "other.json"),
            "--source",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "output-only" in completed.stderr
    assert not output.exists()
    assert not report.exists()
