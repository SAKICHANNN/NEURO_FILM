from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src.color_match import (
    REFERENCE_MATCH_REPORT_SCHEMA_ID,
    ReferenceMatchContractError,
    build_file_match_report,
    match_reference_files,
    save_file_match_report,
)
from src.inference import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "match_reference_color.py"


def _image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).integers(
        16,
        240,
        size=(29, 37, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)


def test_report_covers_run_hashes_diagnostics_and_safety(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report_path = tmp_path / "report.json"
    _image(reference, 27401)
    _image(source, 27402)
    result = match_reference_files(
        reference,
        [source],
        [output],
        recipe_path=recipe,
    )
    report = build_file_match_report(result)

    assert report["schema_id"] == REFERENCE_MATCH_REPORT_SCHEMA_ID
    assert report["reference"]["file_sha256"] == sha256_file(reference)
    assert report["recipe_file"]["sha256"] == sha256_file(recipe)
    assert report["outputs"][0]["source_sha256"] == sha256_file(source)
    assert report["outputs"][0]["output_sha256"] == sha256_file(output)
    assert report["outputs"][0]["safety"]["policy_id"] == (
        "reference-render-guard.v2"
    )
    first_hash = save_file_match_report(result, report_path)
    first_bytes = report_path.read_bytes()
    second_hash = save_file_match_report(result, report_path)
    assert report_path.read_bytes() == first_bytes
    assert first_hash == second_hash


@pytest.mark.parametrize("protected", ["reference", "source", "output", "recipe"])
def test_report_never_overwrites_run_artifacts(
    tmp_path: Path,
    protected: str,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    _image(reference, 27403)
    _image(source, 27404)
    result = match_reference_files(
        reference,
        [source],
        [output],
        recipe_path=recipe,
    )
    target = {
        "reference": reference,
        "source": source,
        "output": output,
        "recipe": recipe,
    }[protected]
    with pytest.raises(ReferenceMatchContractError, match="overwrite"):
        save_file_match_report(result, target)


def test_cli_runs_one_reference_n_sources_and_writes_report(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_output = tmp_path / "first-out.png"
    second_output = tmp_path / "second-out.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27405)
    _image(first, 27406)
    _image(second, 27407)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--source",
            str(first),
            "--source",
            str(second),
            "--output",
            str(first_output),
            "--output",
            str(second_output),
            "--recipe",
            str(recipe),
            "--report",
            str(report),
            "--bit-depth",
            "16",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert summary["output_count"] == 2
    assert (
        summary["applied_count"] + summary["identity_fallback_count"] == 2
    )
    assert summary["applied_count"] == 0
    assert summary["identity_fallback_count"] == 2
    assert payload["recipe_id"] == summary["recipe_id"]
    assert first_output.is_file()
    assert second_output.is_file()
    assert recipe.is_file()


def test_cli_fails_without_partial_outputs_on_batch_mismatch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27408)
    _image(source, 27409)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--source",
            str(source),
            "--output",
            str(output),
            "--recipe",
            str(recipe),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "equal length" in completed.stderr
    assert not output.exists()
    assert not recipe.exists()
    assert not report.exists()


def test_cli_research_override_is_explicit_and_reported(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _image(reference, 27410)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--source",
            str(reference),
            "--output",
            str(output),
            "--recipe",
            str(recipe),
            "--report",
            str(report),
            "--allow-research-baseline",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert summary["applied_count"] == 1
    assert summary["identity_fallback_count"] == 0
    assert payload["outputs"][0]["safety"][
        "research_baseline_override"
    ] is True
