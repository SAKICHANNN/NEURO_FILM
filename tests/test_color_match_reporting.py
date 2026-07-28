from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from PIL import Image

from src.color_match import (
    MAX_REFERENCE_MATCH_BATCH_SOURCES,
    REFERENCE_MATCH_REPORT_SCHEMA_ID,
    ReferenceMatchContractError,
    build_file_match_report,
    match_reference_files,
    save_file_match_report,
)
from src.inference import sha256_file
from src.preprocess import (
    SourceProfile,
    WorkingImage,
    inspect_input,
    save_rec2020_16_png,
)


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


def _rec2020_image(path: Path, seed: int) -> None:
    pixels = np.random.default_rng(seed).uniform(
        0.02,
        0.98,
        size=(29, 37, 3),
    ).astype(np.float32)
    save_rec2020_16_png(
        WorkingImage(
            pixels=pixels,
            working_space="linear_rec2020",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("cicp", "BT.2020 SDR CLI test"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=16,
            source_path=path,
            warnings=[],
        ),
        path,
    )


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


def test_report_retains_the_exact_decoded_input_file_identities(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _image(reference, 27431)
    _image(source, 27432)
    reference_sha256 = sha256_file(reference)
    source_sha256 = sha256_file(source)

    result = match_reference_files(reference, [source], [output])
    _image(reference, 27433)
    _image(source, 27434)
    report = build_file_match_report(result)

    assert result.reference_file_sha256 == reference_sha256
    assert result.outputs[0].source_file_sha256 == source_sha256
    assert report["reference"]["file_sha256"] == reference_sha256
    assert report["outputs"][0]["source_sha256"] == source_sha256


def test_report_rejects_forged_oversized_result(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _image(reference, 27421)
    _image(source, 27422)
    result = match_reference_files(reference, [source], [output])
    forged = replace(
        result,
        outputs=result.outputs * (
            MAX_REFERENCE_MATCH_BATCH_SOURCES + 1
        ),
    )
    with pytest.raises(
        ReferenceMatchContractError,
        match="between 1 and 64 rows",
    ):
        build_file_match_report(forged)


@pytest.mark.parametrize(
    "field",
    ["reference", "source", "output", "recipe"],
)
def test_report_rejects_forged_artifact_identity(
    tmp_path: Path,
    field: str,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _image(reference, 27435)
    _image(source, 27436)
    result = match_reference_files(reference, [source], [output])
    if field == "reference":
        forged = replace(result, reference_file_sha256="not-a-hash")
    elif field == "source":
        forged = replace(
            result,
            outputs=(
                replace(
                    result.outputs[0],
                    source_file_sha256="not-a-hash",
                ),
            ),
        )
    elif field == "output":
        forged = replace(
            result,
            outputs=(
                replace(
                    result.outputs[0],
                    output_sha256="not-a-hash",
                ),
            ),
        )
    else:
        forged = replace(
            result,
            recipe_path=tmp_path / "recipe.json",
            recipe_file_sha256="not-a-hash",
        )
    with pytest.raises(
        ReferenceMatchContractError,
        match={
            "reference": "reference_file_sha256",
            "source": "source_file_sha256",
            "output": "output_sha256",
            "recipe": "recipe_file_sha256",
        }[field]
        + " must be a lowercase SHA-256",
    ):
        build_file_match_report(forged)


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


def test_report_save_rejects_concurrent_same_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.color_match import reporting

    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    report_path = tmp_path / "report.json"
    _image(reference, 27437)
    _image(source, 27438)
    result = match_reference_files(reference, [source], [output])
    entered = threading.Event()
    release = threading.Event()
    original_write = reporting.atomic_write_json
    first_errors: list[BaseException] = []

    def blocked_write(path, payload):
        entered.set()
        assert release.wait(timeout=10)
        return original_write(path, payload)

    monkeypatch.setattr(reporting, "atomic_write_json", blocked_write)

    def first_writer() -> None:
        try:
            save_file_match_report(result, report_path)
        except BaseException as exc:  # pragma: no cover - diagnostic capture.
            first_errors.append(exc)

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert entered.wait(timeout=10)
    try:
        with pytest.raises(
            ReferenceMatchContractError,
            match="already locked",
        ):
            save_file_match_report(result, report_path)
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not first_errors
    assert report_path.is_file()


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


def test_cli_help_documents_the_rec2020_sdr_output_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Rec.2020 SDR" in completed.stdout
    assert "16-bit PNG" in completed.stdout


def test_cli_commits_rec2020_sdr_output_recipe_and_report(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    recipe = tmp_path / "recipe.json"
    report = tmp_path / "report.json"
    _rec2020_image(reference, 27439)
    _rec2020_image(source, 27440)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--output",
            str(output),
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
    assert summary["output_count"] == 1
    assert summary["identity_fallback_count"] == 1
    assert payload["outputs"][0]["candidate_diagnostics"][
        "source_working_space"
    ] == "linear_rec2020"
    assert inspect_input(output).source_profile.kind == "cicp"
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
