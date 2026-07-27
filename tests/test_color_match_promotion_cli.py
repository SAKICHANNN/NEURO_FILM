from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_reference_match_promotion.py"


def _write_fixture(
    directory: Path,
    sample_id: str,
    pixels: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.asarray(np.round(pixels * 255.0), dtype=np.uint8),
        mode="RGB",
    ).save(directory / f"{sample_id}_fixture.png")


def test_promotion_cli_streams_matrix_and_repeats_exactly(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "references"
    source_dir = tmp_path / "sources"
    target_dir = tmp_path / "targets"
    sample_ids = ("a", "b", "c")
    rng = np.random.default_rng(72027)
    for index, sample_id in enumerate(sample_ids):
        source = rng.uniform(0.12, 0.78, size=(9, 11, 3)).astype(np.float32)
        target = np.clip(
            source * np.asarray([0.88, 0.96, 1.06], dtype=np.float32)
            + 0.01 * index,
            0.0,
            1.0,
        )
        _write_fixture(source_dir, sample_id, source)
        _write_fixture(target_dir, sample_id, target)
        _write_fixture(reference_dir, sample_id, target)

    output = tmp_path / "promotion.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--reference-dir",
        str(reference_dir),
        "--source-dir",
        str(source_dir),
        "--target-dir",
        str(target_dir),
        "--sample-ids",
        *sample_ids,
        "--output",
        str(output),
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    first_bytes = output.read_bytes()
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            ROOT
            / "configs"
            / "schemas"
            / "reference_match_promotion_report_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert first.stdout == second.stdout
    assert output.read_bytes() == first_bytes
    assert payload["schema_id"] == (
        "neuro-film.reference-match-promotion-report.v1"
    )
    assert len(payload["known_operator"]["samples"]) == 6
    assert len(payload["photographic_safety"]["samples"]) == 3
    assert payload["visual_review"] is None
    assert payload["promotion_decision"]["status"] != "promoted"

    relocated = tmp_path / "relocated"
    relocated_reference = relocated / "references"
    relocated_source = relocated / "sources"
    relocated_target = relocated / "targets"
    shutil.copytree(reference_dir, relocated_reference)
    shutil.copytree(source_dir, relocated_source)
    shutil.copytree(target_dir, relocated_target)
    relocated_output = relocated / "promotion.json"
    relocated_command = [
        sys.executable,
        str(SCRIPT),
        "--reference-dir",
        str(relocated_reference),
        "--source-dir",
        str(relocated_source),
        "--target-dir",
        str(relocated_target),
        "--sample-ids",
        *sample_ids,
        "--output",
        str(relocated_output),
    ]
    subprocess.run(
        relocated_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relocated_payload = json.loads(
        relocated_output.read_text(encoding="utf-8")
    )
    assert relocated_payload["report_id"] == payload["report_id"]
    assert relocated_payload["reference_paths"] != payload["reference_paths"]
