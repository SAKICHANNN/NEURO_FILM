from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.preprocess import inspect_input, load_working_image

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "u1_5c_libultrahdr_reference_v1.json"


def _contract() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _fixture_rows() -> list[tuple[Path, dict]]:
    contract = _contract()
    directory = ROOT / contract["fixture_directory"]
    return [(directory / row["name"], row) for row in contract["fixtures"]]


def test_u1_5c_fixture_provenance_and_hashes_are_exact() -> None:
    contract = _contract()
    directory = ROOT / contract["fixture_directory"]
    source = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
    licence = (directory / "LICENSE-CC-BY-4.0.txt").read_text(encoding="utf-8")

    assert source["source_repository"] == contract["upstream"]["repository"]
    assert source["source_commit"] == contract["upstream"]["commit"]
    assert source["license"] == contract["upstream"]["data_license"]
    assert "Creative Commons Attribution 4.0 International" in licence
    assert source["files"] == [
        {"name": row["name"], "sha256": row["sha256"]}
        for row in contract["fixtures"]
    ]
    for path, row in _fixture_rows():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("fixture_path,row", _fixture_rows(), ids=lambda value: value.name if isinstance(value, Path) else None)
def test_real_gainmap_reference_rejects_before_working_pixels(
    fixture_path: Path, row: dict
) -> None:
    del row
    expected = _contract()["expected"]
    inspection = inspect_input(fixture_path)
    assert inspection.source_kind == expected["source_kind"]
    assert inspection.format_name == expected["format_name"]
    assert inspection.frame_count == expected["frame_count"]
    warning = next(
        warning
        for warning in inspection.warnings
        if warning.code == "unsupported_dynamic_range"
    )
    assert "urn:com:apple:photo:2020:aux:hdrgainmap" in warning.message
    with pytest.raises(ValueError, match="HDR/gain-map reconstruction is not implemented"):
        load_working_image(fixture_path)


@pytest.mark.parametrize("fixture_path,row", _fixture_rows(), ids=lambda value: value.name if isinstance(value, Path) else None)
def test_renderer_rejects_real_gainmap_reference_without_output(
    tmp_path: Path, fixture_path: Path, row: dict
) -> None:
    del row
    output_path = tmp_path / f"{fixture_path.stem}.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_film.py"),
            str(fixture_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "HDR/gain-map reconstruction is not implemented" in completed.stderr
    assert not output_path.exists()
    assert not output_path.with_suffix(".metrics.json").exists()
