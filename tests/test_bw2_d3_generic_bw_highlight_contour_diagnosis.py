from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.generic_bw_highlight_contour_diagnosis import (
    GenericBwContourDiagnosisError,
    run_diagnosis,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bw2_d3_generic_bw_highlight_contour_diagnosis_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_binds_failure_and_single_variable_variants() -> None:
    config = _config()
    parent = config["parent_evidence"]
    assert hashlib.sha256((ROOT / parent["path"]).read_bytes()).hexdigest() == parent[
        "sha256"
    ]
    assert [row["variant_id"] for row in config["ordered_single_variable_variants"]] == [
        "current",
        "dither_zero",
        "preserve_luma_detail_zero",
        "tone_rolloff_zero",
        "output_margin_zero",
    ]
    assert all(
        len(row["overrides"]) <= 1
        for row in config["ordered_single_variable_variants"]
    )
    assert config["forbidden"]["product_or_population_promotion"] is True


def test_parent_drift_fails_before_output(tmp_path: Path) -> None:
    config = _config()
    config["parent_evidence"]["sha256"] = "0" * 64
    with pytest.raises(GenericBwContourDiagnosisError, match="bound artifact drift"):
        run_diagnosis(config, ROOT, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_directory_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        run_diagnosis(_config(), ROOT, output)


def test_direct_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_bw2_d3_generic_bw_highlight_contour_diagnosis.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
