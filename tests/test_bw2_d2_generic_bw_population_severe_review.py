from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.generic_bw_population_severe_review import (
    GenericBwPopulationReviewError,
    build_population_review,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bw2_d2_generic_bw_population_severe_review_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_binds_population_and_keeps_claim_narrow() -> None:
    config = _config()
    for spec in config["parent_evidence"].values():
        assert hashlib.sha256((ROOT / spec["path"]).read_bytes()).hexdigest() == spec[
            "sha256"
        ]
    assert config["renderer"]["look_id"] == "generic_bw"
    assert config["renderer"]["look_amount"] == 1.0
    assert len(config["source"]["source_ids"]) == 16
    assert config["forbidden"]["named_hp5_or_tri_x_claim"] is True
    assert "not HP5 or Tri-X stock response" in config["claim_ceiling"]


def test_source_manifest_drift_fails_before_output(tmp_path: Path) -> None:
    config = _config()
    config["source"]["manifest_sha256"] = "0" * 64
    with pytest.raises(GenericBwPopulationReviewError, match="source manifest drift"):
        build_population_review(config, ROOT, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_directory_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        build_population_review(_config(), ROOT, output)


def test_direct_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_bw2_d2_generic_bw_population_severe_review.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
