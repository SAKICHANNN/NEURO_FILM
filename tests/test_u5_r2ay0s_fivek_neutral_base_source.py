from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.fivek_neutral_base_source import (
    FiveKNeutralBaseSourceError,
    build_source_evidence,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/u5_r2ay0_fivek_neutral_base_source_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_ay0s_contract_keeps_neutral_base_and_film_look_separate() -> None:
    config = _config()
    validate_contract(ROOT, config)
    assert config["source"]["expected_pair_count"] == 64
    assert config["grouping"]["primary_group"] == (
        "normalized EXIF make + model"
    )
    assert any("AO6" in row for row in config["forbidden"])
    assert any("final RGB" in row for row in config["forbidden"])


def test_ay0s_contract_rejects_hash_and_grouping_drift() -> None:
    config = _config()
    config["source"]["freeze_manifest_sha256"] = "0" * 64
    with pytest.raises(FiveKNeutralBaseSourceError, match="hash drift"):
        validate_contract(ROOT, config)
    config = _config()
    config["grouping"]["primary_group"] = "image"
    with pytest.raises(FiveKNeutralBaseSourceError, match="grouping"):
        validate_contract(ROOT, config)


def test_ay0s_full_source_manifest_is_repeat_exact(tmp_path: Path) -> None:
    config = _config()
    first = build_source_evidence(
        root=ROOT,
        config=copy.deepcopy(config),
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "a",
        software_commit="test",
    )
    second = build_source_evidence(
        root=ROOT,
        config=copy.deepcopy(config),
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "b",
        software_commit="test",
    )
    assert first["report"]["automatic_pass"] is True
    assert first["report"]["pair_count"] == 64
    assert first["report"]["camera_model_group_count"] == 21
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["report_sha256"] == second["report_sha256"]
