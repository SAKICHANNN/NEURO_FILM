from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_projection_curve_basis_development import (
    FiveKProjectionCurveError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bn1_fivek_projection_curve_basis_development_v1.json"
)


def test_contract_binds_source_method_parents_and_populations() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["directions"].shape == (16, 3)
    assert [row["rows"] for row in validated["populations"]] == [64, 63, 64]


def test_contract_rejects_output_clipping_and_semantic_features() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["hard_output_clipping_allowed"] = True
    with pytest.raises(FiveKProjectionCurveError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["spatial_or_semantic_features_allowed"] = True
    with pytest.raises(FiveKProjectionCurveError):
        validate_contract(ROOT, config)


def test_contract_rejects_unlicensed_code_mapping_or_learned_final_rgb() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["basis_prediction"]["learned_final_rgb_allowed"] = True
    with pytest.raises(FiveKProjectionCurveError):
        validate_contract(ROOT, config)
    source_path = ROOT / config["source_method"]["decision"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["code_rights"]["code_or_checkpoint_copy_allowed"] is False
