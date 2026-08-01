from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_normalization_support import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq0s2_fivek_casebank_normalization_v1.json"


def test_casebank_normalization_binds_exact_acquisition_and_prior_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["acquisition"]["rows"]) == 512
    assert config["parent_acquisition"]["required_assets"] == 1024
    assert config["support_and_leakage"]["existing_total_rows"] == 255
    assert config["operator_fitting_allowed"] is False


def test_casebank_normalization_preserves_both_targets_and_blind_split() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    split = config["split"]
    assert normalization["maximum_side"] == 256
    assert normalization["save_aligned_expert_target"] is True
    assert normalization["require_repository_data_junction"] is True
    assert split["target_confirmation_rows"] == 128
    assert split["target_pixels_or_fit_metrics_used"] is False
    assert split["minimum_development_rows"] == 320
