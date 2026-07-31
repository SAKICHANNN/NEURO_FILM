from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.smooth_perceptual_hue_density_regression import (
    SmoothPerceptualRegressionError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk7_smooth_perceptual_hue_density_v1.json"


def test_bk7_regression_binds_frozen_phaseone_failure() -> None:
    validated = validate_contract(
        ROOT, json.loads(CONFIG.read_text(encoding="utf-8"))
    )
    assert validated["source_path"].name == "phaseone_p25plus.png"
    assert validated["source_sha256"] == (
        "a81bd65ce6b88f50d26862ae6f38a03109a7ca02d782eb87ffc3368d657afe9a"
    )


def test_bk7_regression_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["strength"] = 0.75
    with pytest.raises(SmoothPerceptualRegressionError):
        validate_contract(ROOT, config)
