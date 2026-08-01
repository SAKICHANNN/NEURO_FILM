from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_projection_curve_confirmation import (
    FiveKProjectionCurveConfirmationError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bn2_fivek_projection_curve_confirmation_v1.json"


def test_confirmation_contract_is_disjoint_and_threshold_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["pair_overlap"] == []
    assert validated["hash_overlap"] == []
    assert len(validated["confirmation_manifest"]["rows"]) == 64


def test_confirmation_rejects_target_access_and_threshold_relaxation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["leakage_controls"][
        "target_pixels_allowed_for_adaptive_prediction"
    ] = True
    with pytest.raises(FiveKProjectionCurveConfirmationError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["evaluation"]["maximum_adaptive_p95_ratio_to_global"] = 1.01
    with pytest.raises(FiveKProjectionCurveConfirmationError):
        validate_contract(ROOT, config)
