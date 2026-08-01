from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_projection_curve_visual_product_value import (
    ARMS,
    FiveKProjectionCurveVisualError,
    _p999_gradient,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bn3_projection_curve_visual_product_value_v1.json"


def test_bn3_contract_binds_independent_population_and_fixed_arms() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert tuple(config["render_arms"]) == ARMS
    assert len(validated["eligible_ids"]) == 12
    assert len({validated["source_rows"][key]["make"] for key in validated["eligible_ids"]}) == 12


def test_bn3_contract_rejects_target_access_and_arm_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_model"]["target_pixels_available"] = True
    with pytest.raises(FiveKProjectionCurveVisualError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["render_arms"] = list(reversed(config["render_arms"]))
    with pytest.raises(FiveKProjectionCurveVisualError):
        validate_contract(ROOT, config)


def test_gradient_tail_measure_is_finite_and_ordered() -> None:
    smooth = np.zeros((32, 32, 3), dtype=np.float64)
    smooth[:, :, 0] = np.linspace(0.0, 1.0, 32)[None, :]
    impulse = smooth.copy()
    impulse[16, 16] = 1.0
    assert np.isfinite(_p999_gradient(smooth))
    assert _p999_gradient(impulse) > _p999_gradient(smooth)
