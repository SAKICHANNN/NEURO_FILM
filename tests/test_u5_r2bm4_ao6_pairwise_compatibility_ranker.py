from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_pairwise_compatibility_ranker import (
    AO6PairwiseCompatibilityRankerError,
    _fit_ridge,
    _pair_feature,
    _predict,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bm4_ao6_pairwise_compatibility_ranker_v1.json"


def test_bm4_contract_binds_closed_fixed_retrieval_and_oracle() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["source_rows"]) == 17
    assert validated["bm1_report"]["automatic_pass"] is True


def test_pair_feature_has_frozen_dimension_and_no_final_rgb() -> None:
    semantic = np.eye(2)
    tone = np.stack((np.zeros(74), np.ones(74)))
    feature = _pair_feature(0, 1, semantic, tone)
    assert feature.shape == (75,)
    assert feature[0] == 0.0
    assert np.all(feature[1:] == 1.0)


def test_ridge_fit_is_deterministic() -> None:
    rng = np.random.default_rng(20260804)
    x = rng.normal(size=(40, 75))
    y = rng.normal(size=40)
    first = _fit_ridge(x, y, 1.0)
    second = _fit_ridge(x, y, 1.0)
    assert np.array_equal(_predict(*first, x), _predict(*second, x))


def test_bm4_rejects_final_rgb_or_operator_features() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ranker"]["final_rgb_prediction_allowed"] = True
    with pytest.raises(AO6PairwiseCompatibilityRankerError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["features"]["operator_signature_used_at_inference"] = True
    with pytest.raises(AO6PairwiseCompatibilityRankerError):
        validate_contract(ROOT, config)
