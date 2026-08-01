from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_case_operator_oracle import (
    AO6CaseOperatorOracleError,
    _safe_strength_oracle,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bm1_ao6_case_operator_oracle_v1.json"


def test_bm1_contract_validates_exact_parents() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["source_rows"]) == 17
    assert set(validated["fold_map"].values()) == {0, 1, 2}
    assert validated["display_payload"]["residual"]["tone_strength"] == 0.15
    assert validated["display_payload"]["residual"]["chroma_strength"] == 0.35


def test_bm1_rejects_content_features_or_dense_blending() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["protocol"]["content_features_used"] = True
    with pytest.raises(AO6CaseOperatorOracleError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["protocol"]["dense_case_blending_allowed"] = True
    with pytest.raises(AO6CaseOperatorOracleError):
        validate_contract(ROOT, config)


def test_strength_oracle_is_analytic_and_never_clips() -> None:
    source = np.asarray([[[0.1, 0.7, 0.9], [0.8, 0.2, 0.3]]])
    medoid = np.asarray([[[0.4, 0.5, 0.7], [0.6, 0.5, 0.9]]])
    target = source + 1.25 * (medoid - source)
    output, strength = _safe_strength_oracle(source, medoid, target)
    assert 0.0 < strength <= 1.5
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert np.array_equal(output, source + strength * (medoid - source))
