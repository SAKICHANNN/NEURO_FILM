from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_content_hard_retrieval import (
    AO6ContentHardRetrievalError,
    _fold_selector,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bm2_ao6_content_hard_retrieval_v1.json"


def test_bm2_contract_binds_case_oracle_and_model() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["bm1_report"]["automatic_pass"] is True
    assert len(validated["source_rows"]) == 17


def test_bm2_rejects_colour_or_operator_inference_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["selector"]["source_colour_statistics_used"] = True
    with pytest.raises(AO6ContentHardRetrievalError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["selector"]["operator_signatures_used_at_inference"] = True
    with pytest.raises(AO6ContentHardRetrievalError):
        validate_contract(ROOT, config)


def test_fold_selector_uses_development_only_threshold_and_hard_top1() -> None:
    source_ids = ["a", "b", "c", "d", "e"]
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.7, 0.7],
        ],
        dtype=np.float64,
    )
    threshold, rows = _fold_selector(
        normalized_features=features / np.linalg.norm(features, axis=1)[:, None],
        source_ids=source_ids,
        development_ids=["a", "b", "c"],
        held_ids=["d", "e"],
    )
    assert np.isfinite(threshold)
    assert rows["d"]["nearest_source_id"] == "c"
    assert rows["e"]["nearest_source_id"] in {"a", "b", "c"}
    assert all(row["nearest_source_id"] not in {"d", "e"} for row in rows.values())
