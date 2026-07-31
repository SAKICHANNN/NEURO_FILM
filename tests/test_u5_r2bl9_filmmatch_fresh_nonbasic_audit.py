from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest

from src.eval.filmmatch_fresh_nonbasic_audit import (
    FreshNonBasicAuditError,
    _sample_aligned,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bl9_filmmatch_fresh_nonbasic_audit_v1.json"


def test_bl9_aligned_sampling_is_deterministic_and_keeps_endpoints() -> None:
    source = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3) / float(16 * 16 * 3)
    output = source * np.float32(0.75)
    first = _sample_aligned(source, output, 256)
    second = _sample_aligned(source, output, 256)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0][[0, -1]], source.reshape(-1, 3)[[0, -1]])


def test_bl9_sampling_rejects_shape_mismatch() -> None:
    with pytest.raises(FreshNonBasicAuditError, match="shape mismatch"):
        _sample_aligned(np.zeros((16, 16, 3)), np.zeros((8, 8, 3)), 256)


def test_bl9_real_contract_freezes_basic_adversary_before_metrics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["eligible_ids"]) == 17
    assert config["automatic_gate"] == {
        "minimum_candidate_median_non_basic_delta_e76": 4.9,
        "minimum_candidate_median_non_basic_to_style_ratio": 0.35,
        "per_image_non_basic_delta_e76_floor": 3.0,
        "minimum_images_above_non_basic_floor": 12,
    }
    assert config["operator_fitting_allowed"] is False
