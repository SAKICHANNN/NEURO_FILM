from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.three_stock_population_separation import (
    ThreeStockPopulationSeparationError,
    delta_e76_summary,
    run_population_separation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rf3_d14_three_stock_population_separation_v1.json"


def test_delta_e76_summary_is_zero_for_identity_and_rejects_geometry() -> None:
    image = np.zeros((7, 9, 3), dtype=np.uint8)
    assert delta_e76_summary(image, image) == {
        "median_delta_e76": 0.0,
        "p95_delta_e76": 0.0,
        "maximum_delta_e76": 0.0,
    }
    with pytest.raises(ThreeStockPopulationSeparationError, match="geometry"):
        delta_e76_summary(image, image[:, :-1])


def test_rf3_d14_real_population_is_exact_and_ao6_free() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forward = run_population_separation(config, ROOT)
    reverse = run_population_separation(config, ROOT, reverse=True)

    assert forward == reverse
    payload = forward["scientific_payload"]
    assert payload["source_count"] == 16
    assert len(payload["pair_source_rows"]) == 48
    assert len(payload["pair_aggregates"]) == 3
    assert payload["gates"]["exact_16_source_three_arm_population"]
    assert payload["gates"]["u4_3d_severe_review_pass"]
    assert payload["gates"]["parent_threshold_inherited"]
    assert payload["gates"]["ao6_excluded"]
    assert all("ao6" not in arm_id for arm_id in payload["arm_ids"])
    assert payload["render_calls"] == payload["network_reads"] == 0
    assert "not blind human distinguishability" in payload["claim_ceiling"]
