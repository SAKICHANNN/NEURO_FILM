from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.direct_lstar_monotone_tone_transport import (
    _lstar_to_neutral_linear,
    monotone_lstar_quantile_map,
    select_direct_lstar_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb39_direct_lstar_monotone_tone_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb39_direct_lstar_monotone_tone_development_decision_v1.json"


def test_cb39_contract_freezes_direct_lstar_and_source_subset() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB39"
    assert contract["population"]["source_count_exact"] == 16
    assert len(contract["population"]["included_source_ids"]) == 16
    assert contract["operator"]["tone_coordinate"] == "actual frozen float32 CIELAB L-star"


def test_cb39_lstar_map_is_monotone_with_duplicate_source_values() -> None:
    source = np.asarray([[10.0, 10.0, 20.0], [40.0, 70.0, 90.0]])
    target = np.asarray([[3.0, 5.0, 12.0], [55.0, 72.0, 96.0]])
    mapped, facts = monotone_lstar_quantile_map(source, target, knot_count=17)
    order = np.argsort(source.reshape(-1), kind="stable")
    assert np.all(np.diff(mapped.reshape(-1)[order]) >= 0.0)
    assert facts["tone_knot_count"] >= 3


def test_cb39_lstar_inverse_roundtrip_is_tight() -> None:
    lstar = np.linspace(0.0, 100.0, 1001)
    y = _lstar_to_neutral_linear(lstar)
    assert np.min(y) >= 0.0
    assert np.max(y) <= 1.0 + 1e-12
    assert np.all(np.diff(y) >= 0.0)


def test_cb39_selector_has_zero_direct_lstar_inversion() -> None:
    x = np.linspace(0.03, 0.91, 53, dtype=np.float32)
    source = np.repeat(x[None, :, None], 3, axis=2)
    base = np.asarray(np.sqrt(source), dtype=np.float32)
    target = base.copy()
    target[..., 0] += np.float32(0.05)
    target[..., 2] -= np.float32(0.03)
    _, _, luma_error, facts = select_direct_lstar_candidate(
        source,
        base,
        target,
        boundary_epsilon=1.0 / 65535.0,
        tone_knot_count=257,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=10.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert float(np.max(np.abs(luma_error))) < 1e-6
    assert facts["selected_lstar_inversion_fraction"] == 0.0


def test_cb39_decision_opens_positive_identity_slope() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["failure"]["minimum_gradient_ratio"] < 1.35
    assert decision["failure"]["minimum_lstar_inversion_fraction"] > 0.0
    assert decision["decision"] == (
        "close_nonstrict_lstar_quantile_open_identity_slope_successor"
    )
