from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.nonexpansive_fraction_transport import (
    _nonexpansive_fraction_map,
    load_contract,
    nonexpansive_fraction_transport_target,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb32_nonexpansive_fraction_gold_stress_v1.json"


def test_cb32_contract_freezes_nonexpansive_mechanism() -> None:
    contract = load_contract(CONTRACT)
    assert contract["operator"]["fraction_knots"] == 257
    assert contract["operator"]["maximum_fraction_slope"] == 1.0
    assert contract["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"] == 1.35


def test_cb32_fraction_map_is_origin_anchored_and_nonexpansive() -> None:
    base = np.asarray([[0.1, 0.1, 0.2], [0.4, 0.8, 0.8]], dtype=np.float64)
    ao6 = np.asarray([[0.2, 0.5, 0.7], [0.9, 0.95, 0.95]], dtype=np.float64)
    valid = np.ones_like(base, dtype=bool)
    mapped, facts = _nonexpansive_fraction_map(
        base,
        ao6,
        valid,
        valid,
        minimum_valid_fraction=1e-4,
        fraction_knots=257,
        maximum_fraction_slope=1.0,
    )
    assert mapped[0, 0] == mapped[0, 1]
    assert mapped[1, 1] == mapped[1, 2]
    assert facts["maximum_observed_fraction_slope"] <= 1.0 + 1e-12
    ordered_x = np.sort(np.unique(base))
    ordered_y = np.asarray([mapped[base == value][0] for value in ordered_x])
    assert np.all(np.diff(ordered_y) >= 0.0)
    assert np.all(np.diff(ordered_y) <= np.diff(ordered_x) + 1e-12)


def test_cb32_target_is_exact_and_bounded() -> None:
    rng = np.random.default_rng(20260811 + 3200)
    base = rng.uniform(0.05, 0.95, size=(17, 19, 3)).astype(np.float32)
    ao6 = rng.uniform(0.05, 0.95, size=(17, 19, 3)).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    first, facts = nonexpansive_fraction_transport_target(
        base, ao6, weights=weights, return_diagnostics=True
    )
    second = nonexpansive_fraction_transport_target(
        base.copy(), ao6.copy(), weights=weights
    )
    assert np.array_equal(first, second)
    assert facts["maximum_observed_fraction_slope"] <= 1.0 + 1e-12
    assert np.isfinite(first).all()
    assert float(np.min(first)) >= -1e-7
    assert float(np.max(first)) <= 1.0 + 1e-7
