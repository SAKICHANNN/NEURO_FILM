from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.tie_preserving_fraction_transport import (
    _grouped_midrank_quantile_transport,
    load_contract,
    tie_preserving_fraction_transport_target,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb31_tie_preserving_fraction_gold_stress_v1.json"


def test_cb31_contract_changes_only_rank_equivalence_mechanism() -> None:
    contract = load_contract(CONTRACT)
    assert contract["population"]["role"].startswith("development_only")
    assert contract["operator"]["minimum_valid_fraction"] == 0.0001
    assert contract["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"] == 1.35


def test_cb31_grouped_midrank_preserves_exact_ties() -> None:
    base = np.asarray([[0.2, 0.1, 0.2], [0.8, 0.1, 0.8]], dtype=np.float64)
    ao6 = np.asarray([[0.05, 0.3, 0.6], [0.9, 0.4, 0.7]], dtype=np.float64)
    valid = np.ones_like(base, dtype=bool)
    mapped = _grouped_midrank_quantile_transport(
        base, ao6, valid, valid, minimum_valid_fraction=1e-4
    )
    assert mapped[0, 0] == mapped[0, 2]
    assert mapped[0, 1] == mapped[1, 1]
    assert mapped[1, 0] == mapped[1, 2]
    assert np.all(np.diff(np.sort(mapped.reshape(-1))) >= 0.0)


def test_cb31_target_is_exact_and_bounded() -> None:
    rng = np.random.default_rng(20260811 + 3100)
    base = rng.uniform(0.05, 0.95, size=(17, 19, 3)).astype(np.float32)
    base[2:5] = base[1]
    ao6 = rng.uniform(0.05, 0.95, size=(17, 19, 3)).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    first = tie_preserving_fraction_transport_target(base, ao6, weights=weights)
    second = tie_preserving_fraction_transport_target(
        base.copy(), ao6.copy(), weights=weights
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(np.min(first)) >= -1e-7
    assert float(np.max(first)) <= 1.0 + 1e-7
