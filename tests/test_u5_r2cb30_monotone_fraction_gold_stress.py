from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.monotone_fraction_gold_stress import _inputs, load_contract
from src.eval.monotone_fraction_quantile_transport import (
    monotone_fraction_quantile_transport_target,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb30_monotone_fraction_gold_stress_v1.json"


def test_cb30_contract_freezes_partial_gold_scope() -> None:
    contract = load_contract(CONTRACT)
    assert contract["population"]["expected_available_source_count"] == 40
    assert contract["population"]["required_unavailable_ids"] == ["FS_FACE_01"]
    assert "not the complete gold set" in contract["claim_ceiling"]


def test_cb30_inputs_bind_exact_operator_and_population() -> None:
    contract = load_contract(CONTRACT)
    cb11, ao6, artifact, curve, available, unavailable = _inputs(contract, ROOT)
    assert cb11["experiment_id"] == "U5.R2CB11"
    assert ao6["arm_id"] == "fixed_ao6_colour_only_t15_c35"
    assert artifact["bundle_sha256"] == ao6["frozen_bundle_sha256"]
    assert callable(curve)
    assert len(available) == 40
    assert unavailable == ["FS_FACE_01"]


def test_cb30_target_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(20260811 + 3000)
    base = rng.uniform(0.05, 0.95, size=(13, 17, 3)).astype(np.float32)
    ao6 = rng.uniform(0.05, 0.95, size=(13, 17, 3)).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    first = monotone_fraction_quantile_transport_target(
        base, ao6, weights=weights, minimum_valid_fraction=1e-4
    )
    second = monotone_fraction_quantile_transport_target(
        base.copy(), ao6.copy(), weights=weights, minimum_valid_fraction=1e-4
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(np.min(first)) >= -1e-7
    assert float(np.max(first)) <= 1.0 + 1e-7
