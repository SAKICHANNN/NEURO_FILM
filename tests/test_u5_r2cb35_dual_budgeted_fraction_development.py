from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.dual_budgeted_fraction_transport import (
    DualBudgetedFractionTransportError,
    select_dual_budgeted_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb35_dual_budgeted_fraction_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb35_dual_budgeted_fraction_development_decision_v1.json"


def test_cb35_contract_freezes_dual_global_budget_on_disjoint_sources() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB35"
    assert contract["population"]["source_count_exact"] == 12
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb29_cb34_u41"] == 0
    assert contract["operator"]["dose_grid"] == [1.0 - index / 16.0 for index in range(17)]
    assert contract["automatic_gates"]["maximum_adjacent_lstar_gradient_sign_inversion_fraction"] == 0.0


def test_cb35_selector_accepts_identity_at_highest_dose() -> None:
    source = np.full((17, 19, 3), 0.25, dtype=np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    candidate, scale, luma_error, facts = select_dual_budgeted_candidate(
        source,
        source.copy(),
        source.copy(),
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=1.35,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
    )
    assert np.array_equal(candidate, source)
    assert np.all(scale == 1.0)
    assert np.max(np.abs(luma_error)) == 0.0
    assert facts["global_dose"] == 1.0
    assert facts["selected_lstar_inversion_fraction"] == 0.0


def test_cb35_selector_fails_when_base_already_reverses_order() -> None:
    source = np.full((8, 8, 3), 0.2, dtype=np.float32)
    source[:, 4:] = 0.4
    base = np.full_like(source, 0.4)
    base[:, 4:] = 0.2
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    with pytest.raises(
        DualBudgetedFractionTransportError,
        match=r"no dual-safe dose; .*dose0_inversion=",
    ):
        select_dual_budgeted_candidate(
            source,
            base,
            base.copy(),
            weights=weights,
            boundary_epsilon=1.0 / 65535.0,
            dose_grid=[1.0, 0.0],
            maximum_gradient_ratio=10.0,
            maximum_lstar_inversion_fraction=0.0,
            lstar_order_epsilon=0.0001,
        )


def test_cb35_decision_records_base_order_failure() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["report_sha256"] == (
        "e587be3b062e7dbd6f1262c1638d5a91a7c83cd1a297070435d07304bf3d4e66"
    )
    assert decision["failure"]["dose0_lstar_inversion_fraction"] > 0.0
    assert decision["partial_artifacts_removed"] is True
    assert decision["decision"] == (
        "close_global_dual_budget_because_base_itself_breaks_exact_lstar_order"
    )
