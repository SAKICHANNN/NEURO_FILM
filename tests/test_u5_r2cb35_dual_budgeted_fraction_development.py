from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb35_dual_budgeted_fraction_development_v1.json"


def test_cb35_contract_freezes_dual_global_budget_on_disjoint_sources() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB35"
    assert contract["population"]["source_count_exact"] == 12
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb29_cb34_u41"] == 0
    assert contract["operator"]["dose_grid"] == [1.0 - index / 16.0 for index in range(17)]
    assert contract["automatic_gates"]["maximum_adjacent_lstar_gradient_sign_inversion_fraction"] == 0.0
