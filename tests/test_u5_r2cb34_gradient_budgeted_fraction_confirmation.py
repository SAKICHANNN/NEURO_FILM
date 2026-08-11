from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb34_gradient_budgeted_fraction_confirmation_v1.json"


def test_cb34_contract_freezes_exact_cb33_and_disjoint_population() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB34"
    assert contract["parents"]["cb33_required_decision"] == (
        "pass_development_gold_stress_open_source_disjoint_confirmation"
    )
    assert contract["population"]["source_count_exact"] == 12
    assert contract["population"]["exact_decoded_sha_overlap_with_cb27_cb28_cb29_u41"] == 0
    assert contract["operator"]["dose_grid"][-1] == 0.0
    assert contract["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"] == 1.35
