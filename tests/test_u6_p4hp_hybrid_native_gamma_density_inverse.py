from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4hp_contract_preserves_grid_and_changes_numerical_mechanism() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4hp_hybrid_native_gamma_density_inverse_v1.json"
        ).read_text("utf-8")
    )
    assert contract["parents"]["p4ho_evidence"]["required_decision"] == (
        "retain_python_bounded_consumer_without_native_density_inverse"
    )
    assert contract["candidate"]["high_shape_threshold"] == 10000.0
    assert 1000.0 < contract["candidate"]["high_shape_threshold"] < 100000.0
    assert (
        contract["automatic_gates"]["maximum_absolute_developed_density_error"] == 2e-8
    )
