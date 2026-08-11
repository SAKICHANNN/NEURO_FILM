from __future__ import annotations

import json
from pathlib import Path

from src.eval.analytic_y_chromaticity_confirmation import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb51_analytic_y_chromaticity_confirmation_v1.json"


def test_cb51_contract_keeps_cb50_operator_and_gates_exact() -> None:
    current = load_contract(CONTRACT)
    previous = json.loads(
        (
            ROOT / "configs/u5_r2cb50_analytic_y_chromaticity_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "dose_grid",
        "lstar_order_epsilon",
        "minimum_valid_fraction",
        "fraction_knots",
        "maximum_fraction_slope",
    ):
        assert current["operator"][key] == previous["operator"][key]
    assert current["automatic_gates"] == previous["automatic_gates"]
    assert current["population"]["source_count_exact"] == 17
    assert (
        current["parents"]["cb50_required_decision"]
        == "pass_cb50_development_open_source_disjoint_confirmation"
    )
