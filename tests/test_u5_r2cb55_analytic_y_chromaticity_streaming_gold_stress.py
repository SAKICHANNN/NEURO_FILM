from __future__ import annotations

import json
from pathlib import Path

from src.eval.analytic_y_chromaticity_gold_stress import load_contract as load_cb52
from src.eval.analytic_y_chromaticity_streaming_gold_stress import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb55_analytic_y_chromaticity_streaming_gold_stress_v1.json"


def test_cb55_contract_keeps_cb52_population_and_gate_source() -> None:
    current = load_contract(CONTRACT)
    previous = load_cb52(ROOT / current["parents"]["cb52_contract_path"])
    assert current["execution"]["required_source_count"] == previous["population"]["expected_available_source_count"]
    assert current["execution"]["required_output_hash_match_count"] == 40
    assert current["execution"]["required_row_fact_match_count"] == 40


def test_cb55_parent_decision_is_streaming_pass() -> None:
    current = load_contract(CONTRACT)
    decision = json.loads(
        (ROOT / current["parents"]["cb54_decision_path"]).read_text(encoding="utf-8")
    )
    assert decision["decision"] == current["parents"]["cb54_required_decision"]
    assert decision["product_default_changed"] is False
