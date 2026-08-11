from __future__ import annotations

import json
from pathlib import Path

from src.eval.analytic_y_chromaticity_gold_stress import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb52_analytic_y_chromaticity_gold_stress_v1.json"
DECISION = (
    ROOT / "configs/u5_r2cb52_analytic_y_chromaticity_gold_stress_decision_v1.json"
)


def test_cb52_contract_keeps_cb51_operator_and_safety_gates() -> None:
    current = load_contract(CONTRACT)
    previous = json.loads(
        (
            ROOT / "configs/u5_r2cb51_analytic_y_chromaticity_confirmation_v1.json"
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
    for key in (
        "maximum_luminance_reconstruction_error",
        "maximum_new_hard_boundary_fraction",
        "maximum_p999_gradient_ratio_vs_source",
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction",
    ):
        assert current["automatic_gates"][key] == previous["automatic_gates"][key]
    assert current["population"]["required_unavailable_ids"] == ["FS_FACE_01"]


def test_cb52_decision_retains_partial_research_claim() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_and_output_hashes_exact"] is True
    assert decision["visual_review"]["confirmed_severe_artifact_count"] == 0
    assert decision["automatic_metrics"]["unavailable_ids"] == ["FS_FACE_01"]
    assert decision["product_default_changed"] is False
    assert (
        decision["decision"]
        == "pass_cb52_partial_gold_stress_retain_analytic_y_chromaticity_research_champion"
    )
