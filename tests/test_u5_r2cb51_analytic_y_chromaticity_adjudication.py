from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u5_r2cb51_analytic_y_chromaticity_confirmation_decision_v1.json"
)


def test_cb51_decision_retains_only_research_champion() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_and_output_hashes_exact"] is True
    assert decision["automatic_metrics"]["maximum_lstar_inversion_fraction"] == 0
    assert decision["autonomous_visual_metrics"]["candidate_round_choices"] == [
        14,
        14,
        14,
    ]
    assert decision["product_default_changed"] is False
    assert (
        decision["decision"]
        == "pass_cb51_source_disjoint_confirmation_retain_analytic_y_chromaticity_research_champion"
    )
