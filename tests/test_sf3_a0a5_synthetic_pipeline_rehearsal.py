from __future__ import annotations

from pathlib import Path

from scripts.audit_sf3_a0a5_synthetic_pipeline_rehearsal import run_rehearsal

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a0a5_synthetic_pipeline_rehearsal_v1.json"
CONTRACT_V2 = ROOT / "configs/sf3_a0a5_synthetic_pipeline_rehearsal_v2.json"


def test_procedural_three_stock_chain_reaches_blind_adjudication() -> None:
    report = run_rehearsal(CONTRACT)
    assert report["automatic_pass"] is True
    assert report["acquisition_rows"] == 108
    assert report["blind_overall_assignment_accuracy"] == 1.0
    assert report["persistent_pixel_or_report_writes"] == 0
    assert report["network_reads"] == 0
    assert all(
        facts["a2_automatic_pass"]
        and facts["a4_automatic_pass"]
        and facts["confirmation_outputs"] == 4
        for facts in report["stocks"].values()
    )


def test_procedural_three_stock_chain_runs_pooled_global_control() -> None:
    forward = run_rehearsal(CONTRACT_V2)
    reverse = run_rehearsal(CONTRACT_V2, reverse=True)
    assert forward == reverse
    assert forward["automatic_pass"] is True
    assert forward["pooled_global_control_automatic_pass"] is True
    assert (
        forward["pooled_global_control_decision"]
        == "OPEN_STOCK_SPECIFIC_K1_AS_INCREMENTAL_OVER_POOLED_GLOBAL_CONTROL"
    )
