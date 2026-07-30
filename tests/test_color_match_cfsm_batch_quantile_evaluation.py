from __future__ import annotations

from pathlib import Path

from scripts.evaluate_cfsm_batch_quantile import (
    confirmation_decision,
    evaluate_confirmation,
    load_contract,
    selected_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_cfsm_batch_quantile_v1.json"


def _result(median: float, rate: float, worst: float) -> dict:
    return {
        "summary": {
            "median_captured_style_fraction": median,
            "improvement_rate": rate,
            "worst_captured_style_fraction": worst,
            "maximum_new_boundary_fraction": 0.0,
            "constraint_pass_fraction": 1.0,
            "identity_fallback_count": 0,
        }
    }


def test_batch_quantile_contract_freezes_unused_stress_rows() -> None:
    config, _, manifest = load_contract(CONFIG)
    rows = selected_rows(config, manifest)
    assert len(rows) == 24
    assert {row["split"] for row in rows} == {"stress"}
    assert {
        int(row["within_split_index"]) for row in rows
    } == set(range(8, 16))


def test_one_batch_quantile_row_executes_both_controls() -> None:
    config, parent, manifest = load_contract(CONFIG)
    results = evaluate_confirmation(
        config,
        parent,
        selected_rows(config, manifest)[:1],
    )
    assert set(results) == set(config["candidates"])
    for result in results.values():
        assert result["summary"]["observations"] == 2
        assert result["summary"]["constraint_pass_fraction"] == 1.0
        assert result["summary"]["identity_fallback_count"] == 0
        assert {
            record["source_batch_count"] for record in result["records"]
        } == {3}


def test_batch_quantile_gate_never_claims_reference_only_recipe() -> None:
    config, _, _ = load_contract(CONFIG)
    results = {
        "uploaded-source-batch-gaussian-v1": _result(0.10, 0.8, -0.1),
        "uploaded-source-batch-monotone-quantile-v1": _result(
            0.14, 0.8, -0.11
        ),
    }
    decision = confirmation_decision(config, results)
    assert decision["status"].endswith("passed")
    assert decision["real_photo_review_open"] is True
    assert decision["reference_only_recipe_claimed"] is False
    assert decision["product_integration_open"] is False
