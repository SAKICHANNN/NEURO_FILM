from __future__ import annotations

from pathlib import Path

from scripts.evaluate_cfsm_prior_selection import (
    evaluate_split,
    selected_manifest_rows,
)
from scripts.evaluate_cfsm_quantile import (
    confirmation_decision,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_cfsm_quantile_v1.json"


def _result(
    median: float,
    rate: float,
    worst: float,
    *,
    boundary: float = 0.0,
) -> dict:
    return {
        "summary": {
            "median_captured_style_fraction": median,
            "improvement_rate": rate,
            "worst_captured_style_fraction": worst,
            "maximum_new_boundary_fraction": boundary,
            "constraint_pass_fraction": 1.0,
            "identity_fallback_count": 0,
        }
    }


def test_quantile_contract_reserves_stress_confirmation() -> None:
    config, _, manifest = load_contract(CONFIG)
    development = selected_manifest_rows(config, manifest, "development")
    confirmation = selected_manifest_rows(config, manifest, "confirmation")
    assert len(development) == 24
    assert len(confirmation) == 24
    assert {row["split"] for row in development} == {"fit"}
    assert {row["split"] for row in confirmation} == {"stress"}
    assert not (
        {row["operator_id"] for row in development}
        & {row["operator_id"] for row in confirmation}
    )


def test_one_operator_executes_quantile_and_uniform_controls() -> None:
    config, parent, manifest = load_contract(CONFIG)
    row = selected_manifest_rows(config, manifest, "development")[0]
    results = evaluate_split(
        config,
        parent,
        [row],
        split="development",
        prior_kinds=tuple(config["prior_kinds"]),
    )
    assert set(results) == set(config["prior_kinds"])
    for result in results.values():
        assert result["summary"]["observations"] == 2
        assert result["summary"]["constraint_pass_fraction"] == 1.0
        assert result["summary"]["identity_fallback_count"] == 0


def test_quantile_confirmation_gate_rejects_ties() -> None:
    config, _, _ = load_contract(CONFIG)
    tie = {
        "fixed-uniform-cube": _result(0.10, 0.80, -0.10),
        "fixed-uniform-cube-monotone-quantile-v1": _result(
            0.10, 0.80, -0.10
        ),
    }
    rejected = confirmation_decision(config, tie)
    assert rejected["status"] == "quantile-route-closed"
    assert rejected["real_photo_review_open"] is False
    assert rejected["product_integration_open"] is False

    winner = {
        "fixed-uniform-cube": _result(0.10, 0.80, -0.10),
        "fixed-uniform-cube-monotone-quantile-v1": _result(
            0.14, 0.80, -0.11
        ),
    }
    passed = confirmation_decision(config, winner)
    assert passed["status"] == "quantile-synthetic-confirmation-passed"
    assert passed["real_photo_review_open"] is True
    assert passed["product_integration_open"] is False
