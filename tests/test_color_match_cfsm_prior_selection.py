from __future__ import annotations

from pathlib import Path

from scripts.evaluate_cfsm_prior_selection import (
    confirmation_decision,
    evaluate_split,
    load_contract,
    selected_manifest_rows,
    validation_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "reference_match_cfsm_prior_selection_v1.json"
)


def _result(median: float, rate: float, worst: float) -> dict:
    return {
        "summary": {
            "median_captured_style_fraction": median,
            "improvement_rate": rate,
            "worst_captured_style_fraction": worst,
            "constraint_pass_fraction": 1.0,
            "identity_fallback_count": 0,
        }
    }


def test_prior_selection_contract_freezes_disjoint_operator_slices() -> None:
    config, _, manifest = load_contract(CONFIG)
    development = selected_manifest_rows(config, manifest, "development")
    validation = selected_manifest_rows(config, manifest, "validation")
    confirmation = selected_manifest_rows(config, manifest, "confirmation")

    assert len(development) == 24
    assert len(validation) == 12
    assert len(confirmation) == 24
    assert {row["split"] for row in development} == {"fit"}
    assert {row["split"] for row in validation} == {"validation"}
    assert {row["split"] for row in confirmation} == {"confirmation"}
    assert not (
        {row["operator_id"] for row in development}
        & {row["operator_id"] for row in validation}
    )
    assert not (
        {row["operator_id"] for row in validation}
        & {row["operator_id"] for row in confirmation}
    )


def test_one_row_executes_every_requested_prior_without_fallback() -> None:
    config, parent, manifest = load_contract(CONFIG)
    row = selected_manifest_rows(config, manifest, "development")[0]
    kinds = ("fixed-uniform-cube", "analytic-mid-key-neutral-v1")
    results = evaluate_split(
        config,
        parent,
        [row],
        split="development",
        prior_kinds=kinds,
    )

    assert set(results) == set(kinds)
    for result in results.values():
        summary = result["summary"]
        assert summary["observations"] == 2
        assert summary["constraint_pass_fraction"] == 1.0
        assert summary["identity_fallback_count"] == 0
        assert len(result["records"]) == 2


def test_validation_and_confirmation_gates_do_not_promote_ties() -> None:
    config, _, _ = load_contract(CONFIG)
    tied = {
        "fixed-uniform-cube": _result(0.20, 0.8, -0.1),
        "analytic-mid-key-neutral-v1": _result(0.20, 0.8, -0.1),
    }
    rejected = validation_decision(config, tied)

    assert rejected["status"] == "no-analytic-prior-selected"
    assert rejected["confirmation_open"] is False

    winning = {
        "fixed-uniform-cube": _result(0.20, 0.8, -0.1),
        "analytic-mid-key-neutral-v1": _result(0.27, 0.9, -0.11),
    }
    selected = validation_decision(config, winning)
    confirmed = confirmation_decision(
        config,
        winning,
        "analytic-mid-key-neutral-v1",
    )

    assert selected["status"] == "analytic-prior-selected"
    assert selected["confirmation_open"] is True
    assert confirmed["status"] == "synthetic-confirmation-passed"
    assert confirmed["real_matrix_confirmation_open"] is True
