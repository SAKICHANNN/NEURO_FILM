from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.poisson_binomial_granularity import (
    PoissonBinomialGranularityError,
    evaluate_compatibility,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cd_poisson_binomial_granularity_v1.json"
PARENT_REPORT = (
    ROOT
    / "outputs/experiments/u6_p4aw_kodak_250d_same_sheet_granularity_v1/report_a.json"
)


def test_contract_rejects_uniformity_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["uniformity"] = 0.95
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PoissonBinomialGranularityError, match="contract drift"):
        load_contract(path)


def test_contract_rejects_weaker_baseline_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_improvement_vs_characteristic_slope"] = 0.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PoissonBinomialGranularityError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not PARENT_REPORT.is_file(), reason="P4AW report unavailable")
def test_frozen_poisson_binomial_experiment_is_repeatable() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_compatibility(contract, ROOT)
    second = evaluate_compatibility(contract, ROOT)
    assert first == second
    assert first["split_counts"] == {
        "blue": {"development": 13, "confirmation": 6},
        "green": {"development": 17, "confirmation": 9},
        "red": {"development": 18, "confirmation": 9},
    }
    assert first["scores"]["candidate"]["predicted_sigma_d"] != first["scores"][
        "no_competition_poisson"
    ]["predicted_sigma_d"]
    assert first["parameters"]["candidate"]["uniformity"] == 0.98
    assert not first["automatic_pass"]
    assert first["decision"] == (
        "close_poisson_binomial_density_marginal_without_rescue"
    )
    assert first["stable_evidence_id"] == (
        "2398bdcb0f4acb1962a264169879012d14ded8d38a2377e09ed74aabb3b5c53c"
    )
    assert [name for name, passed in first["gate_results"].items() if not passed] == [
        "improvement_vs_characteristic_slope",
        "median_channel_spearman",
    ]


def test_parent_hash_tamper_fails_before_scoring(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["parents"]["p4aw_report_sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = load_contract(path)
    with pytest.raises(PoissonBinomialGranularityError, match="parent integrity"):
        evaluate_compatibility(contract, ROOT)
