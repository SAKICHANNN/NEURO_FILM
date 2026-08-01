from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.nasa_measured_joint_compatibility import (
    MeasuredJointCompatibilityError,
    evaluate_measured_joint_compatibility,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4au_nasa_measured_joint_compatibility_v1.json"
SOURCE = (
    ROOT / "outputs/eval/u6_p4at_nasa_joint_mtf_granularity_source_v1/report_run1.json"
)


def test_contract_keeps_mtf_out_of_fit() -> None:
    contract = load_contract(CONTRACT)
    assert contract["fit"]["fit_reads_mtf"] is False
    assert contract["evaluation"]["mtf_values_are_never_fit"] is True
    assert contract["evidence_role"].startswith("development-only")


def test_contract_rejects_control_gate_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_dispersion_improvement_vs_no_diffusion"] = -1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MeasuredJointCompatibilityError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not SOURCE.is_file(), reason="P4AT source report unavailable")
def test_evaluator_is_exact_and_keeps_source_contradiction() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_measured_joint_compatibility(contract, ROOT)
    second = evaluate_measured_joint_compatibility(contract, ROOT)
    assert first == second
    assert set(first["fits"]) == {"1", "2", "3"}
    assert first["gate_results"]["source_count_contradiction_retained"]
    assert all(fit["sigma_um"] > 0.0 for fit in first["fits"].values())
    assert first["evidence_role"].startswith("development-only")
