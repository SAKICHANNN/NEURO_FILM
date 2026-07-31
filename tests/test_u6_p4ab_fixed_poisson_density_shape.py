from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_poisson_density_shape import (
    PoissonDensityShapeError,
    evaluate_shape,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ab_fixed_poisson_density_shape_v1.json"
REPORT = ROOT / "outputs/eval/u6_p4aa_nasa_density_grain_source_v1/report_run1.json"


def test_contract_freezes_no_fit_shape_and_all_nine_series() -> None:
    contract = load_contract(CONTRACT)
    assert contract["gates"]["required_eligible_series"] == 9
    assert contract["gates"]["no_parameter_fit"]
    assert not contract["evaluation"]["post_result_retuning_allowed"]


def test_contract_rejects_rank_gate_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["median_spearman_minimum"] = -1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PoissonDensityShapeError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not REPORT.is_file(), reason="P4AA report unavailable")
def test_unchanged_p4d_measured_shape_is_repeat_exact_and_closes() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_shape(contract, ROOT)
    second = evaluate_shape(contract, ROOT)
    assert first == second
    assert first["metrics"]["eligible_series"] == 9
    assert not first["passed"]
    assert first["decision"] == (
        "close_p4d_measured_shape_compatibility_without_retuning"
    )
    assert not first["gate_results"]["monotone_high_density_shape"]
