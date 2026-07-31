from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_callier_measurement_compatibility import (
    CallierMeasurementCompatibilityError,
    evaluate_compatibility,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6p_fixed_callier_measurement_compatibility_v1.json"
REPORT = ROOT / "outputs/eval/u6_p6o_callier_measurement_source_v1/report_run1.json"


def test_contract_freezes_profile_and_measurement_without_retuning() -> None:
    contract = load_contract(CONTRACT)
    assert contract["evaluation"]["collimation"] == 1.0
    assert not contract["evaluation"]["post_result_retuning_allowed"]
    assert contract["gates"]["no_parameter_fit"]


def test_contract_rejects_tolerance_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["silver_model_envelope_absolute_tolerance"] = 0.5
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CallierMeasurementCompatibilityError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not REPORT.is_file(), reason="P6O report unavailable")
def test_fixed_profile_compatibility_is_repeat_exact_and_closes() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_compatibility(contract, ROOT)
    second = evaluate_compatibility(contract, ROOT)
    assert first == second
    assert not first["passed"]
    assert first["decision"] == (
        "close_fixed_p6n_quantitative_compatibility_without_retuning"
    )
    assert first["gate_results"]["dye_reported_identity"]
    assert first["gate_results"]["dye_raw_ratio_identity"]
    assert first["gate_results"]["silver_exceeds_dye_same_level"]
    assert not first["gate_results"]["silver_reported_envelope"]
    assert not first["gate_results"]["silver_raw_ratio_envelope"]
