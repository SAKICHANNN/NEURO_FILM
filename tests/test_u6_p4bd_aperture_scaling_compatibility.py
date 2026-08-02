from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.aperture_scaling_compatibility import (
    ApertureScalingCompatibilityError,
    evaluate_compatibility,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bd_aperture_scaling_compatibility_v1.json"


def test_contract_rejects_post_score_exponent_fit(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["parameter_fit_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApertureScalingCompatibilityError, match="contract drift"):
        load_contract(path)


def test_frozen_aperture_law_closes_without_rescue() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_compatibility(contract, ROOT)
    second = evaluate_compatibility(contract, ROOT)
    assert first == second
    assert not first["automatic_pass"]
    assert first["decision"] == "close_independent_cell_spatial_interpretation"
    assert first["complete_comparison_count"] == 27
    assert first["gate_results"]["improvement_over_constant"]
    assert not first["gate_results"]["median_relative_error"]
    assert not first["gate_results"]["worst_relative_error"]
    assert not first["gate_results"]["observed_exponent_compatible"]
