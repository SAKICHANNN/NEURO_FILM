from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.gamutmlp_rec2020_stress_confirmation import (
    CONTRACT_SHA256,
    GamutMLPStressConfirmationError,
    _validate_inputs,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u1_4c7_gamutmlp_rec2020_stress_confirmation_v1.json"


def test_contract_reuses_exact_c4_gates_and_disables_product() -> None:
    contract = load_contract(CONTRACT)
    c4 = json.loads(
        (ROOT / contract["parents"]["c4_contract"]["path"]).read_text(encoding="utf-8")
    )
    assert contract["automatic_gates"] == c4["automatic_gates"]
    assert contract["render"]["styles"] == c4["render"]["styles"]
    assert contract["render"]["gamut_modes"] == c4["render"]["gamut_modes"]
    assert contract["ingress"]["source_or_operator_fitting_allowed"] is False
    assert contract["production_default_changed"] is False
    assert (
        CONTRACT_SHA256
        == "ecf68184e637525dcc0ff08626153e66fa227fc532e65048476356e929a4ab84"
    )


def test_reviewed_stress_rows_validate() -> None:
    rows = _validate_inputs(load_contract(CONTRACT), ROOT)
    assert len(rows) == 24
    assert len({row["camera"] for row in rows}) == 8
    assert len({row["source_id"] for row in rows}) == 24
    assert (
        min(row["precompression_rec2020_out_of_gamut_fraction"] for row in rows)
        >= 0.0001
    )


def test_contract_drift_rejects(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["automatic_gates"]["minimum_worst_render_style_retention_ratio"] = 0.0
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GamutMLPStressConfirmationError, match="contract hash drift"):
        load_contract(mutated)
