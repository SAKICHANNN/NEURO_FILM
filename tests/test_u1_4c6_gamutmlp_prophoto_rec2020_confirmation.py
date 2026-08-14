from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.gamutmlp_prophoto_rec2020_confirmation import (
    CONTRACT_SHA256,
    GamutMLPConfirmationError,
    _validate_inputs,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c6_gamutmlp_prophoto_rec2020_confirmation_v1.json"


def test_contract_identity_scope_and_unchanged_c4_gates() -> None:
    contract = load_contract(CONTRACT)
    c4 = json.loads(
        (ROOT / contract["parents"]["c4_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert CONTRACT_SHA256 == "eae16a5eac0abb86a55a8409a0af834cc94f04929b9d3ef096b488ee47fea7b8"
    assert contract["automatic_gates"] == c4["automatic_gates"]
    assert contract["render"]["styles"] == c4["render"]["styles"]
    assert contract["render"]["gamut_modes"] == c4["render"]["gamut_modes"]
    assert contract["ingress"]["source_or_operator_fitting_allowed"] is False
    assert contract["production_default_changed"] is False


def test_reviewed_source_and_parent_identities_validate() -> None:
    rows = _validate_inputs(load_contract(CONTRACT), ROOT)
    assert len(rows) == 24
    assert len({row["camera"] for row in rows}) == 8
    assert all(row["visual_eligible"] is True for row in rows)


def test_contract_drift_rejects(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["automatic_gates"]["minimum_worst_render_style_retention_ratio"] = 0.0
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GamutMLPConfirmationError, match="contract hash drift"):
        load_contract(mutated)
