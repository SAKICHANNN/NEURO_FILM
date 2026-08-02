from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_aperture_law_confirmation import (
    KodakApertureLawConfirmationError,
    evaluate_confirmation,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bf_kodak_aperture_law_confirmation_v1.json"


def test_contract_rejects_exponent_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["fixed_exponent"] = 0.9
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KodakApertureLawConfirmationError, match="contract drift"):
        load_contract(path)


def test_fixed_law_confirmation_repeats() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_confirmation(contract, ROOT)
    second = evaluate_confirmation(contract, ROOT)
    assert first == second
    assert first["density_group_count"] == 23
    assert first["comparison_count"] == 138
    assert first["automatic_pass"] is True
    assert (
        first["stable_evidence_id"]
        == "ac18ff5e036e2dcbf75799f0a65a7b340065d08ba2d7115f97ff40998c426d7c"
    )
    assert len(first["per_film_median_relative_error"]) == 4
    assert all(row["comparison_aperture_micrometres"] != 48.0 for row in first["rows"])
