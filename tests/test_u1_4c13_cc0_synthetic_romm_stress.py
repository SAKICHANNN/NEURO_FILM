from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.cc0_synthetic_romm_stress_confirmation import (
    CC0SyntheticROMMConfirmationError,
    _validate_sources,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c13_cc0_synthetic_romm_stress_v1.json"


def test_contract_freezes_cc0_synthetic_stress_without_fit() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U1.4C13"
    assert contract["source"]["expected_rows"] == 9
    assert contract["source"]["expected_makes"] == 9
    assert contract["source"]["minimum_rec2020_out_of_gamut_fraction"] == 0.005
    assert contract["source"]["require_oog_per_source"] is True
    assert contract["mapper"]["cohort_fitting_allowed"] is False


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(CONTRACT.read_bytes() + b" ")
    with pytest.raises(CC0SyntheticROMMConfirmationError, match="contract hash drift"):
        load_contract(changed)


def test_source_manifest_is_cc0_and_every_row_is_oog() -> None:
    contract = load_contract(CONTRACT)
    source_config, rows = _validate_sources(contract, ROOT)
    assert source_config["source"]["maximum_evaluation_side"] == 1600
    assert len(rows) == 9
    assert len({row["make"] for row in rows}) == 9
    assert all(
        row["rec2020_out_of_gamut_fraction"] >= 0.005 for row in rows
    )
