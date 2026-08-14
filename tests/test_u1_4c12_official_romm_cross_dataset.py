from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.official_romm_cross_dataset_confirmation import (
    OfficialROMMConfirmationError,
    _validate_sources,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c12_official_romm_cross_dataset_v1.json"


def test_contract_freezes_official_profile_and_unchanged_c11_mapper() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U1.4C12"
    assert contract["source"]["expected_rows"] == 24
    assert contract["source"]["expected_cameras"] == 8
    assert contract["mapper"]["softness"] == 1.0 / 64.0
    assert contract["mapper"]["rgb16_margin"] == 2.0 / 65535.0
    assert contract["mapper"]["cohort_fitting_allowed"] is False


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(CONTRACT.read_bytes() + b" ")
    with pytest.raises(OfficialROMMConfirmationError, match="contract hash drift"):
        load_contract(changed)


def test_source_manifest_and_embedded_profiles_are_exact() -> None:
    contract = load_contract(CONTRACT)
    source_config, rows = _validate_sources(contract, ROOT)
    assert source_config["source"]["maximum_evaluation_side"] == 512
    assert len(rows) == 24
    assert len({row["camera"] for row in rows}) == 8
    assert all(row["allowed_use"] == "internal_research_only" for row in rows)
