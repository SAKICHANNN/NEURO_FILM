from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.gamutmlp_rec2020_stress_source_audit import (
    CONTRACT_SHA256,
    GamutMLPStressSourceError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u1_4c7_gamutmlp_rec2020_stress_source_v1.json"


def test_contract_is_frozen_and_source_only() -> None:
    payload = load_contract(CONFIG)
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert payload["selection"]["expected_rows"] == 24
    assert (
        payload["eligibility"][
            "minimum_precompression_rec2020_out_of_gamut_fraction_per_row"
        ]
        == 0.0001
    )
    assert payload["branches"]["fail"].startswith("close_this_exact")
    assert payload["production_default_changed"] is False


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["selection"]["rows_per_camera"] = 4
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GamutMLPStressSourceError, match="hash drift"):
        load_contract(path)
