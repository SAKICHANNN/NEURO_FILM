from pathlib import Path

import pytest

from src.eval.characteristic_vs_ao6_fresh import (
    CONTRACT_SHA256,
    CharacteristicVsAo6FreshError,
    load_contract,
)
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb12_characteristic_vs_ao6_fresh_v1.json"


def test_contract_is_frozen():
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB12"


def test_contract_drift_fails_closed(tmp_path: Path):
    path = tmp_path / "changed.json"
    path.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(CharacteristicVsAo6FreshError, match="hash drift"):
        load_contract(path)
