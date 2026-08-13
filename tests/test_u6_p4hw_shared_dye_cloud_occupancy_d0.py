from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.shared_dye_cloud_occupancy_d0 import _validate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4hw_shared_dye_cloud_occupancy_d0_v1.json"


def test_contract_is_frozen_and_parent_bound() -> None:
    contract = load_contract(CONFIG)
    _validate(contract)
    for binding in contract["parents"].values():
        if isinstance(binding, dict) and "path" in binding:
            assert (ROOT / binding["path"]).is_file()


def test_mechanism_and_gate_drift_reject() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["mechanisms"]["candidate"]["shared_factor_fraction"] = 0.9
    with pytest.raises(ValueError, match="mechanism drift"):
        _validate(changed)
    changed = copy.deepcopy(contract)
    changed["development"]["photographic_pixels_allowed"] = True
    with pytest.raises(ValueError, match="mechanism drift"):
        _validate(changed)
    changed = copy.deepcopy(contract)
    changed["gates"][
        "maximum_candidate_to_control_high_frequency_chroma_p999_ratio"
    ] = 0.81
    with pytest.raises(ValueError, match="mechanism drift"):
        _validate(changed)
