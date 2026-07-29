from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.velvia_proxy_cube_flow import (
    VelviaProxyCubeFlowError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ar0_velvia_proxy_cube_flow_v1.json"


def test_contract_accepts_exact_frozen_inputs() -> None:
    validate_contract(json.loads(CONFIG.read_bytes()), ROOT)


def test_contract_rejects_candidate_drift() -> None:
    config = json.loads(CONFIG.read_bytes())
    changed = copy.deepcopy(config)
    changed["candidate"]["axis_size"] = 4
    with pytest.raises(VelviaProxyCubeFlowError, match="candidate drift"):
        validate_contract(changed, ROOT)


def test_contract_rejects_removed_exploration_disclosure() -> None:
    config = json.loads(CONFIG.read_bytes())
    changed = copy.deepcopy(config)
    changed["development_disclosure"][
        "small_K2_K3_K4_smoothness_probe_seen_before_freeze"
    ] = False
    with pytest.raises(VelviaProxyCubeFlowError, match="disclosure drift"):
        validate_contract(changed, ROOT)
