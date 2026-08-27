from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval import bw_two_stock_proxy_baseline as target

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "bw2_d0_hp5_trix_k1_baseline_v1.json"


def test_contract_keeps_claim_and_candidate_boundary() -> None:
    contract = target.load_contract(CONTRACT)
    assert contract["direction"]["this_experiment_is_stock_completion"] is False
    assert [arm["style"] for arm in contract["candidates"]] == ["hp5", "tri_x_400"]
    assert contract["render"]["strength_sweep_allowed"] is False
    assert "No HP5/Tri-X authenticity" in contract["claim_ceiling"]


def test_candidate_relabel_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["candidates"][1]["style"] = "hp5"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(target.BwBaselineError, match="candidate order"):
        target.load_contract(path)


def test_missing_gate_is_rejected(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    del contract["gates"]["minimum_population_median_delta_e76"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(target.BwBaselineError, match="gate is missing"):
        target.load_contract(path)
