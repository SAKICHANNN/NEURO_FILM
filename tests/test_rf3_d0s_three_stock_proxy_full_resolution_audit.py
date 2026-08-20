from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import three_stock_proxy_full_resolution_audit as target

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "rf3_d0s_three_stock_proxy_full_resolution_severe_audit_v1.json"


def test_contract_keeps_ao6_and_claim_ceiling_narrow() -> None:
    contract = target.load_contract(CONTRACT)
    assert contract["expected"]["ao6_role"] == "velvia_50_display_proxy_look_approximation_baseline_only"
    assert "No stock truth" in contract["claim_ceiling"]
    assert contract["audit"]["maximum_new_output_boundary_fraction"] == 0.0


def test_contract_rejects_ao6_role_inflation(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = copy.deepcopy(contract)
    changed["expected"]["ao6_role"] = "generic_multistock_simulator"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(target.ThreeStockFullResolutionAuditError, match="AO6 role"):
        target.load_contract(path)


def test_crop_selection_is_deterministic() -> None:
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    array[128:, 128:] = np.indices((128, 128)).sum(axis=0)[..., None] % 2 * 255
    first = target._crop_coordinates(array, 64, 32)
    second = target._crop_coordinates(array.copy(), 64, 32)
    assert first == second
    assert first["lowest_source_gradient"] == [0, 0, 64, 64]
    assert first["highest_source_gradient"] != first["lowest_source_gradient"]


def test_real_bound_audit_replays_exact(tmp_path: Path) -> None:
    first = target.evaluate(CONTRACT, ROOT, tmp_path / "a")
    second = target.evaluate(CONTRACT, ROOT, tmp_path / "b")
    assert first["automatic_integrity_pass"] is True
    assert first["full_resolution_outputs_verified"] == 64
    assert first["scientific_identity"] == second["scientific_identity"]
    assert first["risk_crop_sheet_sha256"] == second["risk_crop_sheet_sha256"]
    assert not first["integrity_failures"]
