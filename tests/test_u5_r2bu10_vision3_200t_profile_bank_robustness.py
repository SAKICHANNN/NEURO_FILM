from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.eval.vision3_200t_profile_bank_robustness import (
    ALL_STOCKS,
    ProfileBankRobustnessError,
    audit_profile_bank,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu10_vision3_200t_profile_bank_robustness_v1.json"
DATA_ROOT = Path(os.environ.get("NF_BU10_DATA_ROOT", ROOT))


def test_bu10_four_stock_audit_is_exact() -> None:
    config = load_contract(CONFIG)
    first = audit_profile_bank(config, ROOT, data_root=DATA_ROOT)
    second = audit_profile_bank(config, ROOT, data_root=DATA_ROOT)
    assert first == second
    assert first["case_count"] == 80
    assert set(first["recall_by_stock"]) == set(ALL_STOCKS)
    assert all(first["gate_results"].values()) == first["audit_pass"]
    assert "not independent real-film" in first["claim_ceiling"]


def test_bu10_rejects_gate_relaxation(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["minimum_joint_top1_accuracy"] = 0.9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProfileBankRobustnessError, match="contract drift"):
        load_contract(changed)
