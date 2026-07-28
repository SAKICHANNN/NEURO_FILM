from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_u5_r2ao5v_three_way_visual import CONFIG_SHA256
from src.eval.three_way_look_visual import blind_orders


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao5v_three_way_visual_v1.json"


def test_ao5v_contract_identity() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    assert len(config["looks"]) == 3
    assert config["blind"]["three_distinct_permutations_required"]
    assert config["stock_response_claim_allowed"] is False


def test_blind_orders_are_repeat_exact_and_distinct() -> None:
    roles = ["combined_proxy", "chart_proxy", "b0_comparator"]
    first = blind_orders(20260729, roles, 3)
    second = blind_orders(20260729, roles, 3)
    assert first == second
    assert len({tuple(order) for order in first}) == 3
    assert all(sorted(order) == sorted(roles) for order in first)
