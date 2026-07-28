from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_u5_r2ao3v_factorized_chart_visual import CONFIG_SHA256
from src.eval.factorized_chart_visual import blind_order


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao3v_factorized_chart_visual_v1.json"


def test_ao3v_contract_is_exact_and_nonpromoting() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    assert config["blind"]["round_count"] == 3
    assert config["full_resolution"]["all_nine_gold_required"]
    assert not config["production_integration_allowed"]
    assert not config["stock_response_claim_allowed"]


def test_blind_orders_are_repeatable_and_cover_both_roles() -> None:
    orders = [blind_order(20260728, index) for index in range(1, 4)]
    assert orders == [blind_order(20260728, index) for index in range(1, 4)]
    assert all(set(order) == {"candidate", "comparator"} for order in orders)
