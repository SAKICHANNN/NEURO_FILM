from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_u5_r2aq4gv_b0_recorder_proxy_residual_visual import (
    CONFIG_SHA256,
)
from src.eval.three_way_look_visual import (
    SUPPORTED_EXPERIMENT_IDS,
    blind_orders,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq4gv_b0_recorder_proxy_residual_visual_v1.json"
)


def test_aq4gv_contract_identity_and_lineage() -> None:
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    assert config["experiment_id"] in SUPPORTED_EXPERIMENT_IDS
    assert len(config["looks"]) == 3
    assert len(config["auxiliary_controls"]) == 2
    assert config["parent_automatic_report_sha256"] == (
        "83fc7cac199ad0920f11cd9a28e3ff451fd9f5880d74cd608d962e5f3ee76abb"
    )
    assert config["stock_response_claim_allowed"] is False


def test_aq4gv_blind_orders_are_repeat_exact_and_distinct() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    roles = [row["role"] for row in config["looks"]]
    first = blind_orders(config["blind"]["seed"], roles, 3)
    second = blind_orders(config["blind"]["seed"], roles, 3)
    assert first == second
    assert len({tuple(order) for order in first}) == 3
    assert all(sorted(order) == sorted(roles) for order in first)
