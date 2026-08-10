from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_trusted_support_selector import (
    evaluate_gates,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/u5_r2bz1_fivek_trusted_support_selector_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_parent_and_only_change_are_exactly_bound() -> None:
    parent, evidence = validate_contract(ROOT, _config())
    assert parent["experiment_id"] == "u5.r2bz0-fedpaie-proxy-exploitation-control-v1"
    assert evidence["decision"] == "close_exact_excess_gap_control_without_transfer_claim"


def test_gate_requires_routing_value_and_tail_improvement() -> None:
    base = {
        "mean_true_gain": 0.35,
        "p95_rmse": 0.06,
        "worst_true_gain": 0.10,
        "mean_positive_proxy_excess": 0.01,
        "maximum_new_boundary_fraction": 0.0,
    }
    primary = dict(base)
    primary.update(
        mean_true_gain=0.38,
        p95_rmse=0.05,
        worst_true_gain=0.12,
        mean_positive_proxy_excess=0.02,
    )
    oracle = dict(base)
    oracle["mean_true_gain"] = 0.50
    gates, _ = evaluate_gates(
        primary=primary,
        fixed=base,
        oracle=oracle,
        win_fraction=0.7,
        config=_config(),
    )
    assert all(gates.values())
    primary["worst_true_gain"] = 0.09
    gates, _ = evaluate_gates(
        primary=primary,
        fixed=base,
        oracle=oracle,
        win_fraction=0.7,
        config=_config(),
    )
    assert gates["primary_improves_worst"] is False
