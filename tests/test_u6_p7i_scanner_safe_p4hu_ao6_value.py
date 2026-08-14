from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.eval.scanner_safe_p4hu_ao6_value import (
    ARMS,
    _scanner_safe_checks,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u6_p7i_scanner_safe_p4hu_ao6_value_v1.json"


def test_p7i_contract_loads_with_fixed_arms() -> None:
    contract = load_contract(CONFIG)
    assert tuple(contract["comparison"]["arms"]) == ARMS
    assert contract["comparison"]["automatic_gates"] == "inherit_exact_u6_p7h"


def test_scanner_safe_checks_pass_exact_boundary() -> None:
    aggregates = {
        "receipt_count": 9,
        "runtime_ids": ["scanner-safe-residual-analytic-v1"],
        "maximum_limited_pixel_fraction": 0.1,
        "minimum_median_scale": 0.95,
        "minimum_scale": 0.25,
        "maximum_collinearity_error": 1e-12,
    }
    gates = load_contract(CONFIG)["additional_gates"]
    assert all(_scanner_safe_checks(aggregates, gates, expected_rows=9).values())


@pytest.mark.parametrize(
    ("key", "value", "failed_gate"),
    [
        ("maximum_limited_pixel_fraction", 0.100001, "limited_pixel_fraction"),
        ("minimum_median_scale", 0.949999, "median_scale"),
        ("maximum_collinearity_error", 1.1e-12, "collinearity"),
        ("minimum_scale", float("nan"), "finite_scanner_safe_receipts"),
    ],
)
def test_scanner_safe_checks_fail_closed(
    key: str, value: float, failed_gate: str
) -> None:
    aggregates = {
        "receipt_count": 9,
        "runtime_ids": ["scanner-safe-residual-analytic-v1"],
        "maximum_limited_pixel_fraction": 0.0,
        "minimum_median_scale": 1.0,
        "minimum_scale": 1.0,
        "maximum_collinearity_error": 0.0,
    }
    aggregates[key] = value
    gates = copy.deepcopy(load_contract(CONFIG)["additional_gates"])
    checks = _scanner_safe_checks(aggregates, gates, expected_rows=9)
    assert checks[failed_gate] is False
