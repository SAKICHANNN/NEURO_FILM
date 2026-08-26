from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_p253_reference_product_chain_android_runtime import (
    P253RuntimeError,
    _canonical_bytes,
    _fixture_rows,
    _truth_table_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "configs/p253_reference_product_chain_android_runtime_v1.json").read_text(
        encoding="utf-8"
    )
)
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/reference_product_chain_conformance_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_orders_are_exact_reversals_with_same_canonical_rows() -> None:
    normal_fixture = _fixture_rows(FIXTURE, "normal")
    reverse_fixture = _fixture_rows(FIXTURE, "reverse")
    assert normal_fixture == list(reversed(reverse_fixture))
    assert sorted(normal_fixture) == sorted(reverse_fixture)
    normal_truth = _truth_table_rows(CONTRACT, "normal")
    reverse_truth = _truth_table_rows(CONTRACT, "reverse")
    assert normal_truth == list(reversed(reverse_truth))
    assert sorted(normal_truth, key=json.dumps) == sorted(reverse_truth, key=json.dumps)


def test_contract_freezes_complete_truth_table_and_private_ceiling() -> None:
    assert len(CONTRACT["truth_table"]) == 8
    assert (
        sum(
            row["expected"] == "authorized-for-staging"
            for row in CONTRACT["truth_table"]
        )
        == 1
    )
    assert CONTRACT["runtime"]["api_level"] == 34
    assert CONTRACT["runtime"]["abi"] == "x86_64"
    assert "arm64-v8a link-only" in CONTRACT["claim_ceiling"]
    assert CONTRACT["decision_if_pass"].startswith("PASS_PRIVATE_")


def test_unknown_order_fails_closed() -> None:
    with pytest.raises(P253RuntimeError, match="unsupported enumeration order"):
        _fixture_rows(FIXTURE, "random")
    with pytest.raises(P253RuntimeError, match="unsupported enumeration order"):
        _truth_table_rows(CONTRACT, "random")


def test_report_serialization_is_order_independent_after_sorting() -> None:
    normal = {
        "identities": sorted(_fixture_rows(FIXTURE, "normal")),
        "truth": sorted(_truth_table_rows(CONTRACT, "normal"), key=json.dumps),
    }
    reverse = {
        "truth": sorted(_truth_table_rows(CONTRACT, "reverse"), key=json.dumps),
        "identities": sorted(_fixture_rows(FIXTURE, "reverse")),
    }
    assert _canonical_bytes(normal) == _canonical_bytes(reverse)
