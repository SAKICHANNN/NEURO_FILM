from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.noever_vision3_resolution_order import (
    ResolutionOrderError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu12_noever_vision3_resolution_order_v1.json"


def test_bu12_exact_order_result() -> None:
    first = evaluate(load_contract(CONTRACT), ROOT)
    second = evaluate(load_contract(CONTRACT), ROOT)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["primary_assignment"]["noninferior_counts"] == {"50D": 6, "200T": 6}
    assert first["primary_assignment"]["strict_win_counts"] == {"50D": 3, "200T": 2}
    assert first["primary_assignment"]["total_strict_wins"] == 5
    assert first["assignment_score_margin"] >= 1
    assert all(len(row["aliasing_flags"]) == 3 for row in first["rows"])


def test_bu12_fails_closed_on_source_or_parent_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    drifted = copy.deepcopy(payload)
    drifted["source"]["values_lp_per_mm"]["500T"]["super16"]["4k"] = 41
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ResolutionOrderError, match="contract drift"):
        load_contract(path)

    locked = copy.deepcopy(payload)
    locked["parents"]["bu10_evidence"]["sha256"] = "0" * 64
    with pytest.raises(ResolutionOrderError, match="locked input drift"):
        evaluate(locked, ROOT)
