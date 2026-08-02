from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3v_bracket_eiv_mechanism_recovery_v1.json"


def test_p3v_contract_freezes_source_only_eiv_before_scoring() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    estimator = contract["estimator"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert estimator["iterations"] == 8
    assert "observed source scalars" in estimator["inputs"]
    assert "development target" in estimator["forbidden_inputs"]
    assert estimator["post_fit_changes_allowed"] is False


def test_p3v_contract_keeps_p3u_observations_and_gates_unchanged() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["roles"]["normalizer_reads_targets"] is False
    assert contract["roles"]["mechanism_fit_reads_confirmation"] is False
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "truth gain jitter" in forbidden
