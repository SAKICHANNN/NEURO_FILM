from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.opponent_diffusion_photographic_confirmation import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "u6_p4hz_opponent_diffusion_photographic_confirmation_v1.json"
)


def test_p4hz_contract_loads_and_freezes_fresh_rows() -> None:
    contract = load_contract(CONFIG)
    assert contract["source"]["expected_evaluation_rows"] == 11
    assert len(contract["comparison"]["arms"]) == 4
    assert contract["candidate"]["cohort_fitting_allowed"] is False
    assert contract["decision_if_pass"] == "open_p4hz_severe_visual_review_only"


def test_p4hz_rejects_schema_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
