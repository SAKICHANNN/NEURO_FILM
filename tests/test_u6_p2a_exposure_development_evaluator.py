from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p2a_exposure_development import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_passes_all_p2a_gates() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u6_p2a_exposure_development_interpretation_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    first = evaluate(config)
    second = evaluate(config)
    assert first == second
    assert first["automatic_pass"] is True
    assert len(first["routes"]) == 4
    assert all(first["guards"].values())
    assert first["duplicate_tone_curve_symbols"] == 0
