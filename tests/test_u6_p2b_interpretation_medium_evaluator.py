from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p2b_interpretation_medium import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_p2b_contract_passes_repeat_exactly() -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p2b_interpretation_medium_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    first = evaluate(config)
    second = evaluate(config)
    assert first == second
    assert first["automatic_pass"] is True
    assert len(first["routes"]) == 4
    assert first["checks"]["negative_slide_bytes"] is True
    assert first["checks"]["negative_slide_polarity"] is True
    assert first["guards"]["non_neutral_bw_rejected"] is True
