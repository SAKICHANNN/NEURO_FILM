from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_u6_p2c_scanned_interpretation import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_p2c_ablation_repeats_and_passes() -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p2c_scanned_interpretation_ablation_v1.json"
        ).read_text(encoding="utf-8")
    )
    first = evaluate(config)
    second = evaluate(config)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["checks"]["order_control"] is True
    assert first["checks"]["identity_complement"] is True
    assert first["guards"]["non_neutral_bw_rejected"] is True
