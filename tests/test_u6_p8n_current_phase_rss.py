from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p8n_current_phase_rss import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8n_current_phase_rss_v1.json"


def test_p8n_contract_requires_actual_entry_and_two_exact_runs() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(config)

    assert profile["node"] == "U6.P8B"
    assert config["scenario"]["height"] * config["scenario"]["width"] == 12_000_000
    assert config["scenario"]["repeats"] == 2
    assert config["measurement"]["production_entry_required"]
    assert config["measurement"][
        "timing_and_rss_excluded_from_stable_evidence_id"
    ]
    assert not config["measurement"]["post_result_retuning_allowed"]
