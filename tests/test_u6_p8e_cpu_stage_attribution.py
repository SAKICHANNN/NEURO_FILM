from __future__ import annotations

import json
from pathlib import Path

from src.eval.physical_profile_stage_attribution import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8e_cpu_stage_attribution_v1.json"


def test_p8e_contract_binds_frozen_stage_inventory() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    profile = validate_contract(ROOT, config)
    assert profile["node"] == "U6.P8B"
    assert config["stages"][0] == "artifact-reconstruct"
    assert config["stages"][-1] == "t15-c35-residual"
    assert config["execution"]["rss_is_stage_end_snapshot_not_peak"]
