from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cd_native_thomas_parallel_quantized_sink_v1.json"


def test_p8cd_freezes_only_parallel_channel_scheduling() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert candidate["worker_count"] == 3
    assert candidate["one_private_workspace_per_channel"] is True
    assert candidate["arithmetic_profiles_and_quantizer_unchanged"] is True
    assert candidate["model_or_profile_change_allowed"] is False


def test_p8cd_requires_material_wall_reduction_without_claim_expansion() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["performance"]["maximum_wall_ratio_vs_p8cb"] <= 0.81
    assert "not an algorithm/profile change" in contract["claim_ceiling"]
