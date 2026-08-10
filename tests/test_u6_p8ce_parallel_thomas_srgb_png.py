from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8ce_parallel_thomas_srgb_png_v1.json"


def test_p8ce_freezes_exact_parent_composition() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["candidate"]["p8cc_encoder_unchanged"] is True
    assert contract["candidate"]["p8cd_scheduler_unchanged"] is True
    assert contract["candidate"]["model_profile_or_sample_change_allowed"] is False


def test_p8ce_requires_material_complete_path_speedup() -> None:
    performance = json.loads(CONTRACT.read_text(encoding="utf-8"))["performance"]
    assert performance["maximum_wall_seconds"] <= 13.5
    assert performance["maximum_wall_ratio_vs_p8cc"] <= 0.86
    assert performance["require_exact_p8cc_png_identity"] is True
