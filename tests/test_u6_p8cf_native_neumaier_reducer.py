from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cf_native_neumaier_reducer_v1.json"


def test_p8cf_freezes_independent_reducer_without_thomas_source_change() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert candidate["thomas_rows_v1_unchanged"] is True
    assert candidate["ordered_float32_to_double_accumulation"] is True
    assert candidate["model_profile_or_output_change_allowed"] is False


def test_p8cf_requires_exact_bits_and_material_speedup() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["conformance"]["require_python_native_mean_bits_exact"] is True
    assert contract["performance"]["maximum_wall_ratio_vs_p8cd"] <= 0.80
