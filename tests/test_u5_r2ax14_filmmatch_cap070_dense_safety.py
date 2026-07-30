import json
from pathlib import Path


def test_ax14_preserves_ax12_automatic_safety_gates() -> None:
    old = json.loads(
        Path(
            "configs/u5_r2ax12_filmmatch_degree5_dense_safety_v1.json"
        ).read_text(encoding="utf-8")
    )
    new = json.loads(
        Path(
            "configs/u5_r2ax14_filmmatch_cap070_dense_safety_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert new["automatic_gate"] == old["automatic_gate"]
    assert (
        new["gates"]["automatic_safety_threshold_relaxation_allowed"]
        is False
    )
