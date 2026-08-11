from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u5_r2cb54_analytic_y_chromaticity_streaming_24mp import (
    _validate,
    worker,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb54_analytic_y_chromaticity_streaming_24mp_v1.json"


def test_cb54_resource_contract_binds_cb53_failure_and_streaming_core() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate(config)
    assert config["workload"]["shape"] == [4000, 6000, 3]
    assert config["workload"]["operator_row_chunk"] == 64


def test_cb54_resource_worker_smoke_is_clean(tmp_path: Path) -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    config["workload"]["shape"] = [96, 128, 3]
    result = worker(config, scratch_root=tmp_path)
    assert result["finite_and_bounded"] is True
    assert result["selected_facts"]["global_dose"] == 1.0
    assert result["scratch_residue"] == []
