from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8ae_native_standard_f32_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u6_p8ae_native_standard_f32_resources_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ae_contract_binds_f32_and_float64_decisions() -> None:
    config = json.loads(CONFIG.read_text())
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["reference_decision"]) == config[
        "reference_decision_sha256"
    ]
    reference = validate_contract(config)
    assert reference["node"] == "U6.P8AC"
    assert config["tile_rows"] == 32
    assert config["scenario"]["height"] * config["scenario"][
        "width"
    ] == 12_000_000
    assert config["gates"][
        "maximum_median_rss_ratio_vs_float64_reference"
    ] == 0.8
    assert config["execution"]["foreign_processes_must_not_be_terminated"]
