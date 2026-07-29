from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8ay_native_ordered_pipeline_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u6_p8ay_native_ordered_pipeline_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8ay_native_ordered_pipeline_resources_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ay_contract_reuses_frozen_12mp_gates() -> None:
    config = json.loads(CONFIG.read_text())
    parent = validate_contract(config)
    frozen = json.loads(
        (ROOT / config["frozen_resource_contract"]).read_text()
    )
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert parent["result"]["selected_pipeline_workers"] == 4
    assert config["scenario"] == frozen["scenario"]
    assert config["safety"] == frozen["safety"]
    assert config["gates"] == frozen["gates"]
    assert config["expected_output_sha256"] == frozen[
        "expected_output_sha256"
    ]


def test_p8ay_decision_binds_formal_pass_evidence() -> None:
    decision = json.loads(DECISION.read_text())
    report_path = ROOT / decision["formal_report"]
    report = json.loads(report_path.read_text())
    assert _sha256(CONFIG) == decision["contract_sha256"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["output_repeat_and_v1_exact"]
    assert report["memory_pass"]
    assert report["elapsed_pass"]
    assert report["pass"]
    assert report["pipeline_workers"] == 4
    assert report["max_in_flight"] == 4
    assert decision["result"]["status"] == "pass"
    assert decision["next_leaf"].startswith("U6.P8AZ")
