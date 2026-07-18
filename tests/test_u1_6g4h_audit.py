from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u1_6g4h_research_adapter import (
    failure_probes,
    static_isolation,
    synthetic_case,
)


ROOT = Path(__file__).resolve().parents[1]


def test_g4h_synthetic_case_is_repeat_identical() -> None:
    case = {"shape": [73, 101, 3], "seed": 451}
    first = synthetic_case(case)
    second = synthetic_case(case)
    assert first.dtype.name == "float32"
    assert first.tobytes() == second.tobytes()


def test_g4h_failure_probes_publish_no_partial_result() -> None:
    result = failure_probes()
    assert result["passed"]
    assert result["executor_failure"]["partial_result_count"] == 0
    assert result["second_row_failure"]["partial_result_count"] == 0


def test_g4h_static_isolation_matches_frozen_contract() -> None:
    config = json.loads(
        (ROOT / "configs/u1_6g4h_research_adapter_v1.json").read_text(
            encoding="utf-8"
        )
    )
    result = static_isolation(config["pre_contract_head"])
    assert result["passed"]
    assert result["production_import_count"] == 0
    assert result["renderer_cli_schema_change_count"] == 0
