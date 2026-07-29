from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8aw_native_display_v4_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8aw_native_display_v4_resources_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p8aw_native_display_v4_resources_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8aw_contract_reuses_frozen_resource_gates() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(config)
    frozen = json.loads(
        (ROOT / config["frozen_resource_contract"]).read_text()
    )
    assert config["scenario"] == frozen["scenario"]
    assert config["tile_rows"] == frozen["tile_rows"]
    assert config["safety"] == frozen["safety"]
    assert config["gates"] == frozen["gates"]
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert config["execution"]["ao6_display_abi"] == "v4"
    assert config["expected_output_sha256"] == frozen[
        "expected_output_sha256"
    ]
    assert config["profile_compiler_config_sha256"] == frozen[
        "profile_compiler_config_sha256"
    ]


def test_p8aw_contract_rejects_evidence_identity_drift() -> None:
    config = json.loads(CONFIG.read_text())
    cases = (
        ("expected_output_sha256", "0" * 64),
        ("profile_compiler_config_sha256", "0" * 64),
    )
    for key, value in cases:
        mutated = json.loads(json.dumps(config))
        mutated[key] = value
        try:
            validate_contract(mutated)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError(f"P8AW accepted drifted {key}")

    mutated = json.loads(json.dumps(config))
    mutated["component_dll_sha256"]["display"] = "0" * 64
    try:
        validate_contract(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError("P8AW accepted drifted display DLL")


def test_p8aw_decision_binds_formal_failure_evidence() -> None:
    decision = json.loads(DECISION.read_text())
    report_path = ROOT / decision["formal_report"]
    report = json.loads(report_path.read_text())
    assert _sha256(CONFIG) == decision["contract_sha256"]
    assert _sha256(report_path) == decision["formal_report_sha256"]
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert report["output_repeat_and_v1_exact"]
    assert report["memory_pass"]
    assert not report["elapsed_pass"]
    assert not report["pass"]
    assert decision["result"]["status"] == "fail-latency"
    assert decision["next_leaf"].startswith("U6.P8AX")
