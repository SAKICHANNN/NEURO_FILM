from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_p223_r1cl_browser_wasm_reproducibility import (
    CONFIG,
    canonical_bytes,
    evaluate_reports,
    extract_report_facts,
    validate_config,
)


def _producer_report(config: dict) -> dict:
    expected = config["expected"]
    return {
        "status": expected["producer_status"],
        "stable_identity": expected["stable_identity"],
        "scientific": {
            "wasm_module_bytes": expected["wasm_module_bytes"],
            "wasm_module_sha256": expected["wasm_module_sha256"],
            "runtime_payload": {
                "output_sha256": expected["output_rgb_sha256"],
                "synthetic": "payload",
            },
            "browsers": [{}, {}],
            "process_observations": [{}, {}, {}, {}],
            "gates": {"all": True},
        },
    }


def test_contract_is_frozen_and_browser_binaries_are_distinct() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["execution"]["fresh_exact_commit_clones"] == 2
    assert config["execution"]["media_reads"] == 0
    assert config["execution"]["network_downloads"] == 0
    assert len({row["executable_sha256"] for row in config["browsers"]}) == 2


def test_wrong_schema_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["schema"] = "wrong"
    with pytest.raises(RuntimeError, match="schema differs"):
        validate_config(config)


def test_extract_report_facts_reads_browser_runtime_payload() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = _producer_report(config)
    config["expected"]["scientific_payload_sha256"] = canonical_payload_sha = (
        hashlib.sha256(
            canonical_bytes(report["scientific"]["runtime_payload"])
        ).hexdigest()
    )
    facts = extract_report_facts(report)
    assert facts["scientific_payload_sha256"] == canonical_payload_sha
    assert facts["browser_count"] == 2
    assert facts["process_observation_count"] == 4
    assert facts["all_inner_producer_gates_pass"] is True


def test_report_evaluation_fails_browser_payload_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first_report = _producer_report(config)
    config["expected"]["scientific_payload_sha256"] = hashlib.sha256(
        canonical_bytes(first_report["scientific"]["runtime_payload"])
    ).hexdigest()
    first = canonical_bytes(first_report)
    second_report = copy.deepcopy(first_report)
    second_report["scientific"]["runtime_payload"]["synthetic"] = "drift"
    second = canonical_bytes(second_report)
    gates = evaluate_reports(config, [first, second], temporary_roots_removed=True)
    assert gates["consumer_reports_byte_exact"] is False
    assert gates["scientific_payload_identity_exact"] is False
    assert gates["temporary_roots_removed"] is True
    assert not all(gates.values())


def test_runner_uses_fresh_no_hardlink_clones() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("audit_p223_r1cl_browser_wasm_reproducibility.py")
    )
    text = runner.read_text(encoding="utf-8")
    assert '"--local",\n                    "--no-hardlinks"' in text
    assert '["git", "config", "core.autocrlf", "false"]' in text
