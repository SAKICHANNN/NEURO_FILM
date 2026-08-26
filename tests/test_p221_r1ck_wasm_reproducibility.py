from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p221_r1ck_wasm_reproducibility import (
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
            "output_sha256": expected["output_rgb_sha256"],
        },
    }


def test_contract_is_frozen_and_has_two_fresh_clones() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["execution"]["fresh_exact_commit_clones"] == 2
    assert config["execution"]["producer_worktree_writes"] == 0
    assert config["execution"]["network_downloads"] == 0


def test_runner_avoids_read_only_local_clone_hardlinks() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/audit_p221_r1ck_wasm_reproducibility.py"
    ).read_text(encoding="utf-8")
    assert '"--local",\n                    "--no-hardlinks"' in runner


def test_extract_report_facts_uses_scientific_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    facts = extract_report_facts(_producer_report(config))
    assert facts == {
        "producer_status": config["expected"]["producer_status"],
        "stable_identity": config["expected"]["stable_identity"],
        "wasm_module_bytes": config["expected"]["wasm_module_bytes"],
        "wasm_module_sha256": config["expected"]["wasm_module_sha256"],
        "output_rgb_sha256": config["expected"]["output_rgb_sha256"],
    }


def test_report_evaluation_fails_identity_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = _producer_report(config)
    first = canonical_bytes(report)
    report["scientific"]["wasm_module_sha256"] = "0" * 64
    second = canonical_bytes(report)
    gates = evaluate_reports(config, [first, second], temporary_roots_removed=True)
    assert gates["consumer_reports_byte_exact"] is False
    assert gates["wasm_module_identity_exact"] is False
    assert gates["temporary_roots_removed"] is True
    assert not all(gates.values())
