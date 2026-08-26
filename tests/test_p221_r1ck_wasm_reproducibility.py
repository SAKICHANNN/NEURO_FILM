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

P221A_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/p221a_r1ck_formal_commit_reproducibility_v1.json"
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


def test_corrected_contract_separates_formal_and_evidence_commits() -> None:
    config = json.loads(P221A_CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["experiment_id"] == "P221A"
    assert config["producer"]["commit"] == (
        "8c66ce51094cab760517301c2b495898482c5bd6"
    )
    assert config["producer"]["evidence_commit"] != config["producer"]["commit"]


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


def test_committed_evidence_records_exact_fail_closed_facts() -> None:
    evidence = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/evidence/P221_R1CK_WASM_REPRODUCIBILITY_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "FAIL_CLOSED_R1CK_CONSUMER_REPRODUCIBILITY"
    assert evidence["consumer"]["forward_report_sha256"] == evidence["consumer"][
        "reverse_report_sha256"
    ]
    assert evidence["exact_mechanism_facts"]["wasm_module_identity_exact"] is True
    assert evidence["exact_mechanism_facts"]["runtime_output_identity_exact"] is True
    assert evidence["failed_gates"] == [
        "each_consumer_report_equals_frozen_producer_report_sha256",
        "stable_identity_exact",
    ]


def test_corrected_evidence_preserves_v1_failure_and_exact_pass() -> None:
    evidence = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/evidence/P221A_R1CK_FORMAL_COMMIT_REPRODUCIBILITY_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "PASS_PRIVATE_R1CK_CONSUMER_REPRODUCIBILITY"
    assert evidence["consumer"]["forward_report_sha256"] == evidence["consumer"][
        "reverse_report_sha256"
    ]
    assert evidence["mechanism_facts"]["formal_report_identity_exact"] is True
    assert evidence["mechanism_facts"]["stable_identity_exact"] is True
    assert evidence["relationship_to_p221_v1"]["p221_v1_status_unchanged"].startswith(
        "FAIL_CLOSED"
    )
    assert evidence["relationship_to_p221_v1"]["not_a_gate_relaxation"] is True
