from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.eval.local_halation_manifest_audit import _candidate_facts

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3z_local_halation_manifest_audit_v1.json"
RUNNER = ROOT / "scripts/run_u6_p3z_local_halation_manifest_audit.py"


def test_p3z_contract_is_bounded_and_metadata_only() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    scope = contract["scope"]
    assert contract["status"] == "contract_frozen_before_content_scan"
    assert scope["extensions"] == [".json", ".jsonl"]
    assert scope["maximum_files"] == 256
    assert scope["maximum_total_bytes"] == 64 * 1024 * 1024
    assert scope["pixel_reads_allowed"] is False


def test_p3z_contract_requires_all_protocol_fields() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "contains every P3Y required_row_field" in contract["candidate_rule"]
    forbidden = " ".join(contract["forbidden"])
    assert "partial key overlap" in forbidden
    assert "image, PDF, archive or media" in forbidden


def test_p3z_candidate_rule_does_not_promote_partial_rows() -> None:
    required = {"a", "b", "c"}
    facts = _candidate_facts([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}], required)
    assert facts["maximum_required_field_overlap_count"] == 3
    assert facts["exact_candidate_node_indices"] == [1]
    assert facts["exact_candidate_count"] == 1


def test_p3z_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
