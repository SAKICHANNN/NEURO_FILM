from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_19C_PRODUCT_BOUNDED_MEMORY_POLICY_RESULT.json"


def test_u7_19c_evidence_preserves_the_formal_negative_and_mechanism_signal() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_U7_19C_FROZEN_VISIBLE_DIMENSION_MISMATCH"
    assert evidence["gates"]["full_resolution_exact"] is False
    assert evidence["gates"]["all_workers_within_resource_limits"] is True
    assert evidence["gates"]["output_and_replay_exact"] is True
    assert evidence["formal_report"]["reverse_not_run"]
    assert evidence["claim"]["formal_u7_19c_promotion"] is False
    assert evidence["claim"]["product_code_retained_as_safe_internal_policy"] is True
    assert all(
        row["peak_process_tree_rss_bytes"] <= 16 * 1024**3
        and row["replay_byte_exact"]
        and [row["visible_output_height"], row["visible_output_width"]] == [8750, 11664]
        for row in evidence["observations"]
    )


def test_u7_19c_evidence_bindings_and_formal_report_are_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for binding in evidence["bindings"].values():
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    report = ROOT / evidence["formal_report"]["path"]
    assert report.stat().st_size == evidence["formal_report"]["bytes"]
    assert (
        hashlib.sha256(report.read_bytes()).hexdigest()
        == evidence["formal_report"]["sha256"]
    )
