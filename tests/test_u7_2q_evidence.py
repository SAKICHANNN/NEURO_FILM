from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2Q_PRODUCT_MINIMAL_ENVIRONMENT_RESULT.json"


def test_u7_2q_evidence_binds_preproduct_binary_wheel_stop() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_BINARY_WHEEL_GAP_BEFORE_PRODUCT_EXECUTION"
    )
    for path, binding in evidence["bindings"]["files"].items():
        assert_historical_evidence_binding(
            ROOT,
            {"path": path, "sha256": binding["sha256"]},
        )

    formal = evidence["formal_report"]
    report_path = ROOT / formal["path"]
    payload = report_path.read_bytes()
    assert len(payload) == formal["bytes"] == 2510
    assert hashlib.sha256(payload).hexdigest() == formal["sha256"]
    report = json.loads(payload)
    assert report["status"] == formal["status"] == "FAIL_CLOSED"
    assert len(report["environments"]) == 2
    assert all(row["create_returncode"] == 0 for row in report["environments"])
    assert all(row["install_returncode"] == 1 for row in report["environments"])
    assert all(row["nested_report_bytes"] == 0 for row in report["environments"])
    assert report["gates"]["owned_runtime_residue_zero"] is True
    assert report["gates"]["source_locks_exact"] is True

    result = evidence["result"]
    assert result["product_execution_count"] == 0
    assert result["product_image_or_recipe_write_count"] == 0
    assert result["replay_not_run_due_to_stop_rule"] is True
    assert (
        evidence["retained_boundaries"]["manifest_or_version_rescue_authorized"]
        is False
    )
