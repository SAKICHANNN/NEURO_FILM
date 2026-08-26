from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P247_ACESCG_OPENEXR_24MP_RESOURCE_RESULT.json"
REPORT = ROOT / "outputs/eval/p247_acescg_openexr_24mp_resources_v1/formal_result.json"


def test_p247_evidence_binds_resource_failure_without_losing_exactness() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_ACESCG_OPENEXR_24MP_RESOURCES"
    assert report["decision"] == evidence["status"]
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == evidence[
        "formal_execution"
    ]["formal_report_sha256"]
    assert report["stable_identity"] == evidence["formal_execution"][
        "stable_identity"
    ]
    assert report["gates"]["worker_peak_rss"] is False
    assert all(
        report["gates"][name]
        for name in report["gates"]
        if name != "worker_peak_rss"
    )


def test_p247_boundary_preserves_p246_and_closes_only_full_frame_24mp() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["propagation"]["p246_exact_small_probe_intake_remains_valid"]
    assert evidence["propagation"]["r1do_r1dp_correctness_and_parity_remain_valid"]
    assert evidence["propagation"]["exact_full_frame_24mp_python_file_api_closed"]
    assert evidence["rights_and_product"]["capability_or_product_mapping"] is False
