from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "evidence" / "U7_2S_PRODUCT_YAML_RUNTIME_DECOUPLING_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2s_evidence_binds_sources_and_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_U7_2S_PRODUCT_YAML_RUNTIME_DECOUPLING"
    for path, binding in evidence["bindings"]["files"].items():
        assert binding["bytes"] > 0
        assert_historical_evidence_binding(
            ROOT,
            {"path": path, "sha256": binding["sha256"]},
        )

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.stat().st_size == reverse.stat().st_size == reports["bytes_each"]
    assert _sha256(forward) == _sha256(reverse) == reports["sha256"]
    assert forward.read_bytes() == reverse.read_bytes()

    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["status"] == evidence["result"]["formal_status"] == "PASS"
    assert all(report["gates"].values())
    assert len(report["gates"]) == evidence["result"]["formal_gate_count"]
    assert all(row["nested_status"] == "FAIL_CLOSED" for row in report["environments"])
    assert all(
        row["nested_failed_gates"] == ["valid_numeric_boundaries_execute"]
        for row in report["environments"]
    )
    assert all(
        row["import_probe"]["omegaconf_loaded"] is False
        and row["import_probe"]["antlr4_loaded"] is False
        for row in report["environments"]
    )
