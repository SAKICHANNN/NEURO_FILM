from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repid10_evidence_binds_frozen_execution() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/U5_R2REPID10_CANONCGT_BOUNDED_BASIS_SELECTOR_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    bindings = {
        "config_sha256": "configs/u5_r2repid10_canoncgt_bounded_basis_selector_v1.json",
        "core_sha256": "src/eval/canoncgt_bounded_basis_selector.py",
        "runner_sha256": "scripts/run_u5_r2repid10_canoncgt_bounded_basis_selector.py",
        "test_sha256": "tests/test_u5_r2repid10_canoncgt_bounded_basis_selector.py",
    }
    for key, relative in bindings.items():
        assert evidence["frozen_inputs"][key] == _sha(ROOT / relative)
    assert evidence["status"] == "FAIL_CLOSED_BOUNDED_BASIS_SELECTOR"
    assert evidence["replay"]["full_report_byte_exact"] is True
    assert evidence["metrics"]["selection_audit_agreement_references"] == 9
    assert evidence["metrics"]["passing_references"] == 0


def test_repid10_raw_reports_are_exact_and_preserve_zero_application_reads() -> None:
    first = ROOT / "outputs/eval/u5_r2repid10/formal_a.json"
    second = ROOT / "outputs/eval/u5_r2repid10/formal_b.json"
    report = json.loads(first.read_text(encoding="utf-8"))
    assert _sha(first) == _sha(second)
    assert (
        _sha(first)
        == "a567b52789b114f61fd007de9619a32be35ecb3148f2bc6f0eeceec1d6b2532e"
    )
    assert report["application_source_reads"] == 0
    assert report["decision"] == "FAIL_CLOSED_BOUNDED_BASIS_SELECTOR"
