from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repid11_evidence_binds_frozen_execution() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/U5_R2REPID11_CANONCGT_EXTENDED_TRIANGULAR_TRANSPORT_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    bindings = {
        "config_sha256": "configs/u5_r2repid11_canoncgt_extended_triangular_transport_v1.json",
        "core_sha256": "src/eval/canoncgt_extended_triangular_transport.py",
        "runner_sha256": "scripts/run_u5_r2repid11_canoncgt_extended_triangular_transport.py",
        "test_sha256": "tests/test_u5_r2repid11_canoncgt_extended_triangular_transport.py",
    }
    for key, relative in bindings.items():
        assert evidence["frozen_inputs"][key] == _sha(ROOT / relative)
    assert evidence["status"] == "FAIL_CLOSED_EXTENDED_TRIANGULAR_TRANSPORT"
    assert evidence["metrics"]["absolute_error_passing_references"] == 9
    assert evidence["metrics"]["jacobian_gate_passing_references"] == 0


def test_repid11_raw_reports_are_exact_and_target_unread() -> None:
    first = ROOT / "outputs/eval/u5_r2repid11/formal_a.json"
    second = ROOT / "outputs/eval/u5_r2repid11/formal_b.json"
    report = json.loads(first.read_text(encoding="utf-8"))
    assert _sha(first) == _sha(second)
    assert (
        _sha(first)
        == "8a993216bdc35054e49f7b65e23a0e4775eda9fd267f5a4f58c473217cd667d8"
    )
    assert report["application_source_reads"] == 0
    assert report["decision"] == "FAIL_CLOSED_EXTENDED_TRIANGULAR_TRANSPORT"
